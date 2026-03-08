import glob
import os
import shutil
import time
import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from module.memory_cleanup import release_torch_resources


CLUSTERING_MANIFEST_FILENAME = "clustering_slices.csv"
CLUSTERING_EMBEDDINGS_FILENAME = "clustering_embeddings.npz"
SLICE_MANIFEST_FILENAMES = ("whisper_slices.csv", "subtitle_slices.csv")


def clustering_embeddings_store_path_from_manifest(manifest_path):
    manifest_path = os.path.abspath(manifest_path)
    data_dir = os.path.dirname(manifest_path)
    return os.path.join(data_dir, CLUSTERING_EMBEDDINGS_FILENAME)


def clustering_embeddings_store_path_from_wav_folder(wav_folder):
    data_dir = os.path.dirname(os.path.abspath(wav_folder))
    return os.path.join(data_dir, CLUSTERING_EMBEDDINGS_FILENAME)


def clustering_embeddings_store_path_from_result_folder(result_folder):
    data_dir = os.path.dirname(os.path.abspath(result_folder))
    return os.path.join(data_dir, CLUSTERING_EMBEDDINGS_FILENAME)


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return default


def _load_slice_metadata_by_filename(data_dir):
    metadata = {}
    for manifest_name in SLICE_MANIFEST_FILENAMES:
        manifest_path = os.path.join(data_dir, manifest_name)
        if not os.path.exists(manifest_path):
            continue

        try:
            with open(manifest_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    filename = (row.get("filename") or "").strip()
                    if not filename:
                        continue
                    metadata[filename] = {
                        "index": str(row.get("index") or "").strip(),
                        "timestamp_start": str(row.get("timestamp_start") or "").strip(),
                        "timestamp_end": str(row.get("timestamp_end") or "").strip(),
                        "transcript": str(row.get("transcript") or "").strip(),
                    }
        except Exception as exc:
            print(f"[WARN] Failed to read slice manifest: {manifest_path}. error={exc}")

    return metadata


def _auto_manifest_scan_workers():
    cpu_count = os.cpu_count() or 4
    return max(2, min(16, cpu_count))


def _scan_cluster_directory(cluster_name, cluster_path):
    scanned_rows = []
    try:
        with os.scandir(cluster_path) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                file_name = entry.name
                if not file_name.lower().endswith(".wav"):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    mtime = 0.0
                scanned_rows.append((file_name, cluster_name, mtime))
    except Exception as exc:
        return cluster_name, [], str(exc)
    return cluster_name, scanned_rows, None


def _collect_cluster_assignments(
    destination_folder,
    include_set=None,
    manifest_scan_workers=None,
    prefer_process_pool=True,
):
    assignments = {}
    if not os.path.isdir(destination_folder):
        return assignments

    cluster_dirs = []
    with os.scandir(destination_folder) as entries:
        for entry in entries:
            if entry.is_dir():
                cluster_dirs.append((entry.name, entry.path))
    cluster_dirs.sort(key=lambda item: item[0])
    if not cluster_dirs:
        return assignments

    def _consume(scanned_rows):
        for file_name, cluster_name, mtime in scanned_rows:
            if include_set is not None and file_name not in include_set:
                continue
            prev = assignments.get(file_name)
            if prev is None or mtime >= prev[1]:
                assignments[file_name] = (cluster_name, mtime)

    worker_count = _safe_int(manifest_scan_workers, _auto_manifest_scan_workers())
    use_process_pool = bool(prefer_process_pool) and worker_count > 1 and len(cluster_dirs) > 1

    if use_process_pool:
        max_workers = max(1, min(worker_count, len(cluster_dirs)))
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_scan_cluster_directory, name, path) for name, path in cluster_dirs]
                for fut in as_completed(futures):
                    cluster_name, scanned_rows, err = fut.result()
                    if err:
                        print(f"[WARN] Failed to scan cluster directory: {cluster_name}. error={err}")
                        continue
                    _consume(scanned_rows)
            return assignments
        except Exception as exc:
            print(f"[WARN] ProcessPool scan failed, fallback to single process. error={exc}")

    for cluster_name, cluster_path in cluster_dirs:
        _, scanned_rows, err = _scan_cluster_directory(cluster_name, cluster_path)
        if err:
            print(f"[WARN] Failed to scan cluster directory: {cluster_name}. error={err}")
            continue
        _consume(scanned_rows)
    return assignments


def export_clustering_manifest(
    wav_folder,
    destination_folder,
    included_filenames=None,
    manifest_scan_workers=None,
    prefer_process_pool=True,
):
    data_dir = os.path.dirname(os.path.abspath(wav_folder))
    manifest_path = os.path.join(data_dir, CLUSTERING_MANIFEST_FILENAME)
    metadata_by_file = _load_slice_metadata_by_filename(data_dir)

    include_set = None
    if included_filenames is not None:
        include_set = {os.path.basename(name) for name in included_filenames}

    assignments = _collect_cluster_assignments(
        destination_folder=destination_folder,
        include_set=include_set,
        manifest_scan_workers=manifest_scan_workers,
        prefer_process_pool=prefer_process_pool,
    )

    rows = []
    for file_name in sorted(assignments):
        cluster_name, _ = assignments[file_name]
        meta = metadata_by_file.get(file_name, {})
        rows.append(
            [
                file_name,
                meta.get("index", ""),
                meta.get("timestamp_start", ""),
                meta.get("timestamp_end", ""),
                meta.get("transcript", ""),
                cluster_name,
            ]
        )

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "index", "timestamp_start", "timestamp_end", "transcript", "cluster_dir"]
        )
        writer.writerows(rows)

    print(f"[INFO] Saved clustering manifest: {manifest_path} (rows={len(rows)}).")
    return manifest_path, len(rows)


def refresh_clustering_manifest_cluster_dirs(
    manifest_path,
    destination_folder,
    append_missing_rows=True,
    manifest_scan_workers=None,
    prefer_process_pool=True,
):
    manifest_path = os.path.abspath(manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if "filename" not in fieldnames:
        raise ValueError(f"'filename' column is missing in manifest: {manifest_path}")
    if "cluster_dir" not in fieldnames:
        fieldnames.append("cluster_dir")

    data_dir = os.path.dirname(manifest_path)
    metadata_by_file = _load_slice_metadata_by_filename(data_dir)

    assignments = _collect_cluster_assignments(
        destination_folder=destination_folder,
        include_set=None,
        manifest_scan_workers=manifest_scan_workers,
        prefer_process_pool=prefer_process_pool,
    )

    seen_names = set()
    changed_rows = 0
    missing_rows = 0
    for row in rows:
        file_name = str(row.get("filename") or "").strip()
        if not file_name:
            continue
        seen_names.add(file_name)

        new_cluster_dir = assignments.get(file_name, ("", 0.0))[0]
        old_cluster_dir = str(row.get("cluster_dir") or "").strip()
        if old_cluster_dir != new_cluster_dir:
            changed_rows += 1
        row["cluster_dir"] = new_cluster_dir
        if not new_cluster_dir:
            missing_rows += 1

    added_rows = 0
    if append_missing_rows:
        for file_name in sorted(assignments):
            if file_name in seen_names:
                continue
            cluster_name, _ = assignments[file_name]
            meta = metadata_by_file.get(file_name, {})
            row = {field: "" for field in fieldnames}
            row["filename"] = file_name
            if "index" in row:
                row["index"] = meta.get("index", "")
            if "timestamp_start" in row:
                row["timestamp_start"] = meta.get("timestamp_start", "")
            if "timestamp_end" in row:
                row["timestamp_end"] = meta.get("timestamp_end", "")
            if "transcript" in row:
                row["transcript"] = meta.get("transcript", "")
            row["cluster_dir"] = cluster_name
            rows.append(row)
            added_rows += 1

    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "[INFO] Refreshed cluster_dir in manifest: "
        f"{manifest_path} (rows={len(rows)}, changed_rows={changed_rows}, "
        f"added_rows={added_rows}, missing_rows={missing_rows})."
    )
    return manifest_path, len(rows), changed_rows, added_rows, missing_rows


def _normalize_embedding_vector_np(vector):
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return None
    return (arr / norm).astype(np.float32)


def load_clustering_embeddings_store(store_path):
    store_path = os.path.abspath(store_path)
    if not os.path.exists(store_path):
        return {}

    try:
        with np.load(store_path, allow_pickle=False) as data:
            filenames = data.get("filenames")
            embeddings = data.get("embeddings")

            if filenames is None or embeddings is None:
                print(f"[WARN] Invalid embeddings store (missing keys): {store_path}")
                return {}
            if embeddings.ndim != 2 or filenames.shape[0] != embeddings.shape[0]:
                print(f"[WARN] Invalid embeddings store (shape mismatch): {store_path}")
                return {}

            loaded = {}
            for idx, raw_name in enumerate(filenames.tolist()):
                file_name = str(raw_name or "").strip()
                if not file_name:
                    continue
                vec = _normalize_embedding_vector_np(embeddings[idx])
                if vec is None:
                    continue
                loaded[file_name] = vec
            return loaded
    except Exception as exc:
        print(f"[WARN] Failed to load embeddings store: {store_path}. error={exc}")
        return {}


def save_clustering_embeddings_store(store_path, embeddings_by_filename):
    store_path = os.path.abspath(store_path)
    os.makedirs(os.path.dirname(store_path), exist_ok=True)

    cleaned = []
    dim = None
    for file_name in sorted(embeddings_by_filename.keys()):
        clean_name = str(file_name or "").strip()
        if not clean_name:
            continue
        vec = _normalize_embedding_vector_np(embeddings_by_filename[file_name])
        if vec is None:
            continue
        if dim is None:
            dim = vec.shape[0]
        if vec.shape[0] != dim:
            print(
                f"[WARN] Skip embedding due to dim mismatch: filename={clean_name} "
                f"dim={vec.shape[0]} expected={dim}"
            )
            continue
        cleaned.append((clean_name, vec))

    if not cleaned:
        filenames_arr = np.array([], dtype=np.str_)
        embeddings_arr = np.empty((0, 0), dtype=np.float32)
    else:
        filenames_arr = np.array([name for name, _ in cleaned], dtype=np.str_)
        embeddings_arr = np.stack([vec for _, vec in cleaned], axis=0).astype(np.float32)

    tmp_path = f"{store_path}.tmp"
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            filenames=filenames_arr,
            embeddings=embeddings_arr,
        )
    os.replace(tmp_path, store_path)

    return len(cleaned)


def upsert_clustering_embeddings_store(
    store_path,
    new_embeddings_by_filename,
    existing_embeddings=None,
    skip_existing=True,
):
    merged = dict(existing_embeddings) if existing_embeddings is not None else load_clustering_embeddings_store(store_path)

    added = 0
    updated = 0
    for file_name, vector in new_embeddings_by_filename.items():
        clean_name = str(file_name or "").strip()
        if not clean_name:
            continue
        if skip_existing and clean_name in merged:
            continue
        vec = _normalize_embedding_vector_np(vector)
        if vec is None:
            continue
        if clean_name in merged:
            updated += 1
        else:
            added += 1
        merged[clean_name] = vec

    if added > 0 or updated > 0 or (existing_embeddings is not None and not os.path.exists(os.path.abspath(store_path))):
        save_clustering_embeddings_store(store_path, merged)

    return merged, added, updated


def _embeddings_tensor_to_filename_vectors(embeddings, valid_wav_files):
    if embeddings is None or not valid_wav_files:
        return {}

    vectors = {}
    try:
        emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    except Exception:
        return {}

    for idx, wav_path in enumerate(valid_wav_files):
        if idx >= emb_np.shape[0]:
            break
        file_name = os.path.basename(str(wav_path))
        if not file_name:
            continue
        vec = _normalize_embedding_vector_np(emb_np[idx])
        if vec is None:
            continue
        vectors[file_name] = vec
    return vectors


def sync_clustering_embeddings_store_for_manifest(
    manifest_path,
    destination_folder,
    cache_dir="./model_cache",
    max_audio_length=10.0,
    min_duration=0.5,
    max_duration=10.0,
    target_sr=16000,
    use_half=False,
    audio_workers=None,
    embedding_batch_size=32,
    prefer_process_pool=True,
    manifest_scan_workers=None,
):
    manifest_path = os.path.abspath(manifest_path)
    destination_folder = os.path.abspath(destination_folder)
    store_path = clustering_embeddings_store_path_from_manifest(manifest_path)

    embeddings_by_filename = load_clustering_embeddings_store(store_path)
    assignments = _collect_cluster_assignments(
        destination_folder=destination_folder,
        include_set=None,
        manifest_scan_workers=manifest_scan_workers,
        prefer_process_pool=True,
    )

    file_to_path = {}
    for file_name, (cluster_name, _mtime) in assignments.items():
        candidate = os.path.join(destination_folder, cluster_name, file_name)
        if os.path.isfile(candidate):
            file_to_path[file_name] = candidate

    expected_files = len(file_to_path)
    missing_names = [name for name in sorted(file_to_path.keys()) if name not in embeddings_by_filename]
    missing_set = set(missing_names)
    added_embeddings = 0

    if missing_names:
        missing_paths = [file_to_path[name] for name in missing_names]
        embeddings, valid_wav_files, _noise_files, _timing = compute_embeddings_for_files(
            wav_files=missing_paths,
            max_audio_length=max_audio_length,
            min_duration=min_duration,
            max_duration=max_duration,
            cache_dir=cache_dir,
            target_sr=target_sr,
            use_half=use_half,
            audio_workers=audio_workers,
            embedding_batch_size=embedding_batch_size,
            prefer_process_pool=prefer_process_pool,
        )

        computed_vectors = _embeddings_tensor_to_filename_vectors(embeddings, valid_wav_files)
        if computed_vectors:
            filtered_vectors = {name: vec for name, vec in computed_vectors.items() if name in missing_set}
            embeddings_by_filename, added_embeddings, _updated = upsert_clustering_embeddings_store(
                store_path=store_path,
                new_embeddings_by_filename=filtered_vectors,
                existing_embeddings=embeddings_by_filename,
                skip_existing=True,
            )
    elif not os.path.exists(store_path):
        # Create an empty store once so downstream code has a stable location.
        save_clustering_embeddings_store(store_path, embeddings_by_filename)

    available_files = sum(1 for name in file_to_path if name in embeddings_by_filename)
    missing_after = max(0, expected_files - available_files)

    print(
        "[INFO] Synced local embeddings store: "
        f"{store_path} (expected_files={expected_files}, available={available_files}, "
        f"added={added_embeddings}, missing_after={missing_after})."
    )
    return store_path, expected_files, available_files, added_embeddings, missing_after


def _auto_audio_workers():
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def _auto_copy_workers():
    cpu_count = os.cpu_count() or 4
    return max(2, min(32, cpu_count * 2))


def _worker_init():
    # Avoid oversubscription when decoding in multiple processes.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _load_and_filter_audio(path, target_sr, max_samples, min_duration, max_duration):
    try:
        info = torchaudio.info(path)
        if info.sample_rate <= 0:
            return path, None, None, "invalid_sample_rate"

        duration_sec = float(info.num_frames) / float(info.sample_rate)
        if duration_sec < min_duration or duration_sec > max_duration:
            return path, None, duration_sec, "out_of_range"

        signal, sr = torchaudio.load(path)
        if signal.numel() == 0:
            return path, None, duration_sec, "empty_signal"

        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)

        if sr != target_sr:
            signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=target_sr)

        signal = signal[:, :max_samples]
        audio = signal.squeeze(0).contiguous().numpy().astype(np.float32)

        if audio.size == 0:
            return path, None, duration_sec, "empty_after_trim"

        return path, audio, duration_sec, None
    except Exception as exc:
        return path, None, None, str(exc)


def _normalize_embeddings(embeddings):
    return F.normalize(embeddings.float(), p=2, dim=1)


def chunked_cosine_similarity(embeddings, device, chunk_size=512, use_half=False):
    """
    Backward-compatible helper for cosine similarity matrix computation.
    """
    embeddings = _normalize_embeddings(embeddings)
    n = embeddings.shape[0]
    sim_mat = torch.zeros((n, n), dtype=torch.float32)

    for start_i in range(0, n, chunk_size):
        end_i = min(start_i + chunk_size, n)
        row = embeddings[start_i:end_i].to(device)
        if use_half:
            row = row.half()
        for start_j in range(0, n, chunk_size):
            end_j = min(start_j + chunk_size, n)
            col = embeddings[start_j:end_j].to(device)
            if use_half:
                col = col.half()
            sim_mat[start_i:end_i, start_j:end_j] = torch.matmul(row, col.t()).float().cpu()

    return sim_mat


def _compute_embeddings_from_wav_files(
    all_wav_files,
    max_audio_length=10.0,
    min_duration=0.5,
    max_duration=10.0,
    cache_dir="./model_cache",
    target_sr=16000,
    use_half=False,
    audio_workers=None,
    embedding_batch_size=32,
    prefer_process_pool=True,
):
    total_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}.")

    os.makedirs(cache_dir, exist_ok=True)
    torch.hub.set_dir(cache_dir)
    if not all_wav_files:
        return None, [], [], {"total_s": 0.0, "prepare_audio_s": 0.0, "embedding_s": 0.0}

    max_samples = int(max_audio_length * target_sr)
    audio_workers = _safe_int(audio_workers, _auto_audio_workers())
    embedding_batch_size = max(1, _safe_int(embedding_batch_size, 32))

    prepare_start = time.perf_counter()
    prepared = []

    if prefer_process_pool and audio_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=audio_workers, initializer=_worker_init) as ex:
                futures = [
                    ex.submit(
                        _load_and_filter_audio,
                        wav_path,
                        target_sr,
                        max_samples,
                        min_duration,
                        max_duration,
                    )
                    for wav_path in all_wav_files
                ]
                for fut in as_completed(futures):
                    prepared.append(fut.result())
        except Exception as exc:
            print(f"[WARN] ProcessPool preparation failed, fallback to ThreadPool. error={exc}")
            prepared = []

    if not prepared:
        with ThreadPoolExecutor(max_workers=audio_workers) as ex:
            futures = [
                ex.submit(
                    _load_and_filter_audio,
                    wav_path,
                    target_sr,
                    max_samples,
                    min_duration,
                    max_duration,
                )
                for wav_path in all_wav_files
            ]
            for fut in as_completed(futures):
                prepared.append(fut.result())

    # Restore input order for predictable output.
    path_to_idx = {p: i for i, p in enumerate(all_wav_files)}
    prepared.sort(key=lambda item: path_to_idx.get(item[0], 10**12))

    valid_wav_files = []
    audio_arrays = []
    for path, audio, duration_sec, error in prepared:
        if audio is None:
            if error == "out_of_range":
                print(f"[INFO] Excluding {path}: duration {duration_sec:.2f}s out of range.")
            else:
                print(f"[WARN] Excluding {path}: {error}")
            continue
        valid_wav_files.append(path)
        audio_arrays.append(audio)
    prepared.clear()

    prepare_s = time.perf_counter() - prepare_start
    if not audio_arrays:
        print("[WARN] No valid audio files found after filtering.")
        return None, [], [], {"prepare_audio_s": prepare_s, "total_s": time.perf_counter() - total_start}

    print("[INFO] Loading ReDimNet model from GitHub repository.")
    model = None
    embed_start = time.perf_counter()
    embedding_chunks = []
    batch = None
    emb = None
    try:
        model = torch.hub.load(
            "IDRnD/ReDimNet",
            "ReDimNet",
            model_name="b6",
            train_type="ft_lm",
            dataset="vox2",
            source="github",
            force_reload=False,
            skip_validation=True,
        )
        model.to(device)
        model.eval()
        if use_half:
            model.half()
        print("[INFO] ReDimNet model loaded successfully.")

        with torch.no_grad():
            for i in range(0, len(audio_arrays), embedding_batch_size):
                batch_audio = audio_arrays[i : i + embedding_batch_size]
                max_len = max(x.shape[0] for x in batch_audio)
                batch = torch.zeros((len(batch_audio), max_len), dtype=torch.float32)
                for row_idx, arr in enumerate(batch_audio):
                    n = arr.shape[0]
                    batch[row_idx, :n] = torch.from_numpy(arr)

                batch = batch.to(device)
                if use_half:
                    batch = batch.half()

                emb = model(batch)
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                embedding_chunks.append(emb.float().cpu())
                batch = None
                emb = None
    finally:
        batch = None
        emb = None
        release_torch_resources("clustering_embeddings", [model])
        model = None

    audio_arrays.clear()
    embeddings = torch.cat(embedding_chunks, dim=0)
    embedding_chunks.clear()
    embeddings = _normalize_embeddings(embeddings)
    embed_s = time.perf_counter() - embed_start

    total_s = time.perf_counter() - total_start
    timing = {
        "prepare_audio_s": prepare_s,
        "embedding_s": embed_s,
        "total_s": total_s,
    }
    print(f"[INFO] Embedding complete. files={len(valid_wav_files)} timing={timing}")

    # noise_files kept for backward compatibility of downstream API.
    noise_files = []
    return embeddings, valid_wav_files, noise_files, timing


def compute_embeddings(
    directory,
    max_audio_length=10.0,
    min_duration=0.5,
    max_duration=10.0,
    cache_dir="./model_cache",
    target_sr=16000,
    use_half=False,
    audio_workers=None,
    embedding_batch_size=32,
    prefer_process_pool=True,
):
    """
    Load WAV files from a directory, filter by duration, compute embeddings using ReDimNet.
    Returns normalized embeddings and valid file paths.
    """
    all_wav_files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    print(f"[INFO] Found {len(all_wav_files)} WAV files in directory: {directory}.")
    return _compute_embeddings_from_wav_files(
        all_wav_files=all_wav_files,
        max_audio_length=max_audio_length,
        min_duration=min_duration,
        max_duration=max_duration,
        cache_dir=cache_dir,
        target_sr=target_sr,
        use_half=use_half,
        audio_workers=audio_workers,
        embedding_batch_size=embedding_batch_size,
        prefer_process_pool=prefer_process_pool,
    )


def compute_embeddings_for_files(
    wav_files,
    max_audio_length=10.0,
    min_duration=0.5,
    max_duration=10.0,
    cache_dir="./model_cache",
    target_sr=16000,
    use_half=False,
    audio_workers=None,
    embedding_batch_size=32,
    prefer_process_pool=True,
):
    """
    Compute embeddings for an explicit WAV path list.
    Returns normalized embeddings and valid file paths.
    """
    seen = set()
    all_wav_files = []
    for path in wav_files or []:
        abs_path = os.path.abspath(str(path))
        if abs_path in seen:
            continue
        if not abs_path.lower().endswith(".wav"):
            continue
        if not os.path.isfile(abs_path):
            continue
        all_wav_files.append(abs_path)
        seen.add(abs_path)

    all_wav_files.sort()
    print(f"[INFO] Found {len(all_wav_files)} WAV files in explicit input list.")
    return _compute_embeddings_from_wav_files(
        all_wav_files=all_wav_files,
        max_audio_length=max_audio_length,
        min_duration=min_duration,
        max_duration=max_duration,
        cache_dir=cache_dir,
        target_sr=target_sr,
        use_half=use_half,
        audio_workers=audio_workers,
        embedding_batch_size=embedding_batch_size,
        prefer_process_pool=prefer_process_pool,
    )


def compute_embeddings_and_distance(
    directory,
    max_audio_length=4.0,
    use_half=False,
    chunk_size=512,
    min_duration=0.5,
    max_duration=10.0,
    cache_dir="./model_cache",
):
    """
    Backward-compatible API that also returns a cosine distance matrix.
    """
    embeddings, valid_wav_files, noise_files, _ = compute_embeddings(
        directory=directory,
        max_audio_length=max_audio_length,
        min_duration=min_duration,
        max_duration=max_duration,
        cache_dir=cache_dir,
        target_sr=16000,
        use_half=use_half,
        audio_workers=None,
        embedding_batch_size=32,
        prefer_process_pool=True,
    )

    if embeddings is None or embeddings.shape[0] == 0:
        return None, [], noise_files, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sim_mat = chunked_cosine_similarity(embeddings, device=device, chunk_size=chunk_size, use_half=use_half)
    sim_mat = torch.clamp(sim_mat, -1.0, 1.0)
    distance_matrix = ((1.0 - sim_mat.numpy()) / 2.0).astype(np.float64)
    return distance_matrix, valid_wav_files, noise_files, embeddings


def _fit_hdbscan_labels(
    embeddings_cpu,
    min_cluster_size=5,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    prefer_gpu_hdbscan=True,
):
    """
    Fit HDBSCAN using cuML on GPU when available, otherwise sklearn fallback.
    Returns labels (np.ndarray) and backend name.
    """
    x_np = embeddings_cpu.detach().cpu().numpy().astype(np.float32)

    if prefer_gpu_hdbscan and torch.cuda.is_available():
        try:
            import cupy as cp
            from cuml.cluster import HDBSCAN as CuHDBSCAN

            x_gpu = cp.asarray(x_np)
            model = CuHDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_epsilon=cluster_selection_epsilon,
            )
            labels_gpu = model.fit_predict(x_gpu)
            labels = cp.asnumpy(labels_gpu).astype(np.int64)
            return labels, "cuml"
        except Exception as exc:
            print(f"[WARN] cuML HDBSCAN unavailable, fallback to sklearn. error={exc}")

    try:
        from sklearn.cluster import HDBSCAN as SkHDBSCAN

        model = SkHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_epsilon=cluster_selection_epsilon,
        )
        labels = model.fit_predict(x_np).astype(np.int64)
        return labels, "sklearn"
    except Exception as exc:
        raise RuntimeError(f"No available HDBSCAN backend. error={exc}") from exc


def _compute_centroids_from_labels(x_cpu, labels_cpu, k):
    d = x_cpu.shape[1]
    centroids = torch.zeros((k, d), dtype=x_cpu.dtype)
    counts = torch.zeros(k, dtype=torch.long)

    centroids.index_add_(0, labels_cpu, x_cpu)
    counts.index_add_(0, labels_cpu, torch.ones_like(labels_cpu, dtype=torch.long))

    nonzero = counts > 0
    centroids[nonzero] = centroids[nonzero] / counts[nonzero].unsqueeze(1)
    centroids[nonzero] = _normalize_embeddings(centroids[nonzero])
    return centroids, counts


def _kmeans_vectorized_cpu(x_cpu, init_centroids_cpu, max_iters=100, tol=1e-4):
    centroids = _normalize_embeddings(init_centroids_cpu.clone())

    for iter_idx in range(1, max_iters + 1):
        similarity = torch.matmul(x_cpu, centroids.t())
        labels = torch.argmax(similarity, dim=1)

        new_centroids, counts = _compute_centroids_from_labels(x_cpu, labels, centroids.shape[0])
        empty = counts == 0
        if empty.any():
            new_centroids[empty] = centroids[empty]

        if torch.allclose(centroids, new_centroids, atol=tol):
            print(f"[INFO] CPU vectorized K-Means converged at iteration={iter_idx}.")
            centroids = new_centroids
            break

        centroids = new_centroids

    return labels, centroids


def _try_torch_kmeans(x_cpu, k, max_iters=100):
    """
    Try torch_kmeans on GPU. Returns (labels_cpu, backend_name) or (None, None) on failure.
    """
    if not torch.cuda.is_available():
        return None, None

    x_gpu = None
    model = None
    try:
        from torch_kmeans import KMeans

        x_gpu = x_cpu.to("cuda").unsqueeze(0)
        model = KMeans(n_clusters=k, max_iter=max_iters, verbose=False)

        labels = None
        if hasattr(model, "fit_predict"):
            pred = model.fit_predict(x_gpu)
            labels = pred
        else:
            out = model(x_gpu)
            if hasattr(out, "labels"):
                labels = out.labels
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                labels = out[0]

        if labels is None:
            return None, None

        labels = labels.squeeze(0).detach().to("cpu").long()
        if labels.ndim != 1 or labels.numel() != x_cpu.shape[0]:
            return None, None

        return labels, "torch_kmeans"
    except Exception as exc:
        print(f"[WARN] torch_kmeans unavailable, fallback to CPU vectorized K-Means. error={exc}")
        return None, None
    finally:
        x_gpu = None
        model = None
        release_torch_resources("clustering_kmeans", [])


def hdbscan_kmeans_clustering(
    embeddings,
    valid_wav_files,
    noise_files,
    destination_folder,
    hdbscan_min_cluster_size=5,
    hdbscan_min_samples=None,
    cluster_selection_epsilon=0.0,
    max_distance=0.2,
    kmeans_max_iters=100,
    prefer_gpu_hdbscan=True,
    prefer_gpu_kmeans=True,
    cpu_math_threads=None,
    copy_workers=None,
):
    """
    Cluster data using HDBSCAN, then refine non-noise points with K-Means.
    Final noise is enforced by both HDBSCAN and centroid distance threshold.
    """
    total_start = time.perf_counter()

    if embeddings is None or embeddings.shape[0] == 0:
        return finalize_clustering([], valid_wav_files, noise_files, destination_folder, copy_workers=copy_workers)

    if cpu_math_threads is not None:
        torch.set_num_threads(max(1, _safe_int(cpu_math_threads, torch.get_num_threads())))

    x_cpu = _normalize_embeddings(embeddings.cpu())

    hdbscan_start = time.perf_counter()
    base_labels, hdbscan_backend = _fit_hdbscan_labels(
        x_cpu,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prefer_gpu_hdbscan=prefer_gpu_hdbscan,
    )
    hdbscan_s = time.perf_counter() - hdbscan_start
    print(f"[INFO] HDBSCAN backend={hdbscan_backend}, clusters={set(base_labels.tolist())}")

    non_noise_indices = np.where(base_labels != -1)[0]
    if non_noise_indices.size == 0:
        print("[WARN] HDBSCAN found only noise.")
        print(f"[INFO] timing={{'hdbscan_s': {hdbscan_s:.3f}, 'total_s': {time.perf_counter() - total_start:.3f}}}")
        return finalize_clustering(base_labels, valid_wav_files, noise_files, destination_folder, copy_workers=copy_workers)

    non_noise_label_values = sorted(set(base_labels[non_noise_indices].tolist()))
    label_to_local = {label: i for i, label in enumerate(non_noise_label_values)}

    x_non_noise = x_cpu[non_noise_indices]
    local_labels_from_hdbscan = torch.tensor(
        [label_to_local[int(base_labels[idx])] for idx in non_noise_indices], dtype=torch.long
    )

    k = len(non_noise_label_values)
    init_centroids = []
    for original_label in non_noise_label_values:
        mask = torch.tensor(base_labels[non_noise_indices] == original_label, dtype=torch.bool)
        init_centroids.append(x_non_noise[mask].mean(dim=0))
    init_centroids = torch.stack(init_centroids, dim=0)

    kmeans_start = time.perf_counter()
    labels_local = None
    kmeans_backend = None

    if prefer_gpu_kmeans:
        labels_local, kmeans_backend = _try_torch_kmeans(x_non_noise, k, max_iters=kmeans_max_iters)

    if labels_local is None:
        labels_local, _ = _kmeans_vectorized_cpu(
            x_non_noise,
            init_centroids,
            max_iters=kmeans_max_iters,
            tol=1e-4,
        )
        kmeans_backend = "cpu_vectorized"

    centroids, _ = _compute_centroids_from_labels(x_non_noise, labels_local, k)
    picked_centroids = centroids[labels_local]
    cosine_dist = ((1.0 - (x_non_noise * picked_centroids).sum(dim=1).clamp(-1.0, 1.0)) / 2.0).cpu().numpy()
    distance_noise = cosine_dist > max_distance
    kmeans_s = time.perf_counter() - kmeans_start

    final_labels = np.full(x_cpu.shape[0], -1, dtype=np.int64)
    final_labels[non_noise_indices] = labels_local.cpu().numpy().astype(np.int64)
    final_labels[non_noise_indices[distance_noise]] = -1
    final_labels[base_labels == -1] = -1

    total_s = time.perf_counter() - total_start
    print(
        "[INFO] Clustering timing="
        f"{{'hdbscan_s': {hdbscan_s:.3f}, 'kmeans_s': {kmeans_s:.3f}, 'total_s': {total_s:.3f}}}, "
        f"hdbscan_backend={hdbscan_backend}, kmeans_backend={kmeans_backend}"
    )

    return finalize_clustering(final_labels, valid_wav_files, noise_files, destination_folder, copy_workers=copy_workers)


def dbscan_kmeans_clustering(
    distance_matrix,
    embeddings,
    valid_wav_files,
    noise_files,
    destination_folder,
    eps=0.05,
    min_samples=2,
    max_distance=0.2,
    max_iters=100,
):
    """
    Backward-compatible wrapper. DBSCAN path is replaced by HDBSCAN path.
    distance_matrix/eps are accepted for compatibility but not used.
    """
    _ = distance_matrix, eps
    return hdbscan_kmeans_clustering(
        embeddings=embeddings,
        valid_wav_files=valid_wav_files,
        noise_files=noise_files,
        destination_folder=destination_folder,
        hdbscan_min_cluster_size=max(2, min_samples),
        hdbscan_min_samples=min_samples,
        cluster_selection_epsilon=0.0,
        max_distance=max_distance,
        kmeans_max_iters=max_iters,
        prefer_gpu_hdbscan=True,
        prefer_gpu_kmeans=True,
    )


def cos_distance(a, b):
    """
    Backward-compatible cosine distance helper for 1D tensors.
    """
    a_n = F.normalize(a.float().unsqueeze(0), p=2, dim=1)
    b_n = F.normalize(b.float().unsqueeze(0), p=2, dim=1)
    sim = (a_n * b_n).sum(dim=1).clamp(-1.0, 1.0)
    dist = (1.0 - sim) / 2.0
    return float(dist.item())


def kmeans_with_noise(
    embeddings_cpu,
    centroids_cpu,
    k,
    distance_func,
    max_distance=0.2,
    max_iters=100,
):
    """
    Backward-compatible K-Means variant wrapper using vectorized CPU implementation.
    """
    _ = distance_func
    if embeddings_cpu is None or embeddings_cpu.shape[0] == 0:
        return centroids_cpu, [[] for _ in range(max(0, k))], np.array([], dtype=bool)

    x = _normalize_embeddings(embeddings_cpu.cpu())
    c = _normalize_embeddings(centroids_cpu.cpu())
    labels, centroids = _kmeans_vectorized_cpu(x, c, max_iters=max_iters, tol=1e-4)

    picked_centroids = centroids[labels]
    cosine_dist = ((1.0 - (x * picked_centroids).sum(dim=1).clamp(-1.0, 1.0)) / 2.0).cpu().numpy()
    noise_points = cosine_dist > max_distance

    idx_clusters = []
    for cluster_idx in range(k):
        idx_in_cluster = torch.where(labels == cluster_idx)[0].cpu().numpy().tolist()
        idx_clusters.append(idx_in_cluster)

    return centroids, idx_clusters, noise_points


def _copy_file(src_path, dest_path):
    try:
        shutil.copy(src_path, dest_path)
        return src_path, dest_path, None
    except Exception as exc:
        return src_path, dest_path, str(exc)


def finalize_clustering(labels, valid_wav_files, noise_files, destination_folder, copy_workers=None):
    """
    Create cluster folders and copy WAV files based on the clustering labels.
    """
    labels_total = list(labels) + ([-1] * len(noise_files))
    all_files = valid_wav_files + noise_files

    os.makedirs(destination_folder, exist_ok=True)
    print(f"[INFO] Creating destination folders in: {destination_folder}.")

    unique_labels = set(labels_total)
    for lbl in unique_labels:
        if lbl == -1:
            cluster_folder = os.path.join(destination_folder, "noise")
        else:
            cluster_folder = os.path.join(destination_folder, f"clustering_{lbl}")
        os.makedirs(cluster_folder, exist_ok=True)

    copy_tasks = []
    for path, lbl in zip(all_files, labels_total):
        if lbl == -1:
            cluster_folder = os.path.join(destination_folder, "noise")
        else:
            cluster_folder = os.path.join(destination_folder, f"clustering_{lbl}")
        file_name = os.path.basename(path)
        dest_path = os.path.join(cluster_folder, file_name)
        copy_tasks.append((path, dest_path))

    copy_workers = _safe_int(copy_workers, _auto_copy_workers())
    if copy_workers <= 1:
        for src_path, dest_path in copy_tasks:
            _, _, err = _copy_file(src_path, dest_path)
            if err:
                print(f"[WARN] Failed to copy {src_path} -> {dest_path}: {err}")
            else:
                print(f"[INFO] Copied file to: {dest_path}.")
    else:
        with ThreadPoolExecutor(max_workers=copy_workers) as ex:
            futures = [ex.submit(_copy_file, src, dst) for src, dst in copy_tasks]
            for fut in as_completed(futures):
                src_path, dest_path, err = fut.result()
                if err:
                    print(f"[WARN] Failed to copy {src_path} -> {dest_path}: {err}")
                else:
                    print(f"[INFO] Copied file to: {dest_path}.")

    print("[INFO] Clustering complete.")
    return labels_total


def remove_small_clusters(destination_folder, min_files=10):
    """
    Remove cluster folders that contain min_files or fewer WAV files.
    """
    cluster_folders = [
        os.path.join(destination_folder, f)
        for f in os.listdir(destination_folder)
        if os.path.isdir(os.path.join(destination_folder, f))
    ]

    for folder in cluster_folders:
        wav_files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        num_files = len(wav_files)
        if num_files <= min_files:
            print(f"[INFO] Deleting folder: {folder} (contains {num_files} files).")
            shutil.rmtree(folder)
        else:
            print(f"[INFO] Keeping folder: {folder} (contains {num_files} files).")

    print("[INFO] Cleanup complete.")


def clustering_for_main(
    wav_folder,
    output_folder,
    cache_dir,
    hdbscan_min_cluster_size=5,
    hdbscan_min_samples=None,
    cluster_selection_epsilon=0.0,
    max_distance=0.2,
    kmeans_max_iters=100,
    prefer_gpu_hdbscan=True,
    prefer_gpu_kmeans=True,
    audio_workers=None,
    embedding_batch_size=32,
    cpu_math_threads=None,
    copy_workers=None,
):
    """
    Main function to run clustering process.
    """
    print(f"[INFO] Starting clustering process for WAV folder: {wav_folder}.")
    embeddings = None
    valid_wav_files = []
    noise_files = []
    embed_timing = {}
    try:
        embeddings, valid_wav_files, noise_files, embed_timing = compute_embeddings(
            directory=wav_folder,
            max_audio_length=10.0,
            min_duration=0.5,
            max_duration=10.0,
            cache_dir=cache_dir,
            target_sr=16000,
            use_half=False,
            audio_workers=audio_workers,
            embedding_batch_size=embedding_batch_size,
            prefer_process_pool=True,
        )

        if embeddings is None or len(valid_wav_files) == 0:
            print("[WARN] No valid audio files found for clustering.")
            export_clustering_manifest(
                wav_folder=wav_folder,
                destination_folder=output_folder,
                included_filenames=[],
            )
            return

        embedding_store_path = clustering_embeddings_store_path_from_wav_folder(wav_folder)
        embedding_vectors = _embeddings_tensor_to_filename_vectors(embeddings, valid_wav_files)
        _, added_embeddings, _updated_embeddings = upsert_clustering_embeddings_store(
            store_path=embedding_store_path,
            new_embeddings_by_filename=embedding_vectors,
            existing_embeddings=None,
            skip_existing=True,
        )
        print(
            "[INFO] Synced local embeddings after clustering run: "
            f"{embedding_store_path} (batch_files={len(embedding_vectors)}, added={added_embeddings})."
        )

        print("[INFO] Performing HDBSCAN + K-Means clustering.")
        hdbscan_kmeans_clustering(
            embeddings=embeddings,
            valid_wav_files=valid_wav_files,
            noise_files=noise_files,
            destination_folder=output_folder,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            max_distance=max_distance,
            kmeans_max_iters=kmeans_max_iters,
            prefer_gpu_hdbscan=prefer_gpu_hdbscan,
            prefer_gpu_kmeans=prefer_gpu_kmeans,
            cpu_math_threads=cpu_math_threads,
            copy_workers=copy_workers,
        )

        print("[INFO] Removing small clusters (folders with insufficient files).")
        remove_small_clusters(destination_folder=output_folder, min_files=10)
        export_clustering_manifest(
            wav_folder=wav_folder,
            destination_folder=output_folder,
            included_filenames=[os.path.basename(p) for p in (valid_wav_files + noise_files)],
        )
        print(f"[INFO] Clustering process completed. embedding_timing={embed_timing}")
    finally:
        embeddings = None
        valid_wav_files = []
        noise_files = []
        release_torch_resources("clustering", [])
