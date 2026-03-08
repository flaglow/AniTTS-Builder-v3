import csv
import os
import re
import shutil
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from module.clustering import (
    clustering_embeddings_store_path_from_result_folder,
    compute_embeddings,
    load_clustering_embeddings_store,
    upsert_clustering_embeddings_store,
)


UNLABELED_CLUSTER_RE = re.compile(r"^clustering_(\d+)$", re.IGNORECASE)


def _cluster_numeric_suffix(cluster_name: str) -> int:
    match = UNLABELED_CLUSTER_RE.fullmatch(str(cluster_name or "").strip())
    if not match:
        return 10**9
    return int(match.group(1))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return None
    return (vector / norm).astype(np.float32)


def is_unlabeled_cluster(cluster_name: str) -> bool:
    return bool(UNLABELED_CLUSTER_RE.fullmatch(str(cluster_name or "").strip()))


def is_noise_cluster(cluster_name: str) -> bool:
    return str(cluster_name or "").strip().lower() == "noise"


def _scan_result_audio_paths(result_folder: str) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    if not os.path.isdir(result_folder):
        return mapping

    with os.scandir(result_folder) as cluster_entries:
        for cluster_entry in cluster_entries:
            if not cluster_entry.is_dir():
                continue
            cluster_name = cluster_entry.name
            file_map: Dict[str, str] = {}
            with os.scandir(cluster_entry.path) as file_entries:
                for file_entry in file_entries:
                    if not file_entry.is_file() or not file_entry.name.lower().endswith(".wav"):
                        continue
                    file_map[file_entry.name] = file_entry.path
            mapping[cluster_name] = file_map

    return mapping


def build_labeling_context(manifest_path: str, result_folder: str) -> Dict[str, Any]:
    manifest_path = os.path.abspath(manifest_path)
    result_folder = os.path.abspath(result_folder)

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            filename = str(row.get("filename") or "").strip()
            if not filename:
                continue
            rows.append(
                {
                    "filename": filename,
                    "index": str(row.get("index") or "").strip(),
                    "timestamp_start": str(row.get("timestamp_start") or "").strip(),
                    "timestamp_end": str(row.get("timestamp_end") or "").strip(),
                    "transcript": str(row.get("transcript") or "").strip(),
                    "cluster_dir": str(row.get("cluster_dir") or "").strip(),
                }
            )

    cluster_rows: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        cluster_rows[row["cluster_dir"]].append(row)

    unlabeled_clusters = []
    unlabeled_sizes = {}
    for cluster_name, items in cluster_rows.items():
        if is_unlabeled_cluster(cluster_name):
            unlabeled_clusters.append(cluster_name)
            unlabeled_sizes[cluster_name] = len(items)

    unlabeled_clusters.sort(key=lambda c: (-unlabeled_sizes.get(c, 0), _cluster_numeric_suffix(c), c))

    audio_paths = _scan_result_audio_paths(result_folder)

    return {
        "manifest_path": manifest_path,
        "result_folder": result_folder,
        "rows": rows,
        "cluster_rows": dict(cluster_rows),
        "audio_paths": audio_paths,
        "unlabeled_clusters": unlabeled_clusters,
        "unlabeled_sizes": unlabeled_sizes,
    }


def get_label_choices(context: Dict[str, Any], exclude_noise: bool = True) -> List[str]:
    cluster_names = set(context.get("cluster_rows", {}).keys()) | set(context.get("audio_paths", {}).keys())
    labels = []
    for cluster_name in sorted(cluster_names):
        clean = str(cluster_name or "").strip()
        if not clean:
            continue
        if is_unlabeled_cluster(clean):
            continue
        if exclude_noise and is_noise_cluster(clean):
            continue
        labels.append(clean)
    return labels


def select_next_unlabeled_cluster(
    context: Dict[str, Any], skipped_clusters: Optional[Iterable[str]] = None
) -> Tuple[Optional[str], int]:
    skipped = {str(x).strip() for x in (skipped_clusters or []) if str(x).strip()}
    ordered = []
    for cluster_name in context.get("unlabeled_clusters", []):
        if cluster_name in skipped:
            continue
        size = int(context.get("unlabeled_sizes", {}).get(cluster_name, 0))
        if size <= 0:
            continue
        ordered.append(cluster_name)

    if not ordered:
        return None, 0
    return ordered[0], len(ordered)


def _build_embedding_entry(vectors: Dict[str, np.ndarray]) -> Dict[str, Any]:
    clean_vectors: Dict[str, np.ndarray] = {}
    for file_name, vec in vectors.items():
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            continue
        normed = _normalize_vector(arr)
        if normed is None:
            continue
        clean_vectors[file_name] = normed

    if not clean_vectors:
        return {
            "vectors": {},
            "file_names": [],
            "matrix": np.empty((0, 0), dtype=np.float32),
            "centroid": None,
        }

    file_names = sorted(clean_vectors.keys())
    matrix = np.stack([clean_vectors[name] for name in file_names], axis=0).astype(np.float32)
    centroid = _normalize_vector(matrix.mean(axis=0))

    return {
        "vectors": clean_vectors,
        "file_names": file_names,
        "matrix": matrix,
        "centroid": centroid,
    }


def _default_embedding_loader(cluster_dir: str, embeddings_cache_dir: str) -> Dict[str, np.ndarray]:
    vectors: Dict[str, np.ndarray] = {}
    embeddings, valid_wav_files, _noise_files, _timing = compute_embeddings(
        directory=cluster_dir,
        max_audio_length=10.0,
        min_duration=0.5,
        max_duration=10.0,
        cache_dir=embeddings_cache_dir,
        target_sr=16000,
        use_half=False,
        audio_workers=None,
        embedding_batch_size=32,
        prefer_process_pool=True,
    )
    if embeddings is None or not valid_wav_files:
        return vectors

    emb_np = embeddings.detach().cpu().numpy().astype(np.float32)
    for idx, wav_path in enumerate(valid_wav_files):
        vectors[os.path.basename(wav_path)] = emb_np[idx]
    return vectors


def _get_local_store_state(context: Dict[str, Any], embedding_cache: Dict[str, Any]) -> Dict[str, Any]:
    state = embedding_cache.setdefault("local_store", {})
    store_path = clustering_embeddings_store_path_from_result_folder(context["result_folder"])
    store_mtime = os.path.getmtime(store_path) if os.path.exists(store_path) else None

    if (
        state.get("path") != store_path
        or state.get("mtime") != store_mtime
        or not isinstance(state.get("vectors"), dict)
    ):
        state = {
            "path": store_path,
            "mtime": store_mtime,
            "vectors": load_clustering_embeddings_store(store_path),
        }
        embedding_cache["local_store"] = state
    return state


def _get_cluster_embedding_entry(
    context: Dict[str, Any],
    cluster_name: str,
    embedding_cache: Dict[str, Any],
    embeddings_cache_dir: str,
    embedding_loader,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    clusters_cache = embedding_cache.setdefault("clusters", {})
    if cluster_name in clusters_cache:
        return clusters_cache[cluster_name], embedding_cache

    cluster_dir = os.path.join(context["result_folder"], cluster_name)
    local_state = _get_local_store_state(context, embedding_cache)
    local_vectors = local_state.get("vectors", {})

    cluster_audio_map = context.get("audio_paths", {}).get(cluster_name, {})
    cluster_files = sorted(cluster_audio_map.keys()) if cluster_audio_map else []

    vectors = {}
    missing_files = []
    if cluster_files:
        for file_name in cluster_files:
            vec = local_vectors.get(file_name)
            if vec is None:
                missing_files.append(file_name)
                continue
            vectors[file_name] = vec
    else:
        missing_files = []

    if (not cluster_files) or missing_files:
        computed_vectors = embedding_loader(cluster_dir, embeddings_cache_dir)
        if not cluster_files:
            vectors.update(computed_vectors)
        else:
            for file_name in missing_files:
                vec = computed_vectors.get(file_name)
                if vec is not None:
                    vectors[file_name] = vec

        # Persist newly computed embeddings into the local store so next recommendation can reuse them.
        if computed_vectors:
            filtered_new = {}
            if cluster_files:
                for file_name in missing_files:
                    if file_name in computed_vectors:
                        filtered_new[file_name] = computed_vectors[file_name]
            else:
                filtered_new = dict(computed_vectors)

            if filtered_new:
                merged_vectors, _added, _updated = upsert_clustering_embeddings_store(
                    store_path=local_state["path"],
                    new_embeddings_by_filename=filtered_new,
                    existing_embeddings=local_vectors,
                    skip_existing=True,
                )
                local_state["vectors"] = merged_vectors
                local_state["mtime"] = os.path.getmtime(local_state["path"]) if os.path.exists(local_state["path"]) else None

    entry = _build_embedding_entry(vectors)
    clusters_cache[cluster_name] = entry
    return entry, embedding_cache


def _choose_representative_samples(
    context: Dict[str, Any],
    cluster_name: str,
    embedding_entry: Dict[str, Any],
    sample_count: int,
) -> List[Dict[str, Any]]:
    rows = list(context.get("cluster_rows", {}).get(cluster_name, []))
    if not rows:
        return []

    vectors = embedding_entry.get("vectors", {})
    centroid = embedding_entry.get("centroid")

    scored = []
    for row in rows:
        file_name = row.get("filename", "")
        start = _safe_float(row.get("timestamp_start", ""), 0.0)
        end = _safe_float(row.get("timestamp_end", ""), 0.0)
        duration = max(0.0, end - start)

        sim = -2.0
        if centroid is not None and file_name in vectors:
            sim = float(np.dot(vectors[file_name], centroid))

        scored.append((sim, duration, file_name, row))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))

    selected_rows = []
    used_files = set()

    for sim, _duration, file_name, row in scored:
        if sim <= -1.5:
            continue
        selected_rows.append(row)
        used_files.add(file_name)
        if len(selected_rows) >= sample_count:
            break

    if len(selected_rows) < sample_count:
        fallback = []
        for row in rows:
            file_name = row.get("filename", "")
            if file_name in used_files:
                continue
            start = _safe_float(row.get("timestamp_start", ""), 0.0)
            end = _safe_float(row.get("timestamp_end", ""), 0.0)
            duration = max(0.0, end - start)
            fallback.append((duration, file_name, row))
        fallback.sort(key=lambda item: (-item[0], item[1]))
        for _duration, file_name, row in fallback:
            selected_rows.append(row)
            used_files.add(file_name)
            if len(selected_rows) >= sample_count:
                break

    output = []
    cluster_audio = context.get("audio_paths", {}).get(cluster_name, {})
    for row in selected_rows[:sample_count]:
        file_name = row.get("filename", "")
        start = _safe_float(row.get("timestamp_start", ""), 0.0)
        end = _safe_float(row.get("timestamp_end", ""), 0.0)
        output.append(
            {
                "cluster_name": cluster_name,
                "filename": file_name,
                "index": row.get("index", ""),
                "timestamp_start": row.get("timestamp_start", ""),
                "timestamp_end": row.get("timestamp_end", ""),
                "duration": max(0.0, end - start),
                "transcript": row.get("transcript", ""),
                "audio_path": cluster_audio.get(file_name, ""),
            }
        )

    return output


def recommend_label_for_cluster(
    context: Dict[str, Any],
    target_cluster: str,
    embedding_cache: Optional[Dict[str, Any]] = None,
    embedding_loader=None,
    embeddings_cache_dir: str = "./module/model/redimmet",
    top_k: int = 3,
    centroid_top_n: int = 8,
    low_score_threshold: float = 0.35,
    low_gap_threshold: float = 0.03,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if embedding_cache is None:
        embedding_cache = {"clusters": {}}
    if embedding_loader is None:
        embedding_loader = _default_embedding_loader

    target_entry, embedding_cache = _get_cluster_embedding_entry(
        context,
        target_cluster,
        embedding_cache,
        embeddings_cache_dir,
        embedding_loader,
    )

    target_matrix = target_entry.get("matrix")
    if target_matrix is None or target_matrix.size == 0:
        return (
            {
                "target_cluster": target_cluster,
                "recommended_label": "",
                "recommended_score": None,
                "candidate_scores": [],
                "confidence_warning": True,
                "confidence_reason": "대상 클러스터 임베딩이 없어 자동 추천을 생성할 수 없습니다.",
                "recommended_samples": [],
                "target_samples": [],
            },
            embedding_cache,
        )

    label_candidates = get_label_choices(context, exclude_noise=True)
    if not label_candidates:
        return (
            {
                "target_cluster": target_cluster,
                "recommended_label": "",
                "recommended_score": None,
                "candidate_scores": [],
                "confidence_warning": True,
                "confidence_reason": "라벨 후보 클러스터가 없어 자동 추천을 생성할 수 없습니다.",
                "recommended_samples": [],
                "target_samples": _choose_representative_samples(context, target_cluster, target_entry, 5),
            },
            embedding_cache,
        )

    target_centroid = target_entry.get("centroid")
    centroid_scores = []
    label_entries: Dict[str, Dict[str, Any]] = {}

    for label_cluster in label_candidates:
        label_entry, embedding_cache = _get_cluster_embedding_entry(
            context,
            label_cluster,
            embedding_cache,
            embeddings_cache_dir,
            embedding_loader,
        )
        label_entries[label_cluster] = label_entry

        label_centroid = label_entry.get("centroid")
        label_matrix = label_entry.get("matrix")
        if (
            target_centroid is None
            or label_centroid is None
            or label_matrix is None
            or label_matrix.size == 0
        ):
            continue

        centroid_sim = float(np.dot(target_centroid, label_centroid))
        centroid_scores.append((label_cluster, centroid_sim))

    if not centroid_scores:
        return (
            {
                "target_cluster": target_cluster,
                "recommended_label": "",
                "recommended_score": None,
                "candidate_scores": [],
                "confidence_warning": True,
                "confidence_reason": "유효 임베딩 라벨 후보가 없어 자동 추천을 생성할 수 없습니다.",
                "recommended_samples": [],
                "target_samples": _choose_representative_samples(context, target_cluster, target_entry, 5),
            },
            embedding_cache,
        )

    centroid_scores.sort(key=lambda item: item[1], reverse=True)
    compressed = centroid_scores[: max(1, centroid_top_n)]

    full_scores = []
    for label_cluster, _centroid_sim in compressed:
        label_entry = label_entries[label_cluster]
        label_matrix = label_entry.get("matrix")
        if label_matrix is None or label_matrix.size == 0:
            continue
        score = float(np.matmul(target_matrix, label_matrix.T).mean())
        full_scores.append((label_cluster, score))

    if not full_scores:
        return (
            {
                "target_cluster": target_cluster,
                "recommended_label": "",
                "recommended_score": None,
                "candidate_scores": [],
                "confidence_warning": True,
                "confidence_reason": "정밀 유사도 계산 결과가 비어 자동 추천을 생성할 수 없습니다.",
                "recommended_samples": [],
                "target_samples": _choose_representative_samples(context, target_cluster, target_entry, 5),
            },
            embedding_cache,
        )

    full_scores.sort(key=lambda item: item[1], reverse=True)
    top_scores = full_scores[: max(1, top_k)]

    recommended_label, recommended_score = top_scores[0]
    top2_score = top_scores[1][1] if len(top_scores) > 1 else top_scores[0][1]
    score_gap = float(recommended_score - top2_score)

    low_abs = recommended_score < low_score_threshold
    low_gap = score_gap < low_gap_threshold if len(top_scores) > 1 else False

    reasons = []
    if low_abs:
        reasons.append(f"top1={recommended_score:.4f} < {low_score_threshold:.2f}")
    if low_gap and len(top_scores) > 1:
        reasons.append(f"top1-top2={score_gap:.4f} < {low_gap_threshold:.2f}")

    confidence_warning = bool(reasons)
    confidence_reason = " / ".join(reasons) if reasons else ""

    recommended_entry = label_entries[recommended_label]

    return (
        {
            "target_cluster": target_cluster,
            "recommended_label": recommended_label,
            "recommended_score": float(recommended_score),
            "candidate_scores": [{"label": label, "score": float(score)} for label, score in top_scores],
            "confidence_warning": confidence_warning,
            "confidence_reason": confidence_reason,
            "recommended_samples": _choose_representative_samples(context, recommended_label, recommended_entry, 3),
            "target_samples": _choose_representative_samples(context, target_cluster, target_entry, 5),
        },
        embedding_cache,
    )


def validate_label_name(label_name: str) -> Tuple[bool, str, str]:
    clean = str(label_name or "").strip()
    if not clean:
        return False, "", "라벨명이 비어 있습니다."
    if clean.lower() == "noise":
        return False, "", "`noise`는 라벨명으로 사용할 수 없습니다."
    if is_unlabeled_cluster(clean):
        return False, "", "`clustering_숫자` 패턴은 라벨명으로 사용할 수 없습니다."
    if "/" in clean or "\\" in clean:
        return False, "", "라벨명에 경로 구분자(`/`, `\\`)는 사용할 수 없습니다."
    return True, clean, ""


def apply_cluster_label(result_folder: str, source_cluster: str, target_label: str) -> Dict[str, Any]:
    result_folder = os.path.abspath(result_folder)
    source_cluster = str(source_cluster or "").strip()
    target_label = str(target_label or "").strip()

    if not source_cluster:
        raise ValueError("source_cluster is empty")
    if not target_label:
        raise ValueError("target_label is empty")

    src_dir = os.path.join(result_folder, source_cluster)
    dst_dir = os.path.join(result_folder, target_label)

    if source_cluster == target_label:
        return {
            "source_cluster": source_cluster,
            "target_label": target_label,
            "moved_files": 0,
            "overwritten_files": 0,
        }

    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source cluster directory not found: {src_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    moved_files = 0
    overwritten_files = 0

    with os.scandir(src_dir) as entries:
        for entry in entries:
            if not entry.is_file() or not entry.name.lower().endswith(".wav"):
                continue
            src_path = entry.path
            dst_path = os.path.join(dst_dir, entry.name)
            if os.path.exists(dst_path):
                os.remove(dst_path)
                overwritten_files += 1
            shutil.move(src_path, dst_path)
            moved_files += 1

    if os.path.isdir(src_dir) and not os.listdir(src_dir):
        os.rmdir(src_dir)

    return {
        "source_cluster": source_cluster,
        "target_label": target_label,
        "moved_files": moved_files,
        "overwritten_files": overwritten_files,
    }
