from module.tools import (
    SUPPORTED_MEDIA_EXTENSIONS,
    PRETRAINED_MODEL_URLS,
    batch_convert_to_wav,
    download_pretrained_models,
    batch_convert_wav_to_mp3,
)
from module.whisper import process_audio_files
from module.msst import msst_for_main
from module.clustering import (
    clustering_for_main,
    refresh_clustering_manifest_cluster_dirs,
    sync_clustering_embeddings_store_for_manifest,
)
from module.ass_slice import run_ass_slice
from module.labeling import (
    apply_cluster_label,
    build_labeling_context,
    get_label_choices,
    recommend_label_for_cluster,
    validate_label_name,
)
import os
import time
import gradio as gr
import warnings
from datetime import datetime
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def _now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _count_files_with_ext(folder, exts):
    if not os.path.isdir(folder):
        return 0
    ext_set = {e.lower() for e in exts}
    count = 0
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in ext_set:
            count += 1
    return count


def _count_wavs_recursive(folder):
    if not os.path.isdir(folder):
        return 0
    total = 0
    for root, _, files in os.walk(folder):
        total += sum(1 for f in files if f.lower().endswith(".wav"))
    return total


def _count_target_model_files():
    root = "./module/model/MSST_WebUI/pretrain"
    total = 0
    for rel_path in PRETRAINED_MODEL_URLS.keys():
        if os.path.exists(os.path.join(root, rel_path)):
            total += 1
    return total


def _stage_log_start(stage_name, input_files=None, extra=None):
    message = f"[INFO] [STAGE] {stage_name} start_time={_now_str()}"
    if input_files is not None:
        message += f" input_files={input_files}"
    if extra:
        message += f" {extra}"
    print(message)


def _stage_log_done(stage_name, started_at_perf, input_files=None, output_files=None, extra=None):
    elapsed_s = time.perf_counter() - started_at_perf
    avg_s = (elapsed_s / input_files) if input_files and input_files > 0 else 0.0
    message = (
        f"[INFO] [STAGE] {stage_name} done_time={_now_str()} "
        f"elapsed_s={elapsed_s:.3f} avg_s_per_file={avg_s:.3f}"
    )
    if input_files is not None:
        message += f" input_files={input_files}"
    if output_files is not None:
        message += f" output_files={output_files}"
    if extra:
        message += f" {extra}"
    print(message)


def _stage_log_fail(stage_name, started_at_perf, input_files=None, output_files=None, error=None):
    elapsed_s = time.perf_counter() - started_at_perf
    message = (
        f"[ERROR] [STAGE] {stage_name} failed_time={_now_str()} "
        f"elapsed_s={elapsed_s:.3f}"
    )
    if input_files is not None:
        message += f" input_files={input_files}"
    if output_files is not None:
        message += f" output_files={output_files}"
    if error:
        message += f" error={error}"
    print(message)


def stage_convert_to_wav(video_folder, wav_folder, pipeline_name):
    input_count = _count_files_with_ext(video_folder, SUPPORTED_MEDIA_EXTENSIONS)
    before_output = _count_files_with_ext(wav_folder, (".wav",))
    stage_name = f"{pipeline_name}#1 Convert to WAV"
    started_at = time.perf_counter()
    _stage_log_start(stage_name, input_files=input_count, extra=f"output_wavs_before={before_output}")
    try:
        batch_convert_to_wav(video_folder, wav_folder)
        after_output = _count_files_with_ext(wav_folder, (".wav",))
        _stage_log_done(
            stage_name,
            started_at,
            input_files=input_count,
            output_files=after_output,
            extra=f"new_output_files={max(0, after_output - before_output)}",
        )
    except Exception as exc:
        _stage_log_fail(stage_name, started_at, input_files=input_count, output_files=before_output, error=exc)
        raise


def stage_download_models(pipeline_name):
    stage_label = "Download Transcribe Models" if pipeline_name == "NoSub" else "Download Separation Models"
    stage_name = f"{pipeline_name}#2 {stage_label}"
    target_count = len(PRETRAINED_MODEL_URLS)
    before_count = _count_target_model_files()
    started_at = time.perf_counter()
    _stage_log_start(
        stage_name,
        input_files=target_count,
        extra=f"models_present_before={before_count} missing_before={max(0, target_count - before_count)}",
    )
    try:
        download_pretrained_models()
        after_count = _count_target_model_files()
        _stage_log_done(
            stage_name,
            started_at,
            input_files=target_count,
            output_files=after_count,
            extra=f"downloaded_now={max(0, after_count - before_count)} missing_after={max(0, target_count - after_count)}",
        )
    except Exception as exc:
        _stage_log_fail(stage_name, started_at, input_files=target_count, output_files=before_count, error=exc)
        raise


def stage_remove_wav_bgm(wav_folder, pipeline_name):
    stage_name = f"{pipeline_name}#3 Remove WAV BGM"
    input_count = _count_files_with_ext(wav_folder, (".wav",))
    started_at = time.perf_counter()
    _stage_log_start(stage_name, input_files=input_count)
    try:
        msst_for_main(wav_folder)
        output_count = _count_files_with_ext(wav_folder, (".wav",))
        _stage_log_done(stage_name, started_at, input_files=input_count, output_files=output_count)
    except Exception as exc:
        _stage_log_fail(stage_name, started_at, input_files=input_count, error=exc)
        raise


def stage_convert_wav_to_mp3(wav_folder, mp3_folder):
    stage_name = "NoSub#4 Convert to MP3"
    input_count = _count_files_with_ext(wav_folder, (".wav",))
    before_output = _count_files_with_ext(mp3_folder, (".mp3",))
    started_at = time.perf_counter()
    _stage_log_start(stage_name, input_files=input_count, extra=f"output_mp3_before={before_output}")
    try:
        batch_convert_wav_to_mp3(wav_folder, mp3_folder)
        after_output = _count_files_with_ext(mp3_folder, (".mp3",))
        _stage_log_done(
            stage_name,
            started_at,
            input_files=input_count,
            output_files=after_output,
            extra=f"new_output_files={max(0, after_output - before_output)}",
        )
    except Exception as exc:
        _stage_log_fail(stage_name, started_at, input_files=input_count, output_files=before_output, error=exc)
        raise


def stage_transcribe(mp3_folder, wav_folder, whisper_cache_dir, model_id):
    stage_name = "NoSub#5 Generate Timestamps"
    input_count = _count_files_with_ext(mp3_folder, (".mp3",))
    wav_before = _count_files_with_ext(wav_folder, (".wav",))
    started_at = time.perf_counter()
    _stage_log_start(
        stage_name,
        input_files=input_count,
        extra=f"output_wavs_before={wav_before} model_id={model_id}",
    )
    try:
        process_audio_files(mp3_folder, wav_folder, whisper_cache_dir, model_id)
        wav_after = _count_files_with_ext(wav_folder, (".wav",))
        _stage_log_done(
            stage_name,
            started_at,
            input_files=input_count,
            output_files=wav_after,
            extra=f"new_output_files={max(0, wav_after - wav_before)}",
        )
    except Exception as exc:
        _stage_log_fail(stage_name, started_at, input_files=input_count, output_files=wav_before, error=exc)
        raise


def stage_slice_by_subtitles(dry_run, auto_filter_non_dialogue):
    stage_name = "Sub#4 Slice by Subtitles"
    subtitle_count = _count_files_with_ext("./data/transcribe", (".ass", ".srt", ".smi"))
    wav_before = _count_files_with_ext("./data/audio_wav", (".wav",))
    txt_before = _count_files_with_ext("./data/transcribe", (".txt",))
    started_at = time.perf_counter()
    _stage_log_start(
        stage_name,
        input_files=subtitle_count,
        extra=(
            f"dry_run={dry_run} auto_filter_non_dialogue={auto_filter_non_dialogue} "
            f"output_wavs_before={wav_before} output_txt_before={txt_before}"
        ),
    )
    try:
        run_ass_slice(dry_run=dry_run, auto_filter_non_dialogue=auto_filter_non_dialogue)
        wav_after = _count_files_with_ext("./data/audio_wav", (".wav",))
        txt_after = _count_files_with_ext("./data/transcribe", (".txt",))
        _stage_log_done(
            stage_name,
            started_at,
            input_files=subtitle_count,
            output_files=wav_after,
            extra=(
                f"new_output_wavs={max(0, wav_after - wav_before)} "
                f"new_output_txts={max(0, txt_after - txt_before)}"
            ),
        )
    except Exception as exc:
        _stage_log_fail(stage_name, started_at, input_files=subtitle_count, output_files=wav_before, error=exc)
        raise


def stage_clustering(wav_folder, result_folder, embeddings_cache_dir, pipeline_name, stage_number):
    stage_name = f"{pipeline_name}#{stage_number} Run Embeddings & Clustering"
    input_count = _count_files_with_ext(wav_folder, (".wav",))
    output_before = _count_wavs_recursive(result_folder)
    started_at = time.perf_counter()
    _stage_log_start(stage_name, input_files=input_count, extra=f"result_wavs_before={output_before}")
    try:
        clustering_for_main(wav_folder, result_folder, embeddings_cache_dir)
        output_after = _count_wavs_recursive(result_folder)
        _stage_log_done(
            stage_name,
            started_at,
            input_files=input_count,
            output_files=output_after,
            extra=f"new_clustered_wavs={max(0, output_after - output_before)}",
        )
    except Exception as exc:
        _stage_log_fail(stage_name, started_at, input_files=input_count, output_files=output_before, error=exc)
        raise


def stage_refresh_cluster_dirs(manifest_path, result_folder, embeddings_cache_dir):
    stage_name = "Utility#1 Refresh cluster_dir"
    input_count = _count_wavs_recursive(result_folder)
    started_at = time.perf_counter()
    _stage_log_start(
        stage_name,
        input_files=input_count,
        extra=(
            f"manifest_path={manifest_path} result_folder={result_folder} "
            f"embeddings_cache_dir={embeddings_cache_dir}"
        ),
    )
    try:
        manifest_path, rows_count, changed_rows, added_rows, missing_rows = refresh_clustering_manifest_cluster_dirs(
            manifest_path=manifest_path,
            destination_folder=result_folder,
            append_missing_rows=True,
        )
        (
            embedding_store_path,
            embedding_expected_files,
            embedding_available_files,
            embedding_added_files,
            embedding_missing_after,
        ) = sync_clustering_embeddings_store_for_manifest(
            manifest_path=manifest_path,
            destination_folder=result_folder,
            cache_dir=embeddings_cache_dir,
        )
        _stage_log_done(
            stage_name,
            started_at,
            input_files=input_count,
            output_files=rows_count,
            extra=(
                f"manifest_path={manifest_path} changed_rows={changed_rows} "
                f"added_rows={added_rows} missing_rows={missing_rows} "
                f"embedding_store_path={embedding_store_path} "
                f"embedding_expected_files={embedding_expected_files} "
                f"embedding_available_files={embedding_available_files} "
                f"embedding_added_files={embedding_added_files} "
                f"embedding_missing_after={embedding_missing_after}"
            ),
        )
    except Exception as exc:
        _stage_log_fail(stage_name, started_at, input_files=input_count, error=exc)
        raise


def _labeling_safe_cache(cache):
    if isinstance(cache, dict) and isinstance(cache.get("clusters"), dict):
        return cache
    return {"clusters": {}}


def _labeling_format_sample(sample):
    if not sample:
        return "샘플 없음"

    transcript = sample.get("transcript") or "(전사 없음)"
    audio_path = sample.get("audio_path") or ""
    audio_note = "" if (audio_path and os.path.exists(audio_path)) else "\n오디오: 없음"
    return (
        f"클러스터: `{sample.get('cluster_name', '')}`\n"
        f"파일: `{sample.get('filename', '')}`\n"
        f"구간: {sample.get('timestamp_start', '')} ~ {sample.get('timestamp_end', '')} "
        f"({float(sample.get('duration', 0.0)):.2f}s)\n"
        f"대사: {transcript}"
        f"{audio_note}"
    )


def _labeling_sample_updates(samples, limit):
    updates = []
    safe_samples = list(samples or [])
    for idx in range(limit):
        if idx < len(safe_samples):
            sample = safe_samples[idx] or {}
            audio_path = sample.get("audio_path") or ""
            audio_value = audio_path if audio_path and os.path.exists(audio_path) else None
            updates.extend([gr.update(value=_labeling_format_sample(sample)), gr.update(value=audio_value)])
        else:
            updates.extend([gr.update(value="샘플 없음"), gr.update(value=None)])
    return updates


def _labeling_normalize_pending(pending_decisions):
    if not isinstance(pending_decisions, list):
        return []

    normalized = []
    seen = set()
    for item in pending_decisions:
        if not isinstance(item, dict):
            continue
        source_cluster = str(item.get("source_cluster") or "").strip()
        target_label = str(item.get("target_label") or "").strip()
        if not source_cluster or not target_label or source_cluster in seen:
            continue
        normalized.append({"source_cluster": source_cluster, "target_label": target_label})
        seen.add(source_cluster)
    return normalized


def _labeling_pending_upsert(pending_decisions, source_cluster, target_label):
    source_cluster = str(source_cluster or "").strip()
    target_label = str(target_label or "").strip()
    pending = _labeling_normalize_pending(pending_decisions)
    if not source_cluster or not target_label:
        return pending

    replaced = False
    for item in pending:
        if item["source_cluster"] == source_cluster:
            item["target_label"] = target_label
            replaced = True
            break
    if not replaced:
        pending.append({"source_cluster": source_cluster, "target_label": target_label})
    return pending


def _labeling_pending_remove(pending_decisions, source_cluster):
    source_cluster = str(source_cluster or "").strip()
    pending = _labeling_normalize_pending(pending_decisions)
    if not source_cluster:
        return pending
    return [item for item in pending if item["source_cluster"] != source_cluster]


def _labeling_build_progress(context, skipped_clusters, pending_decisions):
    if not context:
        return "진행상황: -"

    unlabeled = list(context.get("unlabeled_clusters", []))
    skipped = {str(x).strip() for x in (skipped_clusters or []) if str(x).strip()}
    pending = _labeling_normalize_pending(pending_decisions)
    pending_sources = {item["source_cluster"] for item in pending}

    skipped_active = [c for c in unlabeled if c in skipped]
    pending_active = [c for c in unlabeled if c in pending_sources]
    remaining = [c for c in unlabeled if c not in skipped and c not in pending_sources]
    return (
        f"진행상황: 미라벨 {len(unlabeled)}개 / "
        f"남은 후보 {len(remaining)}개 / "
        f"건너뜀 {len(skipped_active)}개 / "
        f"확정 대기 {len(pending_active)}개"
    )


def _labeling_pick_and_recommend(
    context,
    cache,
    current_cluster,
    skipped_clusters,
    embeddings_cache_dir,
    prefer_current,
    exclude_clusters=None,
):
    if not context:
        return "", None, cache, 0

    skipped = {str(x).strip() for x in (skipped_clusters or []) if str(x).strip()}
    excluded = {str(x).strip() for x in (exclude_clusters or []) if str(x).strip()}
    unlabeled = [c for c in context.get("unlabeled_clusters", []) if context.get("unlabeled_sizes", {}).get(c, 0) > 0]

    def _is_candidate(cluster_name):
        return cluster_name not in skipped and cluster_name not in excluded

    candidates = [c for c in unlabeled if _is_candidate(c)]
    target_cluster = ""

    if (
        prefer_current
        and current_cluster
        and current_cluster in unlabeled
        and _is_candidate(current_cluster)
    ):
        target_cluster = current_cluster
    elif candidates:
        target_cluster = candidates[0]

    if not target_cluster:
        return "", None, cache, 0

    recommendation, cache = recommend_label_for_cluster(
        context=context,
        target_cluster=target_cluster,
        embedding_cache=cache,
        embeddings_cache_dir=embeddings_cache_dir,
        top_k=3,
        centroid_top_n=8,
        low_score_threshold=0.35,
        low_gap_threshold=0.03,
    )
    return target_cluster, recommendation, cache, len(candidates)


def labeling_action_auto(
    manifest_path,
    result_folder,
    embeddings_cache_dir,
    context,
    cache,
    current_cluster,
    skipped_clusters,
    pending_decisions,
):
    try:
        cache = _labeling_safe_cache(cache)
        pending = _labeling_normalize_pending(pending_decisions)
        context = build_labeling_context(manifest_path, result_folder)
        pending_sources = {item["source_cluster"] for item in pending}
        target_cluster, recommendation, cache, remaining = _labeling_pick_and_recommend(
            context=context,
            cache=cache,
            current_cluster=current_cluster,
            skipped_clusters=skipped_clusters,
            embeddings_cache_dir=embeddings_cache_dir,
            prefer_current=bool(current_cluster),
            exclude_clusters=pending_sources,
        )
        review_state = "unreviewed"
        if not target_cluster:
            status = f"모든 미라벨 클러스터 검토가 완료됐습니다. (확정 대기 {len(pending)}건)"
        elif recommendation and recommendation.get("recommended_label"):
            status = (
                f"`{target_cluster}` 추천 완료: `{recommendation['recommended_label']}` "
                f"(score={recommendation.get('recommended_score', 0.0):.4f}, remaining={remaining}, pending={len(pending)})"
            )
        else:
            reason = recommendation.get("confidence_reason", "") if recommendation else ""
            status = f"`{target_cluster}` 자동 추천 생성 실패. {reason}".strip()
        return context, cache, target_cluster, recommendation, review_state, skipped_clusters, pending, status
    except Exception as exc:
        return (
            context,
            cache,
            current_cluster,
            None,
            "unreviewed",
            skipped_clusters,
            _labeling_normalize_pending(pending_decisions),
            f"자동 라벨 실패: {exc}",
        )


def labeling_action_next(
    manifest_path,
    result_folder,
    embeddings_cache_dir,
    context,
    cache,
    current_cluster,
    skipped_clusters,
    pending_decisions,
):
    try:
        cache = _labeling_safe_cache(cache)
        pending = _labeling_normalize_pending(pending_decisions)
        skipped = {str(x).strip() for x in (skipped_clusters or []) if str(x).strip()}
        if current_cluster:
            skipped.add(current_cluster)
        skipped_sorted = sorted(skipped)

        context = build_labeling_context(manifest_path, result_folder)
        pending_sources = {item["source_cluster"] for item in pending}
        target_cluster, recommendation, cache, remaining = _labeling_pick_and_recommend(
            context=context,
            cache=cache,
            current_cluster="",
            skipped_clusters=skipped_sorted,
            embeddings_cache_dir=embeddings_cache_dir,
            prefer_current=False,
            exclude_clusters=pending_sources,
        )
        review_state = "unreviewed"
        if not target_cluster:
            status = f"다음 후보가 없습니다. (확정 대기 {len(pending)}건)"
        else:
            status = f"다음 클러스터로 이동: `{target_cluster}` (remaining={remaining}, pending={len(pending)})"
        return context, cache, target_cluster, recommendation, review_state, skipped_sorted, pending, status
    except Exception as exc:
        return (
            context,
            cache,
            current_cluster,
            None,
            "unreviewed",
            skipped_clusters,
            _labeling_normalize_pending(pending_decisions),
            f"다음 클러스터 이동 실패: {exc}",
        )


def labeling_action_skip(
    manifest_path,
    result_folder,
    embeddings_cache_dir,
    context,
    cache,
    current_cluster,
    skipped_clusters,
    pending_decisions,
):
    context, cache, target_cluster, recommendation, review_state, skipped_sorted, pending, status = labeling_action_next(
        manifest_path,
        result_folder,
        embeddings_cache_dir,
        context,
        cache,
        current_cluster,
        skipped_clusters,
        pending_decisions,
    )
    if target_cluster:
        status = f"현재 클러스터를 건너뛰고 `{target_cluster}`로 이동했습니다. (pending={len(pending)})"
    return context, cache, target_cluster, recommendation, review_state, skipped_sorted, pending, status


def labeling_action_refresh(
    manifest_path,
    result_folder,
    embeddings_cache_dir,
    context,
    cache,
    current_cluster,
    skipped_clusters,
    pending_decisions,
):
    try:
        pending = _labeling_normalize_pending(pending_decisions)
        _manifest, rows_count, changed_rows, added_rows, missing_rows = refresh_clustering_manifest_cluster_dirs(
            manifest_path=manifest_path,
            destination_folder=result_folder,
            append_missing_rows=True,
        )
        (
            embedding_store_path,
            embedding_expected_files,
            embedding_available_files,
            embedding_added_files,
            embedding_missing_after,
        ) = sync_clustering_embeddings_store_for_manifest(
            manifest_path=manifest_path,
            destination_folder=result_folder,
            cache_dir=embeddings_cache_dir,
        )
        context = build_labeling_context(manifest_path, result_folder)
        cache = {"clusters": {}}
        pending_sources = {item["source_cluster"] for item in pending}
        target_cluster, recommendation, cache, remaining = _labeling_pick_and_recommend(
            context=context,
            cache=cache,
            current_cluster=current_cluster,
            skipped_clusters=skipped_clusters,
            embeddings_cache_dir=embeddings_cache_dir,
            prefer_current=True,
            exclude_clusters=pending_sources,
        )
        if target_cluster:
            status = (
                "Refresh 완료: "
                f"rows={rows_count}, changed={changed_rows}, added={added_rows}, missing={missing_rows}. "
                f"embedding_available={embedding_available_files}/{embedding_expected_files}, "
                f"embedding_added={embedding_added_files}, embedding_missing_after={embedding_missing_after}. "
                f"현재 대상 `{target_cluster}` (remaining={remaining}, pending={len(pending)})"
            )
        else:
            status = (
                "Refresh 완료: "
                f"rows={rows_count}, changed={changed_rows}, added={added_rows}, missing={missing_rows}. "
                f"embedding_store={embedding_store_path}, "
                f"embedding_available={embedding_available_files}/{embedding_expected_files}, "
                f"embedding_added={embedding_added_files}, embedding_missing_after={embedding_missing_after}. "
                f"남은 미라벨 클러스터가 없습니다. (pending={len(pending)})"
            )
        return context, cache, target_cluster, recommendation, "unreviewed", skipped_clusters, pending, status
    except Exception as exc:
        return (
            context,
            cache,
            current_cluster,
            None,
            "unreviewed",
            skipped_clusters,
            _labeling_normalize_pending(pending_decisions),
            f"Refresh 실패: {exc}",
        )


def labeling_action_accept(
    manifest_path,
    result_folder,
    embeddings_cache_dir,
    context,
    cache,
    current_cluster,
    recommendation,
    review_state,
    skipped_clusters,
    pending_decisions,
    status_text,
):
    _ = review_state, status_text
    cache = _labeling_safe_cache(cache)
    pending = _labeling_normalize_pending(pending_decisions)
    if not recommendation or not recommendation.get("recommended_label") or not current_cluster:
        status = "추천 결과가 없어 O를 적용할 수 없습니다."
        return context, cache, current_cluster, recommendation, "unreviewed", skipped_clusters, pending, status

    label = recommendation["recommended_label"]
    pending = _labeling_pending_upsert(pending, current_cluster, label)

    if not context:
        context = build_labeling_context(manifest_path, result_folder)

    pending_sources = {item["source_cluster"] for item in pending}
    target_cluster, next_recommendation, cache, remaining = _labeling_pick_and_recommend(
        context=context,
        cache=cache,
        current_cluster="",
        skipped_clusters=skipped_clusters,
        embeddings_cache_dir=embeddings_cache_dir,
        prefer_current=False,
        exclude_clusters=pending_sources,
    )

    if target_cluster:
        status = (
            f"O 저장: `{current_cluster}` -> `{label}` (확정 대기 {len(pending)}건). "
            f"다음 대상 `{target_cluster}` (remaining={remaining})"
        )
    else:
        status = (
            f"O 저장: `{current_cluster}` -> `{label}` (확정 대기 {len(pending)}건). "
            "남은 후보가 없습니다. `확정`으로 일괄 반영하세요."
        )
    return context, cache, target_cluster, next_recommendation, "unreviewed", skipped_clusters, pending, status


def labeling_action_reject(
    context,
    cache,
    current_cluster,
    recommendation,
    review_state,
    skipped_clusters,
    pending_decisions,
    status_text,
):
    _ = review_state, status_text
    cache = _labeling_safe_cache(cache)
    pending = _labeling_pending_remove(pending_decisions, current_cluster)
    status = "X 선택됨: 기존 라벨 선택 또는 새 라벨 입력 후 `확정`을 눌러 주세요."
    return context, cache, current_cluster, recommendation, "rejected", skipped_clusters, pending, status


def labeling_action_confirm(
    manifest_path,
    result_folder,
    embeddings_cache_dir,
    context,
    cache,
    current_cluster,
    recommendation,
    review_state,
    selected_label,
    new_label,
    skipped_clusters,
    pending_decisions,
):
    try:
        pending = _labeling_normalize_pending(pending_decisions)

        manual_label = ""
        if str(new_label or "").strip():
            manual_label = str(new_label).strip()
        elif str(selected_label or "").strip():
            manual_label = str(selected_label).strip()

        if review_state == "accepted" and recommendation and recommendation.get("recommended_label") and current_cluster:
            pending = _labeling_pending_upsert(pending, current_cluster, recommendation["recommended_label"])
        elif manual_label and current_cluster:
            pending = _labeling_pending_upsert(pending, current_cluster, manual_label)

        if not pending:
            raise ValueError("확정할 항목이 없습니다. O 선택 또는 수동 라벨 지정이 필요합니다.")

        move_results = []
        for item in pending:
            ok, normalized_label, error_message = validate_label_name(item["target_label"])
            if not ok:
                raise ValueError(f"{item['source_cluster']} -> {item['target_label']}: {error_message}")
            move_result = apply_cluster_label(
                result_folder=result_folder,
                source_cluster=item["source_cluster"],
                target_label=normalized_label,
            )
            move_results.append(move_result)

        _manifest, rows_count, changed_rows, added_rows, missing_rows = refresh_clustering_manifest_cluster_dirs(
            manifest_path=manifest_path,
            destination_folder=result_folder,
            append_missing_rows=True,
        )
        (
            _embedding_store_path,
            embedding_expected_files,
            embedding_available_files,
            embedding_added_files,
            embedding_missing_after,
        ) = sync_clustering_embeddings_store_for_manifest(
            manifest_path=manifest_path,
            destination_folder=result_folder,
            cache_dir=embeddings_cache_dir,
        )

        context = build_labeling_context(manifest_path, result_folder)
        cache = {"clusters": {}}
        applied_sources = {item["source_cluster"] for item in move_results}

        skipped = {str(x).strip() for x in (skipped_clusters or []) if str(x).strip()}
        skipped = {x for x in skipped if x and x not in applied_sources}
        skipped_sorted = sorted(skipped)

        target_cluster, recommendation, cache, remaining = _labeling_pick_and_recommend(
            context=context,
            cache=cache,
            current_cluster="",
            skipped_clusters=skipped_sorted,
            embeddings_cache_dir=embeddings_cache_dir,
            prefer_current=False,
            exclude_clusters=set(),
        )

        moved_total = sum(int(item.get("moved_files", 0)) for item in move_results)
        overwritten_total = sum(int(item.get("overwritten_files", 0)) for item in move_results)

        if target_cluster:
            status = (
                f"라벨 확정 완료: clusters={len(move_results)}, moved={moved_total}, overwritten={overwritten_total}. "
                f"Refresh rows={rows_count}, changed={changed_rows}, added={added_rows}, missing={missing_rows}. "
                f"embedding_available={embedding_available_files}/{embedding_expected_files}, "
                f"embedding_added={embedding_added_files}, embedding_missing_after={embedding_missing_after}. "
                f"다음 대상 `{target_cluster}` (remaining={remaining})"
            )
        else:
            status = (
                f"라벨 확정 완료: clusters={len(move_results)}, moved={moved_total}, overwritten={overwritten_total}. "
                f"Refresh rows={rows_count}, changed={changed_rows}, added={added_rows}, missing={missing_rows}. "
                f"embedding_available={embedding_available_files}/{embedding_expected_files}, "
                f"embedding_added={embedding_added_files}, embedding_missing_after={embedding_missing_after}. "
                "남은 미라벨 클러스터가 없습니다."
            )

        return context, cache, target_cluster, recommendation, "unreviewed", skipped_sorted, [], status
    except Exception as exc:
        return (
            context,
            cache,
            current_cluster,
            recommendation,
            review_state,
            skipped_clusters,
            _labeling_normalize_pending(pending_decisions),
            f"라벨 확정 실패: {exc}",
        )


def labeling_render(context, current_cluster, recommendation, review_state, status_text, skipped_clusters, pending_decisions):
    pending = _labeling_normalize_pending(pending_decisions)
    pending_map = {item["source_cluster"]: item["target_label"] for item in pending}
    status_value = status_text or "라벨링 준비 완료. `자동 라벨`을 눌러 시작하세요."
    progress_value = _labeling_build_progress(context, skipped_clusters, pending)

    target_title = "현재 대상 클러스터: -"
    rec_title = "추천 라벨: -"
    rec_candidates = "상위 후보: -"
    rec_warning = "신뢰도: -"
    decision_value = "판단 상태: 미선택"

    if recommendation:
        target_cluster = recommendation.get("target_cluster") or ""
        if target_cluster:
            target_title = f"현재 대상 클러스터: `{target_cluster}`"

        recommended_label = recommendation.get("recommended_label") or ""
        recommended_score = recommendation.get("recommended_score")
        if recommended_label and recommended_score is not None:
            rec_title = f"추천 라벨: `{recommended_label}` (평균 유사도 {float(recommended_score):.4f})"

        candidates = recommendation.get("candidate_scores") or []
        if candidates:
            lines = []
            for idx, item in enumerate(candidates, start=1):
                lines.append(f"{idx}. `{item.get('label', '')}` ({float(item.get('score', 0.0)):.4f})")
            rec_candidates = "상위 후보:\n" + "\n".join(lines)

        if recommendation.get("confidence_warning"):
            reason = recommendation.get("confidence_reason") or "임계값 미달"
            rec_warning = f"[경고] 추천 신뢰도 낮음: {reason}"
        elif recommended_label:
            rec_warning = "신뢰도: 기준 통과"

        if current_cluster and current_cluster in pending_map:
            decision_value = f"판단 상태: 확정 대기 (`{pending_map[current_cluster]}`)"
        elif review_state == "rejected":
            decision_value = "판단 상태: X (수동 라벨 지정 필요)"
        elif review_state == "accepted" and recommended_label:
            decision_value = f"판단 상태: O (추천 라벨 `{recommended_label}` 적용 대기)"

    label_choices = get_label_choices(context, exclude_noise=True) if context else []
    dropdown_value = None
    if current_cluster and current_cluster in pending_map and pending_map[current_cluster] in label_choices:
        dropdown_value = pending_map[current_cluster]
    elif (
        review_state == "accepted"
        and recommendation
        and recommendation.get("recommended_label") in label_choices
    ):
        dropdown_value = recommendation.get("recommended_label")

    dropdown_update = gr.update(choices=label_choices, value=dropdown_value)
    rec_sample_updates = _labeling_sample_updates(
        recommendation.get("recommended_samples", []) if recommendation else [],
        limit=3,
    )
    target_sample_updates = _labeling_sample_updates(
        recommendation.get("target_samples", []) if recommendation else [],
        limit=5,
    )

    return (
        [status_value, progress_value, target_title, rec_title, rec_candidates, rec_warning, decision_value, dropdown_update]
        + rec_sample_updates
        + target_sample_updates
    )

with gr.Blocks() as demo:
    gr.Markdown("## AniTTS Builder-v3")
    gr.Markdown(
        "애니메이션 음성을 TTS 데이터셋으로 변환하는 도구입니다. "
        "아래에서 **자막 파일이 있는 경우 / 없는 경우**를 선택해서 위에서 아래 순서대로 버튼을 눌러 주세요."
    )
    video_folder = gr.Textbox(value="./data/video", interactive=False, visible=False)
    wav_folder = gr.Textbox(value="./data/audio_wav", interactive=False, visible=False)
    mp3_folder = gr.Textbox(value="./data/audio_mp3", interactive=False, visible=False)
    text_folder = gr.Textbox(value="./data/transcribe", interactive=False, visible=False)
    result_folder = gr.Textbox(value="./data/result", interactive=False, visible=False)
    whisper_cache_dir = gr.Textbox(value="./module/model/whisper", interactive=False, visible=False)
    embeddings_cache_dir = gr.Textbox(value="./module/model/redimmet", interactive=False, visible=False)
    refresh_manifest_path = gr.Textbox(value="./data_mydata/clustering_slices.csv", interactive=False, visible=False)
    refresh_result_folder = gr.Textbox(value="./data_mydata/result", interactive=False, visible=False)
    labeling_context_state = gr.State(value=None)
    labeling_embedding_cache_state = gr.State(value={"clusters": {}})
    labeling_current_cluster_state = gr.State(value="")
    labeling_recommendation_state = gr.State(value=None)
    labeling_review_state = gr.State(value="unreviewed")
    labeling_skipped_state = gr.State(value=[])
    labeling_pending_state = gr.State(value=[])
    labeling_status_state = gr.State(value="라벨링 준비 완료. `자동 라벨`을 눌러 시작하세요.")
    # 버튼 활성화 상태 저장용 state 추가
    button_state = gr.State(value=True)

    with gr.Tabs():
        # -------------------------
        # 1) 자막 파일이 없는 경우
        # -------------------------
        with gr.Tab("1) 자막 파일이 **없는** 경우"):
            gr.Markdown(
                "### 자막이 없는 일반 파이프라인\n"
                "1. `./data/video` 폴더에 영상 파일(`.mkv`, `.mp4` 등)을 넣어 주세요.\n"
                "2. 아래 버튼을 **1 → 6 순서**로 한 번씩 눌러 주세요.\n"
                "3. 마지막에 `./data/result` 폴더에서 화자별 클러스터링 결과를 확인할 수 있습니다."
            )
            btn_ns_convert_wav = gr.Button("1. 동영상을 WAV로 변환 (Convert to WAV)")
            btn_ns_download_model = gr.Button("2. 음성 인식 모델 다운로드 (Download Transcribe Models)")
            btn_ns_msst_wav = gr.Button("3. BGM 제거 (Remove WAV BGM)")

            gr.Markdown(
                "#### 타임스탬프 생성 단계 (Whisper 사용)\n"
                "4. BGM이 제거된 WAV를 MP3로 변환합니다.\n"
                "5. Whisper로 음성을 인식해 발화 구간을 자동으로 잘라내고, "
                "`./data/whisper_slices.csv`에 파일명/인덱스/타임스탬프/전사 텍스트를 저장합니다."
            )
            btn_ns_convert_mp3 = gr.Button("4. WAV를 MP3로 변환 (Convert to MP3)")
            txt_model_id = gr.Textbox(
                label="5. Whisper Model ID (변경하지 않으면 기본값 사용)",
                value="openai/whisper-large-v3",
            )
            btn_ns_transcribe = gr.Button("5. Whisper로 타임스탬프 생성 (Generate Timestamps)")

            gr.Markdown(
                "#### 마지막 단계: 화자 클러스터링\n"
                "6. 잘려진 음성 조각들을 임베딩하고, 비슷한 목소리끼리 자동으로 묶은 뒤 "
                "`./data/clustering_slices.csv`에 클러스터 디렉토리를 함께 저장합니다."
            )
            btn_ns_clustering = gr.Button("6. 화자 임베딩 & 클러스터링 실행 (Run Embeddings & Clustering)")

        # -------------------------
        # 2) 자막 파일이 있는 경우
        # -------------------------
        with gr.Tab("2) 자막 파일이 **있는** 경우 (.ass/.srt/.smi)"):
            gr.Markdown(
                "### 자막 기반 파이프라인 (ASS/SRT/SMI)\n"
                "1. `./data/video` 폴더에 영상 파일을 넣고, **같은 파일 이름의 자막(`.ass`, `.srt`, `.smi`)**을 "
                "`./data/transcribe` 폴더에 넣어 주세요.\n"
                "   - 예) `[Moozzi2] Steins;Gate - 01 ....mkv` ↔ 같은 이름의 `.ass` / `.srt` / `.smi`\n"
                "2. 아래 버튼을 **1 → 5 순서**로 한 번씩 눌러 주세요.\n"
                "3. Whisper 대신 자막 타임스탬프를 사용해 음성 조각을 생성합니다."
            )
            btn_ws_convert_wav = gr.Button("1. 동영상을 WAV로 변환 (Convert to WAV)")
            btn_ws_download_model = gr.Button("2. 분리 모델 다운로드 (Download Separation Models)")
            btn_ws_msst_wav = gr.Button("3. BGM 제거 (Remove WAV BGM)")

            gr.Markdown(
                "#### 자막 타임스탬프로 음성 조각 만들기\n"
                "4. ASS/SRT/SMI 자막의 타임스탬프를 사용해 WAV를 잘게 자르고, "
                "`./data/transcribe`에 `[화자] 대사` 텍스트와 "
                "`./data/subtitle_slices.csv`에 파일명/인덱스/타임스탬프/전사 텍스트를 저장합니다."
            )
            ws_auto_filter = gr.Checkbox(label="4-A. 비대사 자막 자동 필터링", value=True)
            ws_dry_run = gr.Checkbox(label="4-B. Dry-run (파일 생성 없이 개수만 확인)", value=False)
            btn_ws_ass_slice = gr.Button("4. 자막 기준으로 음성 조각 생성 (Slice by Subtitles)")

            gr.Markdown(
                "#### 마지막 단계: 화자 클러스터링\n"
                "5. 자막 기반으로 잘려진 음성 조각들을 임베딩하고, 비슷한 목소리끼리 자동으로 묶은 뒤 "
                "`./data/clustering_slices.csv`에 클러스터 디렉토리를 함께 저장합니다."
            )
            btn_ws_clustering = gr.Button("5. 화자 임베딩 & 클러스터링 실행 (Run Embeddings & Clustering)")

        with gr.Tab("3) 라벨링"):
            gr.Markdown(
                "### 라벨링 탭\n"
                "`./data_mydata/clustering_slices.csv`를 기반으로 미라벨 클러스터(`clustering_숫자`)를 "
                "자동 추천 + O/X 검수로 확정합니다."
            )

            with gr.Row():
                btn_labeling_auto = gr.Button("자동 라벨")
                btn_labeling_next = gr.Button("다음 클러스터")
                btn_labeling_refresh = gr.Button("Refresh cluster_dir")

            labeling_status_md = gr.Markdown("라벨링 준비 완료. `자동 라벨`을 눌러 시작하세요.")
            labeling_progress_md = gr.Markdown("진행상황: -")

            with gr.Row():
                with gr.Column(scale=1):
                    labeling_target_title_md = gr.Markdown("현재 대상 클러스터: -")
                    labeling_recommend_title_md = gr.Markdown("추천 라벨: -")
                    labeling_candidates_md = gr.Markdown("상위 후보: -")
                    labeling_warning_md = gr.Markdown("신뢰도: -")

                    gr.Markdown("#### 추천 라벨 샘플 3개")
                    rec_sample_mds = []
                    rec_sample_audios = []
                    for idx in range(3):
                        rec_sample_mds.append(gr.Markdown(f"추천 샘플 {idx + 1}: 샘플 없음"))
                        rec_sample_audios.append(
                            gr.Audio(
                                label=f"추천 샘플 {idx + 1} 재생",
                                type="filepath",
                                interactive=False,
                                value=None,
                            )
                        )

                with gr.Column(scale=1):
                    gr.Markdown("#### 현재 미라벨 클러스터 샘플 5개")
                    target_sample_mds = []
                    target_sample_audios = []
                    for idx in range(5):
                        target_sample_mds.append(gr.Markdown(f"대상 샘플 {idx + 1}: 샘플 없음"))
                        target_sample_audios.append(
                            gr.Audio(
                                label=f"대상 샘플 {idx + 1} 재생",
                                type="filepath",
                                interactive=False,
                                value=None,
                            )
                        )

            with gr.Row():
                btn_labeling_accept = gr.Button("O (유사함)")
                btn_labeling_reject = gr.Button("X (다름)")
                btn_labeling_skip = gr.Button("건너뛰기")

            labeling_decision_md = gr.Markdown("판단 상태: 미선택")

            with gr.Row():
                labeling_existing_label = gr.Dropdown(
                    label="기존 라벨 선택",
                    choices=[],
                    value=None,
                    allow_custom_value=False,
                )
                labeling_new_label = gr.Textbox(
                    label="새 라벨 생성",
                    placeholder="예: 오카베 린타로",
                    value="",
                )
            btn_labeling_confirm = gr.Button("확정")

    gr.Markdown(
        "### 유틸리티\n"
        "`./data_mydata/clustering_slices.csv`의 `cluster_dir`를 현재 `./data_mydata/result` 구조로 최신화합니다."
    )
    btn_refresh_cluster_dirs = gr.Button("클러스터 디렉토리 리프래시 (Refresh cluster_dir)")

    # 모든 버튼을 리스트에 저장
    all_buttons = [
        btn_ns_convert_wav,
        btn_ns_download_model,
        btn_ns_msst_wav,
        btn_ns_convert_mp3,
        btn_ns_transcribe,
        btn_ns_clustering,
        btn_ws_convert_wav,
        btn_ws_download_model,
        btn_ws_msst_wav,
        btn_ws_ass_slice,
        btn_ws_clustering,
        btn_refresh_cluster_dirs,
        btn_labeling_auto,
        btn_labeling_next,
        btn_labeling_refresh,
        btn_labeling_accept,
        btn_labeling_reject,
        btn_labeling_confirm,
        btn_labeling_skip,
    ]

    labeling_state_outputs = [
        labeling_context_state,
        labeling_embedding_cache_state,
        labeling_current_cluster_state,
        labeling_recommendation_state,
        labeling_review_state,
        labeling_skipped_state,
        labeling_pending_state,
        labeling_status_state,
    ]
    labeling_render_outputs = [
        labeling_status_md,
        labeling_progress_md,
        labeling_target_title_md,
        labeling_recommend_title_md,
        labeling_candidates_md,
        labeling_warning_md,
        labeling_decision_md,
        labeling_existing_label,
    ]
    for md_comp, audio_comp in zip(rec_sample_mds, rec_sample_audios):
        labeling_render_outputs.extend([md_comp, audio_comp])
    for md_comp, audio_comp in zip(target_sample_mds, target_sample_audios):
        labeling_render_outputs.extend([md_comp, audio_comp])

    # 모든 버튼 비활성화 함수
    def disable_all():
        return [gr.update(interactive=False) for _ in all_buttons] + [False]

    # 모든 버튼 활성화 함수
    def enable_all():
        return [gr.update(interactive=True) for _ in all_buttons] + [True]

    # 1) 자막이 없는 경우 파이프라인
    btn_ns_convert_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(lambda v, w: stage_convert_to_wav(v, w, "NoSub"), inputs=[video_folder, wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_download_model.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(lambda: stage_download_models("NoSub"), outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_msst_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(lambda w: stage_remove_wav_bgm(w, "NoSub"), inputs=[wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_convert_mp3.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(stage_convert_wav_to_mp3, inputs=[wav_folder, mp3_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_transcribe.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(stage_transcribe, inputs=[mp3_folder, wav_folder, whisper_cache_dir, txt_model_id], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_clustering.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            lambda w, r, c: stage_clustering(w, r, c, "NoSub", 6),
            inputs=[wav_folder, result_folder, embeddings_cache_dir],
            outputs=[],
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    # 2) 자막이 있는 경우 파이프라인
    btn_ws_convert_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(lambda v, w: stage_convert_to_wav(v, w, "Sub"), inputs=[video_folder, wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ws_download_model.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(lambda: stage_download_models("Sub"), outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ws_msst_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(lambda w: stage_remove_wav_bgm(w, "Sub"), inputs=[wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ws_ass_slice.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            stage_slice_by_subtitles,
            inputs=[
                ws_dry_run,
                ws_auto_filter,
            ],
            outputs=[],
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ws_clustering.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            lambda w, r, c: stage_clustering(w, r, c, "Sub", 5),
            inputs=[wav_folder, result_folder, embeddings_cache_dir],
            outputs=[],
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_refresh_cluster_dirs.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            stage_refresh_cluster_dirs,
            inputs=[refresh_manifest_path, refresh_result_folder, embeddings_cache_dir],
            outputs=[],
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_labeling_auto.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            labeling_action_auto,
            inputs=[
                refresh_manifest_path,
                refresh_result_folder,
                embeddings_cache_dir,
                labeling_context_state,
                labeling_embedding_cache_state,
                labeling_current_cluster_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_state_outputs,
        ) \
        .then(
            labeling_render,
            inputs=[
                labeling_context_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_status_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_render_outputs,
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_labeling_next.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            labeling_action_next,
            inputs=[
                refresh_manifest_path,
                refresh_result_folder,
                embeddings_cache_dir,
                labeling_context_state,
                labeling_embedding_cache_state,
                labeling_current_cluster_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_state_outputs,
        ) \
        .then(
            labeling_render,
            inputs=[
                labeling_context_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_status_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_render_outputs,
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_labeling_skip.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            labeling_action_skip,
            inputs=[
                refresh_manifest_path,
                refresh_result_folder,
                embeddings_cache_dir,
                labeling_context_state,
                labeling_embedding_cache_state,
                labeling_current_cluster_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_state_outputs,
        ) \
        .then(
            labeling_render,
            inputs=[
                labeling_context_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_status_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_render_outputs,
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_labeling_refresh.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            labeling_action_refresh,
            inputs=[
                refresh_manifest_path,
                refresh_result_folder,
                embeddings_cache_dir,
                labeling_context_state,
                labeling_embedding_cache_state,
                labeling_current_cluster_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_state_outputs,
        ) \
        .then(
            labeling_render,
            inputs=[
                labeling_context_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_status_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_render_outputs,
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_labeling_accept.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            labeling_action_accept,
            inputs=[
                refresh_manifest_path,
                refresh_result_folder,
                embeddings_cache_dir,
                labeling_context_state,
                labeling_embedding_cache_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_skipped_state,
                labeling_pending_state,
                labeling_status_state,
            ],
            outputs=labeling_state_outputs,
        ) \
        .then(
            labeling_render,
            inputs=[
                labeling_context_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_status_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_render_outputs,
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_labeling_reject.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            labeling_action_reject,
            inputs=[
                labeling_context_state,
                labeling_embedding_cache_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_skipped_state,
                labeling_pending_state,
                labeling_status_state,
            ],
            outputs=labeling_state_outputs,
        ) \
        .then(
            labeling_render,
            inputs=[
                labeling_context_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_status_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_render_outputs,
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_labeling_confirm.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            labeling_action_confirm,
            inputs=[
                refresh_manifest_path,
                refresh_result_folder,
                embeddings_cache_dir,
                labeling_context_state,
                labeling_embedding_cache_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_existing_label,
                labeling_new_label,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_state_outputs,
        ) \
        .then(
            labeling_render,
            inputs=[
                labeling_context_state,
                labeling_current_cluster_state,
                labeling_recommendation_state,
                labeling_review_state,
                labeling_status_state,
                labeling_skipped_state,
                labeling_pending_state,
            ],
            outputs=labeling_render_outputs,
        ) \
        .then(lambda: gr.update(value=""), outputs=[labeling_new_label]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    demo.load(
        labeling_render,
        inputs=[
            labeling_context_state,
            labeling_current_cluster_state,
            labeling_recommendation_state,
            labeling_review_state,
            labeling_status_state,
            labeling_skipped_state,
            labeling_pending_state,
        ],
        outputs=labeling_render_outputs,
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
