from module.tools import (
    SUPPORTED_MEDIA_EXTENSIONS,
    PRETRAINED_MODEL_URLS,
    batch_convert_to_wav,
    download_pretrained_models,
    batch_convert_wav_to_mp3,
)
from module.whisper import process_audio_files
from module.msst import msst_for_main
from module.clustering import clustering_for_main
from module.ass_slice import run_ass_slice
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
                "5. Whisper로 음성을 인식해 발화 구간을 자동으로 잘라냅니다."
            )
            btn_ns_convert_mp3 = gr.Button("4. WAV를 MP3로 변환 (Convert to MP3)")
            txt_model_id = gr.Textbox(
                label="5. Whisper Model ID (변경하지 않으면 기본값 사용)",
                value="openai/whisper-large-v3",
            )
            btn_ns_transcribe = gr.Button("5. Whisper로 타임스탬프 생성 (Generate Timestamps)")

            gr.Markdown(
                "#### 마지막 단계: 화자 클러스터링\n"
                "6. 잘려진 음성 조각들을 임베딩하고, 비슷한 목소리끼리 자동으로 묶습니다."
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
                "`./data/transcribe`에 `[화자] 대사` 텍스트를 저장합니다."
            )
            ws_auto_filter = gr.Checkbox(label="4-A. 비대사 자막 자동 필터링", value=True)
            ws_dry_run = gr.Checkbox(label="4-B. Dry-run (파일 생성 없이 개수만 확인)", value=False)
            btn_ws_ass_slice = gr.Button("4. 자막 기준으로 음성 조각 생성 (Slice by Subtitles)")

            gr.Markdown(
                "#### 마지막 단계: 화자 클러스터링\n"
                "5. 자막 기반으로 잘려진 음성 조각들을 임베딩하고, 비슷한 목소리끼리 자동으로 묶습니다."
            )
            btn_ws_clustering = gr.Button("5. 화자 임베딩 & 클러스터링 실행 (Run Embeddings & Clustering)")

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
    ]

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

demo.launch(server_name="0.0.0.0", server_port=7860)
