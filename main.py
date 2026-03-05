from module.tools import batch_convert_to_wav, download_pretrained_models, batch_convert_wav_to_mp3
from module.whisper import process_audio_files
from module.msst import msst_for_main
from module.clustering import clustering_for_main
from module.ass_slice import run_ass_slice
import os
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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
    sync_profile_path = gr.Textbox(value="./data/transcribe/sync_profile.json", interactive=False, visible=False)
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
                "3. Whisper 대신 자막 타임스탬프를 사용하되, 필요 시 자동 싱크 보정(오프셋/드리프트)을 적용합니다."
            )
            btn_ws_convert_wav = gr.Button("1. 동영상을 WAV로 변환 (Convert to WAV)")
            btn_ws_download_model = gr.Button("2. 분리 모델 다운로드 (Download Separation Models)")
            btn_ws_msst_wav = gr.Button("3. BGM 제거 (Remove WAV BGM)")

            gr.Markdown(
                "#### 자막 타임스탬프로 음성 조각 만들기\n"
                "4. ASS/SRT/SMI 자막의 타임스탬프를 사용해 WAV를 잘게 자르고, "
                "`./data/transcribe`에 `[화자] 대사` 텍스트를 저장합니다.\n"
                "- 자동 보정 실패 시 해당 에피소드는 스킵되고 `./logs/ass_sync` 및 "
                "`sync_profile.json`에 `needs_manual`로 기록됩니다."
            )
            ws_auto_sync = gr.Checkbox(label="4-A. 자동 싱크 보정 사용", value=True)
            ws_manual_offset = gr.Number(label="4-B. 수동 오프셋 (ms)", value=0.0, precision=2)
            ws_manual_drift = gr.Number(label="4-C. 수동 드리프트 (ppm)", value=0.0, precision=2)
            ws_auto_filter = gr.Checkbox(label="4-D. 비대사 자막 자동 필터링", value=True)
            ws_use_profile = gr.Checkbox(label="4-E. sync_profile.json 사용", value=True)
            ws_dry_run = gr.Checkbox(label="4-F. Dry-run (파일 생성 없이 리포트만)", value=False)
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
        .then(batch_convert_to_wav, inputs=[video_folder, wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_download_model.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(download_pretrained_models, inputs=[], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_msst_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(msst_for_main, inputs=[wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_convert_mp3.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(batch_convert_wav_to_mp3, inputs=[wav_folder, mp3_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_transcribe.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(process_audio_files, inputs=[mp3_folder, wav_folder, whisper_cache_dir, txt_model_id], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ns_clustering.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(clustering_for_main, inputs=[wav_folder, result_folder, embeddings_cache_dir], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    # 2) 자막이 있는 경우 파이프라인
    btn_ws_convert_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(batch_convert_to_wav, inputs=[video_folder, wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ws_download_model.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(download_pretrained_models, inputs=[], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ws_msst_wav.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(msst_for_main, inputs=[wav_folder], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ws_ass_slice.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(
            run_ass_slice,
            inputs=[
                ws_auto_sync,
                ws_manual_offset,
                ws_manual_drift,
                ws_use_profile,
                sync_profile_path,
                ws_dry_run,
                ws_auto_filter,
            ],
            outputs=[],
        ) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

    btn_ws_clustering.click(lambda: disable_all(), outputs=all_buttons + [button_state]) \
        .then(clustering_for_main, inputs=[wav_folder, result_folder, embeddings_cache_dir], outputs=[]) \
        .then(lambda: enable_all(), outputs=all_buttons + [button_state])

demo.launch(server_name="0.0.0.0", server_port=7860)
