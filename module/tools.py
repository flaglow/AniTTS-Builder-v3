import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import ffmpeg
import requests
import shutil
from tqdm import tqdm

SUPPORTED_MEDIA_EXTENSIONS = (
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".webm",
    ".mp3",
    ".aac",
    ".flac",
    ".ogg",
    ".m4a",
    ".wav",
)

PRETRAINED_MODEL_URLS = {
    "vocal_models/Kim_MelBandRoformer.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/vocal_models/Kim_MelBandRoformer.ckpt",
    "vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    "single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    "single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt": "https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
}


def _safe_positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None


def _auto_wav_convert_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count // 2))


def _resolve_wav_convert_workers(workers):
    if workers is not None:
        parsed_workers = _safe_positive_int(workers)
        if parsed_workers is not None:
            return parsed_workers, "argument"
        print(f"[WARN] Invalid workers argument '{workers}'. Falling back to env/auto.")

    env_workers = os.getenv("ANITTS_WAV_CONVERT_WORKERS")
    if env_workers is not None:
        parsed_env = _safe_positive_int(env_workers)
        if parsed_env is not None:
            return parsed_env, "env"
        print(f"[WARN] Invalid ANITTS_WAV_CONVERT_WORKERS='{env_workers}'. Falling back to auto.")

    return _auto_wav_convert_workers(), "auto"


def _build_wav_conversion_jobs(input_folder, output_folder):
    jobs = []
    skipped_collisions = []
    seen_outputs = {}

    for file_name in sorted(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, file_name)
        if not os.path.isfile(input_path):
            continue
        if not file_name.lower().endswith(SUPPORTED_MEDIA_EXTENSIONS):
            continue

        output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".wav")
        previous_input = seen_outputs.get(output_path)
        if previous_input is not None:
            skipped_collisions.append((input_path, previous_input, output_path))
            continue

        seen_outputs[output_path] = input_path
        jobs.append((input_path, output_path))

    return jobs, skipped_collisions

def convert_to_wav(input_file, output_wav):
    """
    Convert an audio or video file to WAV format.
    
    Args:
        input_file (str): Path to the input file (audio or video).
        output_wav (str): Path to save the output WAV file.
    """
    try:
        ext = os.path.splitext(input_file)[1].lower()
        
        if ext in SUPPORTED_MEDIA_EXTENSIONS:
            print(f"[INFO] Converting: {input_file} -> {output_wav}")
            (
                ffmpeg
                .input(input_file)
                .output(output_wav, format="wav", acodec="pcm_s16le", ac=1)
                .run(overwrite_output=True)
            )
            print(f"[INFO] Conversion completed: {output_wav}")
            return True
        else:
            print(f"[WARN] Unsupported file format: {ext}")
            return False
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "Unknown error"
        print(f"[ERROR] FFmpeg conversion error: {error_message}")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected conversion error for '{input_file}': {e}")
        return False

def batch_convert_to_wav(input_folder, output_folder, workers=None):
    """
    Convert all audio and video files in a folder to WAV format.
    
    Args:
        input_folder (str): Path to the folder containing input files.
        output_folder (str): Path to save the converted WAV files.
        workers (int | None): Number of parallel conversion workers.
    """
    print(f"[INFO] Starting batch conversion to WAV.")
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Output folder: {output_folder}")

    if not os.path.isdir(input_folder):
        print(f"[ERROR] Input folder does not exist: {input_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    jobs, skipped_collisions = _build_wav_conversion_jobs(input_folder, output_folder)
    for current_input, previous_input, output_path in skipped_collisions:
        print(
            f"[WARN] Skipping '{current_input}' because output collision exists with "
            f"'{previous_input}' -> {output_path}"
        )

    if not jobs:
        print(
            "[INFO] Batch conversion to WAV completed. "
            f"total=0 converted=0 failed=0 skipped={len(skipped_collisions)} workers=1(source=auto)"
        )
        return

    resolved_workers, worker_source = _resolve_wav_convert_workers(workers)
    effective_workers = min(resolved_workers, len(jobs))
    print(
        f"[INFO] WAV conversion worker policy: requested={resolved_workers} "
        f"effective={effective_workers} source={worker_source}"
    )

    files_converted = 0
    files_failed = 0
    failed_inputs = []

    if effective_workers <= 1:
        for input_path, output_path in jobs:
            if convert_to_wav(input_path, output_path):
                files_converted += 1
            else:
                files_failed += 1
                failed_inputs.append(input_path)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(convert_to_wav, input_path, output_path): input_path
                for input_path, output_path in jobs
            }
            for future in as_completed(futures):
                input_path = futures[future]
                try:
                    success = bool(future.result())
                except Exception as e:
                    print(f"[ERROR] Worker crashed while converting '{input_path}': {e}")
                    success = False

                if success:
                    files_converted += 1
                else:
                    files_failed += 1
                    failed_inputs.append(input_path)

    print(
        "[INFO] Batch conversion to WAV completed. "
        f"total={len(jobs)} converted={files_converted} "
        f"failed={files_failed} skipped={len(skipped_collisions)} "
        f"workers={effective_workers}(source={worker_source})"
    )
    if failed_inputs:
        print("[WARN] Failed input files:")
        for failed_path in failed_inputs:
            print(f"[WARN] - {failed_path}")

def convert_wav_to_mp3(input_wav, output_mp3):
    """
    Convert a WAV file to MP3 format (mono, 16kHz).
    
    Args:
        input_wav (str): Path to the input WAV file.
        output_mp3 (str): Path to save the output MP3 file.
    """
    try:
        print(f"[INFO] Converting WAV to MP3: {input_wav} -> {output_mp3}")
        (
            ffmpeg
            .input(input_wav)
            .output(output_mp3, format='mp3', acodec='libmp3lame', ar='16000', ac=1)
            .run(overwrite_output=True)
        )
        print(f"[INFO] Conversion to MP3 completed: {output_mp3}")
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else "Unknown error"
        print(f"[ERROR] FFmpeg error during WAV to MP3 conversion: {error_message}")

def batch_convert_wav_to_mp3(input_folder, output_folder):
    """
    Convert all WAV files in a folder to MP3 format (mono, 16kHz).
    
    Args:
        input_folder (str): Path to the folder containing WAV files.
        output_folder (str): Path to save the converted MP3 files.
    """
    print(f"[INFO] Starting batch conversion from WAV to MP3.")
    print(f"[INFO] Input folder: {input_folder}")
    print(f"[INFO] Output folder: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    files_converted = 0
    for file_name in os.listdir(input_folder):
        input_path = os.path.join(input_folder, file_name)
        if os.path.isfile(input_path) and file_name.lower().endswith(".wav"):
            output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp3")
            convert_wav_to_mp3(input_path, output_path)
            files_converted += 1
    print(f"[INFO] Batch conversion from WAV to MP3 completed. Total files converted: {files_converted}")

def download_file(url, save_path):
    """
    Download a file from a URL and save it to a specified path.
    
    Args:
        url (str): URL of the file to download.
        save_path (str): Path to save the downloaded file.
    """
    if os.path.exists(save_path):
        print(f"[INFO] File already exists: {save_path}")
        return
    
    print(f"[INFO] Starting download from: {url}")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "wb") as file, tqdm(
        desc=os.path.basename(save_path), total=total_size, unit="B", unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            bar.update(len(chunk))
    
    print(f"[INFO] Download completed: {save_path}")

def download_pretrained_models():
    """
    Download necessary pretrained models from Hugging Face.
    """
    print("[INFO] Starting download of pretrained models.")
    for filename, url in PRETRAINED_MODEL_URLS.items():
        save_path = os.path.join("./module/model/MSST_WebUI/pretrain/", filename)
        download_file(url, save_path)
    print("[INFO] All pretrained models have been downloaded.")

def move_matching_text_files(folder1, folder2):
    """
    Moves text files from folder1 to the corresponding subfolders in folder2 
    based on matching filenames with WAV files.

    Args:
        folder1 (str): Path to the folder containing text files.
        folder2 (str): Path to the main folder containing subfolders with WAV files.
    """
    print(f"[INFO] Starting to move matching text files from '{folder1}' to corresponding subfolders in '{folder2}'.")
    # Get all text file names in folder1 (without extension)
    text_files = {os.path.splitext(f)[0]: os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(".txt")}
    
    # Track moved text files
    moved_files = set()

    # Iterate through all subfolders in folder2
    for root, _, files in os.walk(folder2):
        for file in files:
            if file.endswith(".wav"):
                wav_name = os.path.splitext(file)[0]  # Get the name without extension
                if wav_name in text_files:
                    text_file_path = text_files[wav_name]
                    target_path = os.path.join(root, os.path.basename(text_file_path))
                    
                    # Move text file
                    shutil.move(text_file_path, target_path)
                    moved_files.add(wav_name)
                    print(f"[INFO] Moved text file: {text_file_path} -> {target_path}")

    # Remove remaining text files in folder1
    remaining_files = set(text_files.keys()) - moved_files
    for file_name in remaining_files:
        file_path = text_files[file_name]
        os.remove(file_path)
        print(f"[INFO] Deleted remaining text file: {file_path}")
    print("[INFO] Completed moving matching text files.")
