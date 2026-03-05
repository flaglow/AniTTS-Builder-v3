import os
import sys

sys.path.append(f"{os.path.dirname(__file__)}/model/MSST_WebUI")

from module.memory_cleanup import release_torch_resources
from module.model.MSST_WebUI.inference.msst_infer import MSSeparator
from module.model.MSST_WebUI.utils.logger import get_logger

def load_separator(model, model_type, folder_path):
    """
    Initialize MSSeparator with specified model and parameters.
    """
    print(f"[INFO] Initializing MSSeparator with model type '{model_type}' for folder '{folder_path}'.")
    logger = get_logger()
    config_path = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/model/MSST_WebUI/configs_backup/{model[0]}")
    model_path = os.path.abspath(f"{os.path.dirname(os.path.abspath(__file__))}/model/MSST_WebUI/pretrain/{model[1]}")
    print(f"[INFO] Using config: {config_path}")
    print(f"[INFO] Using model checkpoint: {model_path}")
    separator = MSSeparator(
        model_type=model_type,
        config_path=config_path,
        model_path=model_path,
        device='auto',
        device_ids=[0],
        output_format='wav',
        use_tta=False,
        store_dirs=folder_path,
        logger=logger,
        debug=True
    )
    print("[INFO] MSSeparator initialized successfully.")
    return separator

def process_msst(model, model_type, folder_path, stem):
    """
    Process audio files using the MSSeparator model.
    """
    print(f"[INFO] Starting processing with stem '{stem}' in folder '{folder_path}'.")
    separator = None
    inputs_list = []
    try:
        separator = load_separator(model, model_type, folder_path)
        print("[INFO] Processing folder using MSSeparator.")
        inputs_list = separator.process_folder(folder_path)
        print(f"[INFO] Processing complete. {len(inputs_list)} files processed.")
    finally:
        release_torch_resources("msst", [separator])

    results_list = [[f"{folder_path}/{i[:-4]}_{stem}.wav", f"{folder_path}/{i}"] for i in inputs_list]
    for old, new in results_list:
        print(f"[INFO] Replacing file: {new} with {old}.")
        os.remove(new)
        os.rename(old, new)

    # Remove files not in inputs_list
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename not in inputs_list:
            print(f"[INFO] Removing extra file: {filename}.")
            os.remove(file_path)
    print(f"[INFO] Processing with stem '{stem}' completed.")

def msst_for_main(folder_path):
    """
    Apply MSSeparator with multiple models for different audio processing tasks.
    """
    print(f"[INFO] Starting MSST processing for folder: {folder_path}.")
    models = [
        (["vocal_models/config_Kim_MelBandRoformer.yaml", "vocal_models/Kim_MelBandRoformer.ckpt"], 'vocals'),
        (["vocal_models/config_mel_band_roformer_karaoke.yaml", "vocal_models/model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"], 'karaoke'),
        (["single_stem_models/dereverb_mel_band_roformer_anvuew.yaml", "single_stem_models/dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt"], 'noreverb'),
        (["single_stem_models/model_mel_band_roformer_denoise.yaml", "single_stem_models/denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt"], 'dry')
    ]
    model_type = "mel_band_roformer"
    
    for model, stem in models:
        print(f"[INFO] Processing model for stem '{stem}'.")
        process_msst(model, model_type, folder_path, stem)
        print(f"[INFO] Completed processing for stem '{stem}'.")
    print("[INFO] All MSST processing tasks completed.")
