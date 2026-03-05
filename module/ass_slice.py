import os
import re
import torchaudio

DATA_DIR = "./data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio_wav")
ASS_DIR = os.path.join(DATA_DIR, "transcribe")
OUT_DIR = os.path.join(DATA_DIR, "audio_wav")  # 그대로 audio_wav에 조각을 만듦

os.makedirs(OUT_DIR, exist_ok=True)

time_pattern = re.compile(r"(\d+):(\d+):(\d+\.\d+)")
dialogue_prefix = "Dialogue:"

def ass_time_to_sec(t: str) -> float:
    m = time_pattern.match(t.strip())
    if not m:
        return 0.0
    h, m_, s = m.groups()
    return int(h) * 3600 + int(m_) * 60 + float(s)

def parse_ass_dialogues(path):
    lines = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith(dialogue_prefix):
                continue
            # Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = line.strip().split(",", 9)
            if len(parts) < 10:
                continue
            _, start, end, style, actor, *_rest, text = parts
            start_s = ass_time_to_sec(start)
            end_s = ass_time_to_sec(end)
            if end_s <= start_s:
                continue
            # \N 줄바꿈, 태그 제거 등 최소 정리
            clean_text = re.sub(r"\{.*?\}", "", text).replace("\\N", " ").strip()
            lines.append({
                "start": start_s,
                "end": end_s,
                "actor": actor.strip(),
                "style": style.strip(),
                "text": clean_text
            })
    return lines

def run_ass_slice():
    ass_files = [f for f in os.listdir(ASS_DIR) if f.lower().endswith(".ass")]
    ass_files.sort()
    print(f"Found {len(ass_files)} ASS files.")

    global_idx = 0

    for ass_name in ass_files:
        base = os.path.splitext(ass_name)[0]
        wav_name = base + ".wav"
        wav_path = os.path.join(AUDIO_DIR, wav_name)
        ass_path = os.path.join(ASS_DIR, ass_name)

        if not os.path.exists(wav_path):
            print(f"[WARN] WAV not found for {ass_name} -> {wav_name}, skip.")
            continue

        print(f"[INFO] Processing {ass_name} with {wav_name}")
        dialogues = parse_ass_dialogues(ass_path)
        if not dialogues:
            print(f"[WARN] No dialogues in {ass_name}")
            continue

        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono

        for d in dialogues:
            start_s = d["start"]
            end_s = d["end"]
            # 너무 짧거나 너무 긴 구간은 옵션으로 필터링
            dur = end_s - start_s
            if dur < 0.3 or dur > 15.0:
                continue

            start_idx = int(start_s * sample_rate)
            end_idx = int(end_s * sample_rate)
            if end_idx <= start_idx or end_idx > waveform.shape[1]:
                continue

            slice_wave = waveform[:, start_idx:end_idx]
            out_stem = f"{base}__{global_idx:06d}"
            wav_out_name = f"{out_stem}.wav"
            txt_out_name = f"{out_stem}.txt"

            wav_out_path = os.path.join(OUT_DIR, wav_out_name)
            txt_out_path = os.path.join(DATA_DIR, "transcribe", txt_out_name)

            torchaudio.save(wav_out_path, slice_wave, sample_rate)

            with open(txt_out_path, "w", encoding="utf-8") as tf:
                # 화자 정보(ASS의 Name 필드)를 같이 저장
                actor = d["actor"] or "Unknown"
                tf.write(f"[{actor}] {d['text']}\n")

            global_idx += 1

        print(f"[INFO] Done {ass_name}, total slices so far: {global_idx}")

    print("[INFO] ASS 기반 슬라이싱이 완료되었습니다.")

if __name__ == "__main__":
    run_ass_slice()
