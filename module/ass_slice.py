import html
import os
import re
import torchaudio

DATA_DIR = "./data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio_wav")
ASS_DIR = os.path.join(DATA_DIR, "transcribe")
OUT_DIR = os.path.join(DATA_DIR, "audio_wav")  # 그대로 audio_wav에 조각을 만듦
SUPPORTED_SUBTITLE_EXTS = (".ass", ".srt", ".smi")

os.makedirs(OUT_DIR, exist_ok=True)

ass_time_pattern = re.compile(r"(\d+):(\d+):(\d+\.\d+)")
srt_time_pattern = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")
dialogue_prefix = "Dialogue:"
non_dialogue_style_pattern = re.compile(
    r"(?i)(op|ed|title|sign|kara|lyric|gasa|ruby|cell|mail|phone|cm|preview)"
)
smi_sync_pattern = re.compile(r"(?is)<sync\b[^>]*?start\s*=\s*(\d+)[^>]*>")
smi_class_pattern = re.compile(r"(?is)<p\b[^>]*?class\s*=\s*['\"]?([^'\"\s>]+)")

def ass_time_to_sec(t: str) -> float:
    m = ass_time_pattern.match(t.strip())
    if not m:
        return 0.0
    h, m_, s = m.groups()
    return int(h) * 3600 + int(m_) * 60 + float(s)


def srt_time_to_sec(t: str) -> float:
    m = srt_time_pattern.match(t.strip())
    if not m:
        return 0.0
    h, m_, s, ms = m.groups()
    return int(h) * 3600 + int(m_) * 60 + int(s) + (int(ms) / (10 ** len(ms)))


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_text_file(path: str) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()


def _normalize_subtitle_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\\N", " ").replace("\\n", " ")
    cleaned = re.sub(r"(?i)<br\s*/?>", " ", cleaned)
    cleaned = re.sub(r"\{.*?\}", "", cleaned)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned).replace("\xa0", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def detect_subtitle_format(path: str, sample_text: str | None = None) -> str | None:
    sample = (sample_text or "").lower()
    if "<sync" in sample or "<sami" in sample:
        return "smi"
    if "dialogue:" in sample and "[events]" in sample:
        return "ass"
    if "-->" in sample and re.search(r"\d+:\d{2}:\d{2}[,.]\d+", sample):
        return "srt"

    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_SUBTITLE_EXTS:
        return ext[1:]
    return None


def _parse_ass_dialogues_from_text(text: str):
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith(dialogue_prefix):
            continue
        # Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
        parts = stripped.split(",", 9)
        if len(parts) < 10:
            continue
        _, start, end, style, actor, *_rest, raw_text = parts
        start_s = ass_time_to_sec(start)
        end_s = ass_time_to_sec(end)
        if end_s <= start_s:
            continue
        clean_text = _normalize_subtitle_text(raw_text)
        if not clean_text:
            continue
        lines.append(
            {
                "start": start_s,
                "end": end_s,
                "actor": actor.strip(),
                "style": (style or "Default").strip(),
                "text": clean_text,
            }
        )
    return lines


def _is_phase1_style(style: str) -> bool:
    s = (style or "").strip()
    if s == "Default":
        return True
    return re.match(r"^\d+-", s) is not None


def _filter_by_duration(dialogues, min_duration=0.3, max_duration=15.0):
    out = []
    for d in dialogues:
        s = _safe_float(d.get("start", 0.0), 0.0)
        e = _safe_float(d.get("end", 0.0), 0.0)
        dur = e - s
        if dur < min_duration or dur > max_duration:
            continue
        out.append(d)
    return out


def filter_dialogues_for_speech(dialogues, auto_filter_non_dialogue=True):
    if not auto_filter_non_dialogue:
        return list(dialogues), "disabled"

    phase1 = [d for d in dialogues if _is_phase1_style(d.get("style", ""))]
    if len(phase1) >= 80:
        return phase1, "phase1_default_or_numbered_style"

    phase2 = []
    for d in dialogues:
        style = d.get("style", "")
        text = d.get("text", "")
        if non_dialogue_style_pattern.search(style):
            continue
        if "♪" in text or "♬" in text:
            continue
        phase2.append(d)
    return phase2, "phase2_exclude_non_dialogue_style"


def _extract_actor(dialogue):
    actor = (dialogue.get("actor") or "").strip()
    if actor:
        return actor
    style = (dialogue.get("style") or "").strip()
    if "-" in style:
        _, right = style.split("-", 1)
        right = right.strip()
        if right:
            return right
    if style and style.lower() != "default":
        return style
    return "Unknown"


def parse_ass_dialogues(path):
    return _parse_ass_dialogues_from_text(_read_text_file(path))


def _parse_srt_dialogues_from_text(text: str):
    lines = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", normalized)
    range_pattern = re.compile(
        r"\s*(\d+:\d{2}:\d{2}[,.]\d+)\s*-->\s*(\d+:\d{2}:\d{2}[,.]\d+)"
    )

    for block in blocks:
        rows = [r.strip() for r in block.split("\n") if r.strip()]
        if not rows:
            continue
        time_line_idx = next((i for i, row in enumerate(rows) if "-->" in row), -1)
        if time_line_idx < 0:
            continue
        m = range_pattern.match(rows[time_line_idx])
        if not m:
            continue
        start_s = srt_time_to_sec(m.group(1))
        end_s = srt_time_to_sec(m.group(2))
        if end_s <= start_s:
            continue
        text_rows = rows[time_line_idx + 1 :]
        clean_text = _normalize_subtitle_text(" ".join(text_rows))
        if not clean_text:
            continue
        lines.append(
            {
                "start": start_s,
                "end": end_s,
                "actor": "",
                "style": "Default",
                "text": clean_text,
            }
        )
    return lines


def parse_srt_dialogues(path):
    return _parse_srt_dialogues_from_text(_read_text_file(path))


def _parse_smi_dialogues_from_text(text: str):
    lines = []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(smi_sync_pattern.finditer(normalized))
    if not matches:
        return lines

    for i, match in enumerate(matches):
        start_s = int(match.group(1)) / 1000.0
        next_match = matches[i + 1] if i + 1 < len(matches) else None
        if next_match is None:
            continue
        end_s = int(next_match.group(1)) / 1000.0
        if end_s <= start_s:
            continue
        segment = normalized[match.end() : next_match.start()]
        class_match = smi_class_pattern.search(segment)
        style = class_match.group(1).strip() if class_match else "Default"
        clean_text = _normalize_subtitle_text(segment)
        if not clean_text:
            continue
        lines.append(
            {
                "start": start_s,
                "end": end_s,
                "actor": "",
                "style": style or "Default",
                "text": clean_text,
            }
        )
    return lines


def parse_smi_dialogues(path):
    return _parse_smi_dialogues_from_text(_read_text_file(path))


def parse_subtitle_dialogues(path):
    text = _read_text_file(path)
    subtitle_format = detect_subtitle_format(path, sample_text=text[:20000])
    if subtitle_format == "ass":
        return _parse_ass_dialogues_from_text(text), subtitle_format
    if subtitle_format == "srt":
        return _parse_srt_dialogues_from_text(text), subtitle_format
    if subtitle_format == "smi":
        return _parse_smi_dialogues_from_text(text), subtitle_format
    return [], None

def run_ass_slice(
    dry_run: bool = False,
    auto_filter_non_dialogue: bool = True,
):
    os.makedirs(ASS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    subtitle_files = [f for f in os.listdir(ASS_DIR) if os.path.splitext(f)[1].lower() in SUPPORTED_SUBTITLE_EXTS]
    subtitle_files.sort()
    print(f"Found {len(subtitle_files)} subtitle files ({', '.join(SUPPORTED_SUBTITLE_EXTS)}).")
    print(
        "[INFO] Slice options: "
        f"dry_run={dry_run}, auto_filter_non_dialogue={auto_filter_non_dialogue}"
    )

    global_idx = 0
    total_slices = 0

    for subtitle_name in subtitle_files:
        base = os.path.splitext(subtitle_name)[0]
        wav_name = base + ".wav"
        wav_path = os.path.join(AUDIO_DIR, wav_name)
        subtitle_path = os.path.join(ASS_DIR, subtitle_name)

        if not os.path.exists(wav_path):
            print(f"[WARN] WAV not found for {subtitle_name} -> {wav_name}, skip.")
            continue

        raw_dialogues, subtitle_format = parse_subtitle_dialogues(subtitle_path)
        if subtitle_format is None:
            print(f"[WARN] Unsupported subtitle format: {subtitle_name}")
            continue

        print(f"[INFO] Processing {subtitle_name} ({subtitle_format}) with {wav_name}")
        if not raw_dialogues:
            print(f"[WARN] No dialogues in {subtitle_name}")
            continue

        filtered_dialogues, filter_mode = filter_dialogues_for_speech(
            raw_dialogues, auto_filter_non_dialogue=auto_filter_non_dialogue
        )
        slice_candidates = _filter_by_duration(filtered_dialogues, min_duration=0.3, max_duration=15.0)
        if not slice_candidates:
            print(f"[WARN] No slice candidates after filtering in {subtitle_name}")
            continue

        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono

        episode_slice_count = 0
        for d in slice_candidates:
            start_s = d["start"]
            end_s = d["end"]
            start_idx = int(start_s * sample_rate)
            end_idx = int(end_s * sample_rate)
            if end_idx <= start_idx or end_idx > waveform.shape[1]:
                continue

            if dry_run:
                episode_slice_count += 1
                continue

            slice_wave = waveform[:, start_idx:end_idx]
            out_stem = f"{base}__{global_idx:06d}"
            wav_out_name = f"{out_stem}.wav"
            txt_out_name = f"{out_stem}.txt"

            wav_out_path = os.path.join(OUT_DIR, wav_out_name)
            txt_out_path = os.path.join(DATA_DIR, "transcribe", txt_out_name)

            torchaudio.save(wav_out_path, slice_wave, sample_rate)

            with open(txt_out_path, "w", encoding="utf-8") as tf:
                # 자막 스타일/화자 정보를 바탕으로 화자 힌트를 같이 저장
                actor = _extract_actor(d)
                tf.write(f"[{actor}] {d['text']}\n")

            global_idx += 1
            episode_slice_count += 1

        total_slices += episode_slice_count
        print(
            f"[INFO] Done {subtitle_name}: slices={episode_slice_count}, "
            f"filter_mode={filter_mode}, duration_filtered={len(slice_candidates)}"
        )

    print(
        f"[INFO] 자막 기반 슬라이싱이 완료되었습니다. total_slices={total_slices}, dry_run={dry_run}"
    )

if __name__ == "__main__":
    run_ass_slice()
