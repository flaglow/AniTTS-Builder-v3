import os
import re
import torchaudio

from module.sync_calibration import (
    apply_correction_to_dialogues,
    build_profile_entry,
    estimate_sync_parameters,
    load_sync_profile,
    save_sync_profile,
    write_sync_report,
)

DATA_DIR = "./data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio_wav")
ASS_DIR = os.path.join(DATA_DIR, "transcribe")
OUT_DIR = os.path.join(DATA_DIR, "audio_wav")  # 그대로 audio_wav에 조각을 만듦
SYNC_LOG_DIR = "./logs/ass_sync"
DEFAULT_SYNC_PROFILE_PATH = os.path.join(DATA_DIR, "transcribe", "sync_profile.json")

os.makedirs(OUT_DIR, exist_ok=True)

time_pattern = re.compile(r"(\d+):(\d+):(\d+\.\d+)")
dialogue_prefix = "Dialogue:"
non_dialogue_style_pattern = re.compile(
    r"(?i)(op|ed|title|sign|kara|lyric|gasa|ruby|cell|mail|phone|cm|preview)"
)

def ass_time_to_sec(t: str) -> float:
    m = time_pattern.match(t.strip())
    if not m:
        return 0.0
    h, m_, s = m.groups()
    return int(h) * 3600 + int(m_) * 60 + float(s)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


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


def _manual_override_from_profile(episode_profile):
    if not isinstance(episode_profile, dict):
        return None
    override = episode_profile.get("manual_override")
    if not isinstance(override, dict):
        return None
    if "offset_ms" not in override and "drift_ppm" not in override:
        return None
    return {
        "offset_ms": _safe_float(override.get("offset_ms", 0.0), 0.0),
        "drift_ppm": _safe_float(override.get("drift_ppm", 0.0), 0.0),
    }


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

def run_ass_slice(
    enable_auto_sync: bool = True,
    manual_offset_ms: float = 0.0,
    manual_drift_ppm: float = 0.0,
    use_sync_profile: bool = True,
    sync_profile_path: str = DEFAULT_SYNC_PROFILE_PATH,
    dry_run: bool = False,
    auto_filter_non_dialogue: bool = True,
    max_shift_sec: float = 8.0,
    quality_target_ms: float = 200.0,
):
    os.makedirs(SYNC_LOG_DIR, exist_ok=True)
    os.makedirs(ASS_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    profile = load_sync_profile(sync_profile_path) if use_sync_profile else {"version": 1, "episodes": {}}
    episodes = profile.setdefault("episodes", {})

    ass_files = [f for f in os.listdir(ASS_DIR) if f.lower().endswith(".ass")]
    ass_files.sort()
    print(f"Found {len(ass_files)} ASS files.")
    print(
        "[INFO] Sync options: "
        f"auto_sync={enable_auto_sync}, manual_offset_ms={manual_offset_ms}, "
        f"manual_drift_ppm={manual_drift_ppm}, use_sync_profile={use_sync_profile}, "
        f"dry_run={dry_run}, auto_filter_non_dialogue={auto_filter_non_dialogue}"
    )

    global_idx = 0
    total_slices = 0
    needs_manual_count = 0

    for ass_name in ass_files:
        base = os.path.splitext(ass_name)[0]
        wav_name = base + ".wav"
        wav_path = os.path.join(AUDIO_DIR, wav_name)
        ass_path = os.path.join(ASS_DIR, ass_name)
        report_path = os.path.join(SYNC_LOG_DIR, f"{base}.json")
        episode_profile = episodes.get(ass_name, {})
        existing_manual_override = (
            episode_profile.get("manual_override")
            if isinstance(episode_profile, dict) and isinstance(episode_profile.get("manual_override"), dict)
            else None
        )

        if not os.path.exists(wav_path):
            print(f"[WARN] WAV not found for {ass_name} -> {wav_name}, skip.")
            entry = build_profile_entry(
                status="skipped_no_wav",
                confidence="low",
                offset_ms=0.0,
                drift_ppm=0.0,
                metrics={},
                manual_override=existing_manual_override,
            )
            if use_sync_profile:
                episodes[ass_name] = entry
            write_sync_report(
                report_path,
                {
                    "ass_name": ass_name,
                    "wav_name": wav_name,
                    "status": "skipped_no_wav",
                    "slices_created": 0,
                    "dry_run": dry_run,
                },
            )
            continue

        print(f"[INFO] Processing {ass_name} with {wav_name}")
        raw_dialogues = parse_ass_dialogues(ass_path)
        if not raw_dialogues:
            print(f"[WARN] No dialogues in {ass_name}")
            entry = build_profile_entry(
                status="no_dialogues",
                confidence="low",
                offset_ms=0.0,
                drift_ppm=0.0,
                metrics={},
                manual_override=existing_manual_override,
            )
            if use_sync_profile:
                episodes[ass_name] = entry
            write_sync_report(
                report_path,
                {
                    "ass_name": ass_name,
                    "wav_name": wav_name,
                    "status": "no_dialogues",
                    "slices_created": 0,
                    "dry_run": dry_run,
                },
            )
            continue

        filtered_dialogues, filter_mode = filter_dialogues_for_speech(
            raw_dialogues, auto_filter_non_dialogue=auto_filter_non_dialogue
        )
        slice_candidates = _filter_by_duration(filtered_dialogues, min_duration=0.3, max_duration=15.0)
        if not slice_candidates:
            print(f"[WARN] No slice candidates after filtering in {ass_name}")
            entry = build_profile_entry(
                status="no_candidates_after_filter",
                confidence="low",
                offset_ms=0.0,
                drift_ppm=0.0,
                metrics={},
                manual_override=existing_manual_override,
            )
            if use_sync_profile:
                episodes[ass_name] = entry
            write_sync_report(
                report_path,
                {
                    "ass_name": ass_name,
                    "wav_name": wav_name,
                    "status": "no_candidates_after_filter",
                    "filter_mode": filter_mode,
                    "line_counts": {
                        "raw": len(raw_dialogues),
                        "filtered": len(filtered_dialogues),
                        "duration_filtered": len(slice_candidates),
                    },
                    "slices_created": 0,
                    "dry_run": dry_run,
                },
            )
            continue

        waveform, sample_rate = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        audio_duration = waveform.shape[1] / float(sample_rate)

        profile_override = _manual_override_from_profile(episode_profile)
        has_ui_manual = abs(_safe_float(manual_offset_ms, 0.0)) > 1e-9 or abs(_safe_float(manual_drift_ppm, 0.0)) > 1e-9

        status = "no_sync"
        confidence = "low"
        metrics = {}
        offset_ms = 0.0
        drift_ppm = 0.0

        if profile_override is not None:
            status = "manual_override"
            confidence = "manual"
            offset_ms = profile_override["offset_ms"]
            drift_ppm = profile_override["drift_ppm"]
        elif has_ui_manual:
            status = "manual_input"
            confidence = "manual"
            offset_ms = _safe_float(manual_offset_ms, 0.0)
            drift_ppm = _safe_float(manual_drift_ppm, 0.0)
        elif enable_auto_sync:
            auto_result = estimate_sync_parameters(
                mono_waveform=waveform[0],
                sample_rate=sample_rate,
                dialogues=slice_candidates,
                max_shift_sec=max_shift_sec,
                resolution_sec=0.02,
                quality_target_ms=quality_target_ms,
            )
            offset_ms = _safe_float(auto_result.get("offset_ms", 0.0), 0.0)
            drift_ppm = _safe_float(auto_result.get("drift_ppm", 0.0), 0.0)
            confidence = str(auto_result.get("confidence", "low"))
            metrics = auto_result.get("metrics", {})
            if auto_result.get("success", False):
                status = "auto_applied"
            else:
                status = "needs_manual"
        elif use_sync_profile and isinstance(episode_profile, dict) and ("offset_ms" in episode_profile or "drift_ppm" in episode_profile):
            status = "profile_applied"
            confidence = str(episode_profile.get("confidence", "medium"))
            offset_ms = _safe_float(episode_profile.get("offset_ms", 0.0), 0.0)
            drift_ppm = _safe_float(episode_profile.get("drift_ppm", 0.0), 0.0)
            metrics = episode_profile.get("metrics", {}) if isinstance(episode_profile.get("metrics"), dict) else {}

        if status == "needs_manual":
            needs_manual_count += 1
            print(
                f"[WARN] Low confidence sync for {ass_name}: "
                f"offset_ms={offset_ms:.2f}, drift_ppm={drift_ppm:.2f}, metrics={metrics}"
            )
            entry = build_profile_entry(
                status=status,
                confidence=confidence,
                offset_ms=offset_ms,
                drift_ppm=drift_ppm,
                metrics=metrics,
                manual_override=existing_manual_override,
            )
            if use_sync_profile:
                episodes[ass_name] = entry
            write_sync_report(
                report_path,
                {
                    "ass_name": ass_name,
                    "wav_name": wav_name,
                    "status": status,
                    "confidence": confidence,
                    "filter_mode": filter_mode,
                    "line_counts": {
                        "raw": len(raw_dialogues),
                        "filtered": len(filtered_dialogues),
                        "duration_filtered": len(slice_candidates),
                    },
                    "applied": {
                        "offset_ms": offset_ms,
                        "drift_ppm": drift_ppm,
                    },
                    "metrics": metrics,
                    "slices_created": 0,
                    "dry_run": dry_run,
                },
            )
            continue

        corrected_dialogues = apply_correction_to_dialogues(
            dialogues=slice_candidates,
            offset_ms=offset_ms,
            drift_ppm=drift_ppm,
            clip_end_sec=audio_duration,
        )
        corrected_dialogues = _filter_by_duration(corrected_dialogues, min_duration=0.3, max_duration=15.0)

        episode_slice_count = 0
        for d in corrected_dialogues:
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
                # 화자 정보(ASS의 Name 필드)를 같이 저장
                actor = _extract_actor(d)
                tf.write(f"[{actor}] {d['text']}\n")

            global_idx += 1
            episode_slice_count += 1

        total_slices += episode_slice_count
        entry = build_profile_entry(
            status=status,
            confidence=confidence,
            offset_ms=offset_ms,
            drift_ppm=drift_ppm,
            metrics=metrics,
            manual_override=existing_manual_override,
        )
        if use_sync_profile:
            episodes[ass_name] = entry
        write_sync_report(
            report_path,
            {
                "ass_name": ass_name,
                "wav_name": wav_name,
                "status": status,
                "confidence": confidence,
                "filter_mode": filter_mode,
                "line_counts": {
                    "raw": len(raw_dialogues),
                    "filtered": len(filtered_dialogues),
                    "duration_filtered": len(slice_candidates),
                    "corrected": len(corrected_dialogues),
                },
                "applied": {
                    "offset_ms": offset_ms,
                    "drift_ppm": drift_ppm,
                },
                "metrics": metrics,
                "slices_created": episode_slice_count,
                "dry_run": dry_run,
            },
        )
        print(
            f"[INFO] Done {ass_name}: status={status}, confidence={confidence}, "
            f"offset_ms={offset_ms:.2f}, drift_ppm={drift_ppm:.2f}, slices={episode_slice_count}"
        )

    if use_sync_profile:
        save_sync_profile(sync_profile_path, profile)
    print(
        f"[INFO] ASS 기반 슬라이싱이 완료되었습니다. total_slices={total_slices}, "
        f"needs_manual={needs_manual_count}, dry_run={dry_run}"
    )

if __name__ == "__main__":
    run_ass_slice()
