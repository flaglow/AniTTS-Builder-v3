import bisect
import json
import math
import os
from datetime import datetime, timezone

import torch


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)


def load_sync_profile(path: str):
    if not path:
        return {"version": 1, "episodes": {}}
    if not os.path.exists(path):
        return {"version": 1, "episodes": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {"version": 1, "episodes": {}}
    if not isinstance(data, dict):
        return {"version": 1, "episodes": {}}
    episodes = data.get("episodes")
    if not isinstance(episodes, dict):
        data["episodes"] = {}
    if "version" not in data:
        data["version"] = 1
    return data


def save_sync_profile(path: str, profile: dict):
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)


def write_sync_report(path: str, payload: dict):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _quantile(x: torch.Tensor, q: float):
    if x.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    q = min(1.0, max(0.0, float(q)))
    try:
        return torch.quantile(x, q)
    except Exception:
        sorted_vals, _ = torch.sort(x)
        idx = int(round(q * (sorted_vals.numel() - 1)))
        return sorted_vals[idx]


def _fill_short_gaps(activity: torch.Tensor, max_gap_frames: int):
    if activity.numel() == 0 or max_gap_frames <= 0:
        return activity
    out = activity.clone()
    n = out.shape[0]
    i = 0
    while i < n:
        if out[i] > 0.5:
            i += 1
            continue
        j = i + 1
        while j < n and out[j] <= 0.5:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap_frames:
            out[i:j] = 1.0
        i = j
    return out


def _frame_audio_activity(mono_waveform: torch.Tensor, sample_rate: int, resolution_sec: float):
    x = mono_waveform.float().flatten()
    hop = max(1, _safe_int(round(sample_rate * resolution_sec), 1))
    n_frames = x.numel() // hop
    if n_frames <= 0:
        return torch.zeros(0, dtype=torch.float32)
    x = x[: n_frames * hop].reshape(n_frames, hop)
    rms = torch.sqrt(torch.mean(x * x, dim=1) + 1e-9)
    thr = _quantile(rms, 0.67)
    activity = (rms > thr).to(torch.float32)
    max_gap_frames = max(1, _safe_int(round(0.12 / resolution_sec), 1))
    return _fill_short_gaps(activity, max_gap_frames=max_gap_frames)


def build_subtitle_activity(dialogues, total_duration_sec: float, resolution_sec: float):
    total_frames = max(1, _safe_int(math.ceil(total_duration_sec / resolution_sec), 1))
    act = torch.zeros(total_frames, dtype=torch.float32)
    for d in dialogues:
        s = _safe_float(d.get("start", 0.0), 0.0)
        e = _safe_float(d.get("end", 0.0), 0.0)
        if e <= s:
            continue
        i = max(0, _safe_int(math.floor(s / resolution_sec), 0))
        j = min(total_frames, _safe_int(math.ceil(e / resolution_sec), total_frames))
        if j > i:
            act[i:j] = 1.0
    return act


def _alignment_score(audio_zm: torch.Tensor, sub_zm: torch.Tensor, lag_frames: int):
    if lag_frames >= 0:
        a = audio_zm[lag_frames:]
        b = sub_zm[: a.shape[0]]
    else:
        b = sub_zm[-lag_frames:]
        a = audio_zm[: b.shape[0]]
    if a.shape[0] < 50:
        return None
    return float(torch.mean(a * b).item())


def _estimate_lag(audio_activity: torch.Tensor, sub_activity: torch.Tensor, max_lag_frames: int):
    n = min(audio_activity.shape[0], sub_activity.shape[0])
    if n < 200:
        return 0, 0.0, 0.0
    audio_zm = audio_activity[:n] - torch.mean(audio_activity[:n])
    sub_zm = sub_activity[:n] - torch.mean(sub_activity[:n])

    scores = []
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        s = _alignment_score(audio_zm, sub_zm, lag)
        if s is None:
            continue
        scores.append((lag, s))

    if not scores:
        return 0, 0.0, 0.0

    scores.sort(key=lambda x: x[1], reverse=True)
    best_lag, best_score = scores[0]

    runner_up = 0.0
    for lag, score in scores[1:]:
        if abs(lag - best_lag) <= 2:
            continue
        runner_up = score
        break
    if runner_up <= 0:
        peak_ratio = 10.0 if best_score > 0 else 0.0
    else:
        peak_ratio = best_score / (runner_up + 1e-9)
    return best_lag, best_score, peak_ratio


def _fit_line(xs: torch.Tensor, ys: torch.Tensor):
    x_mean = torch.mean(xs)
    y_mean = torch.mean(ys)
    var_x = torch.sum((xs - x_mean) ** 2)
    if float(var_x.item()) < 1e-12:
        return 0.0, float(y_mean.item())
    cov_xy = torch.sum((xs - x_mean) * (ys - y_mean))
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    return float(slope.item()), float(intercept.item())


def _robust_fit_line(points):
    if len(points) < 2:
        if len(points) == 1:
            return 0.0, float(points[0][1]), 1.0, len(points)
        return 0.0, 0.0, 0.0, 0
    xs = torch.tensor([p[0] for p in points], dtype=torch.float32)
    ys = torch.tensor([p[1] for p in points], dtype=torch.float32)

    slope, intercept = _fit_line(xs, ys)
    preds = xs * slope + intercept
    residuals = torch.abs(ys - preds)
    mad = float(torch.median(residuals).item())
    if mad <= 1e-6:
        return slope, intercept, 1.0, len(points)

    threshold = max(0.08, 3.0 * mad)
    inlier_mask = residuals <= threshold
    inlier_count = int(torch.sum(inlier_mask).item())
    inlier_ratio = inlier_count / max(1, len(points))
    if inlier_count >= 2:
        slope, intercept = _fit_line(xs[inlier_mask], ys[inlier_mask])
    return slope, intercept, inlier_ratio, inlier_count


def apply_correction_to_dialogues(dialogues, offset_ms: float, drift_ppm: float, clip_end_sec: float):
    offset_sec = _safe_float(offset_ms, 0.0) / 1000.0
    factor = 1.0 + (_safe_float(drift_ppm, 0.0) / 1_000_000.0)
    out = []
    for d in dialogues:
        s = _safe_float(d.get("start", 0.0), 0.0) * factor + offset_sec
        e = _safe_float(d.get("end", 0.0), 0.0) * factor + offset_sec
        if e <= s:
            continue
        if e <= 0.0:
            continue
        if s >= clip_end_sec:
            continue
        s = max(0.0, s)
        e = min(clip_end_sec, e)
        if e <= s:
            continue
        c = dict(d)
        c["start"] = s
        c["end"] = e
        out.append(c)
    return out


def _estimate_local_lags(
    audio_activity: torch.Tensor,
    sub_activity: torch.Tensor,
    resolution_sec: float,
    window_sec: float = 180.0,
    step_sec: float = 120.0,
    max_local_shift_sec: float = 2.0,
):
    n = min(audio_activity.shape[0], sub_activity.shape[0])
    if n <= 0:
        return []

    audio = audio_activity[:n]
    sub = sub_activity[:n]
    w = max(200, _safe_int(round(window_sec / resolution_sec), 200))
    step = max(50, _safe_int(round(step_sec / resolution_sec), 50))
    max_lag = max(1, _safe_int(round(max_local_shift_sec / resolution_sec), 1))
    points = []

    for start in range(0, n, step):
        end = min(n, start + w)
        if end - start < 200:
            continue
        a_win = audio[start:end]
        s_win = sub[start:end]
        if float(torch.mean(a_win).item()) < 0.05:
            continue
        if float(torch.mean(s_win).item()) < 0.03:
            continue
        lag, score, _ = _estimate_lag(a_win, s_win, max_lag)
        if score <= 0:
            continue
        center_t = (start + (end - start) / 2.0) * resolution_sec
        points.append((center_t, lag * resolution_sec))
    return points


def _compute_residual_metrics(
    corrected_dialogues,
    audio_activity: torch.Tensor,
    resolution_sec: float,
    quality_target_ms: float,
):
    active = torch.where(audio_activity > 0.5)[0].tolist()
    if not active:
        return {
            "residual_p50_ms": 1_000_000_000.0,
            "inlier_ratio": 0.0,
            "num_residual_points": 0,
        }
    search_radius = max(1, _safe_int(round(2.0 / resolution_sec), 1))
    residuals_ms = []

    for d in corrected_dialogues:
        idx = _safe_int(round(_safe_float(d.get("start", 0.0), 0.0) / resolution_sec), 0)
        left = idx - search_radius
        right = idx + search_radius

        pos = bisect.bisect_left(active, idx)
        cands = []
        if pos < len(active):
            cands.append(active[pos])
        if pos > 0:
            cands.append(active[pos - 1])
        if pos + 1 < len(active):
            cands.append(active[pos + 1])
        cands = [c for c in cands if left <= c <= right]
        if not cands:
            continue
        nearest = min(cands, key=lambda c: abs(c - idx))
        residual_ms = abs((nearest - idx) * resolution_sec * 1000.0)
        residuals_ms.append(residual_ms)

    if not residuals_ms:
        return {
            "residual_p50_ms": 1_000_000_000.0,
            "inlier_ratio": 0.0,
            "num_residual_points": 0,
        }

    residuals_ms.sort()
    mid = len(residuals_ms) // 2
    if len(residuals_ms) % 2 == 0:
        p50 = (residuals_ms[mid - 1] + residuals_ms[mid]) / 2.0
    else:
        p50 = residuals_ms[mid]
    inlier = sum(1 for v in residuals_ms if v <= quality_target_ms) / len(residuals_ms)
    return {
        "residual_p50_ms": float(p50),
        "inlier_ratio": float(inlier),
        "num_residual_points": len(residuals_ms),
    }


def estimate_sync_parameters(
    mono_waveform: torch.Tensor,
    sample_rate: int,
    dialogues,
    max_shift_sec: float = 8.0,
    resolution_sec: float = 0.02,
    quality_target_ms: float = 200.0,
):
    if mono_waveform is None or mono_waveform.numel() == 0:
        return {
            "success": False,
            "offset_ms": 0.0,
            "drift_ppm": 0.0,
            "confidence": "low",
            "metrics": {
                "residual_p50_ms": 1_000_000_000.0,
                "inlier_ratio": 0.0,
                "peak_ratio": 0.0,
                "num_local_points": 0,
                "global_score": 0.0,
            },
        }

    total_duration = mono_waveform.shape[-1] / float(sample_rate)
    audio_activity = _frame_audio_activity(mono_waveform, sample_rate, resolution_sec)
    subtitle_activity = build_subtitle_activity(dialogues, total_duration, resolution_sec)

    max_lag_frames = max(1, _safe_int(round(max_shift_sec / resolution_sec), 1))
    lag_frames, global_score, peak_ratio = _estimate_lag(audio_activity, subtitle_activity, max_lag_frames)
    base_offset_sec = lag_frames * resolution_sec

    offset_ms = base_offset_sec * 1000.0
    drift_ppm = 0.0
    local_points = []
    local_inlier_ratio = 0.0

    shifted_dialogues = apply_correction_to_dialogues(
        dialogues=dialogues,
        offset_ms=offset_ms,
        drift_ppm=0.0,
        clip_end_sec=total_duration,
    )
    shifted_sub = build_subtitle_activity(shifted_dialogues, total_duration, resolution_sec)
    local_points = _estimate_local_lags(
        audio_activity=audio_activity,
        sub_activity=shifted_sub,
        resolution_sec=resolution_sec,
        window_sec=180.0,
        step_sec=120.0,
        max_local_shift_sec=2.0,
    )
    if len(local_points) >= 2:
        slope, intercept, local_inlier_ratio, _ = _robust_fit_line(local_points)
        slope = min(0.0015, max(-0.0015, slope))
        drift_ppm = slope * 1_000_000.0
        offset_ms = (base_offset_sec + intercept) * 1000.0

    corrected = apply_correction_to_dialogues(
        dialogues=dialogues,
        offset_ms=offset_ms,
        drift_ppm=drift_ppm,
        clip_end_sec=total_duration,
    )
    residual = _compute_residual_metrics(
        corrected_dialogues=corrected,
        audio_activity=audio_activity,
        resolution_sec=resolution_sec,
        quality_target_ms=quality_target_ms,
    )

    metrics = {
        "residual_p50_ms": residual["residual_p50_ms"],
        "inlier_ratio": residual["inlier_ratio"],
        "peak_ratio": float(peak_ratio),
        "num_local_points": len(local_points),
        "local_inlier_ratio": float(local_inlier_ratio),
        "global_score": float(global_score),
    }

    success = (
        residual["residual_p50_ms"] <= quality_target_ms
        and residual["inlier_ratio"] >= 0.60
        and peak_ratio >= 1.15
    )
    if success:
        confidence = "high"
    elif (
        residual["residual_p50_ms"] <= quality_target_ms * 1.5
        and residual["inlier_ratio"] >= 0.45
        and peak_ratio >= 1.05
    ):
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "success": bool(success),
        "offset_ms": float(offset_ms),
        "drift_ppm": float(drift_ppm),
        "confidence": confidence,
        "metrics": metrics,
    }


def build_profile_entry(
    status: str,
    confidence: str,
    offset_ms: float,
    drift_ppm: float,
    metrics: dict | None = None,
    manual_override: dict | None = None,
):
    entry = {
        "offset_ms": float(offset_ms),
        "drift_ppm": float(drift_ppm),
        "status": status,
        "confidence": confidence,
        "metrics": metrics or {},
        "manual_override": manual_override,
        "updated_at": _utc_now_iso(),
    }
    return entry
