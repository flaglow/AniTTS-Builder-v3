from dataclasses import dataclass
import subprocess

import psutil


@dataclass
class ResourceSnapshot:
    cpu_percent: float
    ram_total_gb: float
    ram_available_gb: float
    ram_percent: float
    gpu_util_percent: float | None = None
    vram_total_mb: int | None = None
    vram_used_mb: int | None = None
    vram_free_mb: int | None = None
    gpu_source: str | None = None


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def _read_nvidia_smi(device_id):
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    rows = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            (
                _safe_int(parts[0]),
                _safe_float(parts[1]),
                _safe_int(parts[2]),
                _safe_int(parts[3]),
                _safe_int(parts[4]),
            )
        )

    if not rows:
        return None

    if device_id is None:
        selected = rows[0]
    else:
        selected = None
        for row in rows:
            if row[0] == int(device_id):
                selected = row
                break
        if selected is None:
            selected = rows[0]

    _, gpu_util, vram_total, vram_used, vram_free = selected
    return gpu_util, vram_total, vram_used, vram_free, "nvidia-smi"


def _read_torch_vram(device_id):
    try:
        import torch
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    try:
        idx = int(device_id) if device_id is not None else torch.cuda.current_device()
    except Exception:
        idx = 0

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
    except Exception:
        return None

    vram_total = int(total_bytes // (1024 * 1024))
    vram_free = int(free_bytes // (1024 * 1024))
    vram_used = vram_total - vram_free
    return None, vram_total, vram_used, vram_free, "torch"


def collect_resource_snapshot(device_id=None):
    cpu_percent = float(psutil.cpu_percent(interval=0.2))
    vm = psutil.virtual_memory()

    snapshot = ResourceSnapshot(
        cpu_percent=cpu_percent,
        ram_total_gb=float(vm.total / (1024**3)),
        ram_available_gb=float(vm.available / (1024**3)),
        ram_percent=float(vm.percent),
    )

    gpu_values = _read_nvidia_smi(device_id)
    if gpu_values is None:
        gpu_values = _read_torch_vram(device_id)

    if gpu_values is not None:
        (
            snapshot.gpu_util_percent,
            snapshot.vram_total_mb,
            snapshot.vram_used_mb,
            snapshot.vram_free_mb,
            snapshot.gpu_source,
        ) = gpu_values

    return snapshot


def recommend_batch_size(base_batch, current_batch, snapshot, max_batch=4):
    base_batch = max(1, int(base_batch or 1))
    current_batch = max(1, int(current_batch or base_batch))
    max_batch = max(1, int(max_batch or base_batch))

    vram_free = snapshot.vram_free_mb
    gpu_util = snapshot.gpu_util_percent

    hard_pressure = (
        snapshot.cpu_percent >= 90.0
        or snapshot.ram_available_gb < 6.0
        or (gpu_util is not None and gpu_util >= 95.0)
        or (vram_free is not None and vram_free < 3072)
    )
    soft_pressure = (
        snapshot.cpu_percent >= 80.0
        or snapshot.ram_available_gb < 10.0
        or (gpu_util is not None and gpu_util >= 90.0)
        or (vram_free is not None and vram_free < 6144)
    )
    relaxed = (
        snapshot.cpu_percent <= 75.0
        and snapshot.ram_available_gb >= 12.0
        and (gpu_util is None or gpu_util <= 85.0)
        and (vram_free is None or vram_free >= 8192)
    )
    very_relaxed = (
        snapshot.cpu_percent <= 60.0
        and snapshot.ram_available_gb >= 24.0
        and (gpu_util is None or gpu_util <= 70.0)
        and (vram_free is None or vram_free >= 12288)
    )

    target_batch = current_batch
    if hard_pressure:
        target_batch = 1
    elif soft_pressure:
        target_batch = max(1, min(base_batch, current_batch))
    elif very_relaxed:
        target_batch = min(max_batch, max(base_batch, 4))
    elif relaxed:
        target_batch = min(max_batch, max(base_batch, 2))

    # Scale-up smoothing: +1 per probe. Scale-down is immediate.
    if target_batch > current_batch:
        target_batch = min(target_batch, current_batch + 1)

    return max(1, min(max_batch, target_batch))


def format_snapshot(snapshot):
    gpu_util = "na" if snapshot.gpu_util_percent is None else f"{snapshot.gpu_util_percent:.1f}%"
    vram_free = "na" if snapshot.vram_free_mb is None else f"{snapshot.vram_free_mb}MB"
    return (
        f"cpu={snapshot.cpu_percent:.1f}% "
        f"ram_avail={snapshot.ram_available_gb:.1f}GB "
        f"gpu_util={gpu_util} "
        f"vram_free={vram_free}"
    )
