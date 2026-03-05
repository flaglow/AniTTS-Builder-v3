import gc
from typing import Iterable, Optional

import torch


def _move_module_to_cpu(module, stage_name: str, label: str):
    try:
        module.to("cpu")
    except Exception as exc:
        print(f"[WARN] Failed to move {label} to CPU at stage '{stage_name}': {exc}")


def release_torch_resources(stage_name: str, objects: Optional[Iterable[object]] = None):
    """
    Best-effort release for model objects and Torch caches.
    """
    for obj in (objects or []):
        if obj is None:
            continue

        del_cache = getattr(obj, "del_cache", None)
        if callable(del_cache):
            try:
                del_cache()
            except Exception as exc:
                print(f"[WARN] del_cache failed at stage '{stage_name}': {exc}")

        model = getattr(obj, "model", None)
        if isinstance(model, torch.nn.Module):
            _move_module_to_cpu(model, stage_name, "obj.model")
            try:
                obj.model = None
            except Exception:
                pass

        if isinstance(obj, torch.nn.Module):
            _move_module_to_cpu(obj, stage_name, "obj")

    collected = gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[WARN] CUDA cache clear failed at stage '{stage_name}': {exc}")

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception as exc:
            print(f"[WARN] MPS cache clear failed at stage '{stage_name}': {exc}")

    print(f"[INFO] Memory cleanup finished for stage '{stage_name}'. gc_collected={collected}")
