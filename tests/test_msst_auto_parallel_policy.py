import unittest

from module.model.MSST_WebUI.utils.system_resources import ResourceSnapshot, recommend_batch_size


def _snapshot(cpu, ram_available, gpu_util=None, vram_free=None):
    return ResourceSnapshot(
        cpu_percent=float(cpu),
        ram_total_gb=64.0,
        ram_available_gb=float(ram_available),
        ram_percent=0.0,
        gpu_util_percent=None if gpu_util is None else float(gpu_util),
        vram_total_mb=24576 if vram_free is not None else None,
        vram_used_mb=None,
        vram_free_mb=None if vram_free is None else int(vram_free),
        gpu_source="test",
    )


class TestMsstAutoParallelPolicy(unittest.TestCase):
    def test_scale_up_uses_smoothing(self):
        snap = _snapshot(cpu=35, ram_available=40, gpu_util=20, vram_free=20000)
        self.assertEqual(recommend_batch_size(1, 1, snap, max_batch=4), 2)
        self.assertEqual(recommend_batch_size(1, 2, snap, max_batch=4), 3)
        self.assertEqual(recommend_batch_size(1, 3, snap, max_batch=4), 4)

    def test_hard_pressure_drops_to_one_immediately(self):
        snap = _snapshot(cpu=95, ram_available=32, gpu_util=30, vram_free=18000)
        self.assertEqual(recommend_batch_size(2, 4, snap, max_batch=4), 1)

    def test_soft_pressure_falls_back_to_base_or_lower(self):
        snap = _snapshot(cpu=82, ram_available=32, gpu_util=40, vram_free=18000)
        self.assertEqual(recommend_batch_size(2, 4, snap, max_batch=4), 2)

    def test_missing_gpu_metrics_still_scales_up_from_cpu_ram(self):
        snap = _snapshot(cpu=40, ram_available=30, gpu_util=None, vram_free=None)
        self.assertEqual(recommend_batch_size(1, 1, snap, max_batch=4), 2)

    def test_vram_guard_can_force_hard_pressure(self):
        snap = _snapshot(cpu=20, ram_available=32, gpu_util=20, vram_free=2000)
        self.assertEqual(recommend_batch_size(2, 3, snap, max_batch=4), 1)

    def test_max_batch_cap_is_respected(self):
        snap = _snapshot(cpu=20, ram_available=40, gpu_util=10, vram_free=24000)
        self.assertEqual(recommend_batch_size(1, 3, snap, max_batch=3), 3)


if __name__ == "__main__":
    unittest.main()
