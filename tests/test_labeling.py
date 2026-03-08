import csv
import os
import sys
import tempfile
import types
import unittest

try:
    import numpy as np

    HAS_NUMPY = all(
        [
            hasattr(np, "ndarray"),
            hasattr(np, "float32"),
            hasattr(np, "asarray"),
            hasattr(np, "stack"),
            hasattr(np, "empty"),
            hasattr(np, "matmul"),
            hasattr(np, "dot"),
            hasattr(getattr(np, "linalg", None), "norm"),
        ]
    )
    if not HAS_NUMPY:
        raise ImportError("incomplete numpy module")
except Exception:
    HAS_NUMPY = False
    np = types.ModuleType("numpy")
    np.ndarray = object
    np.float32 = float
    np.linalg = types.SimpleNamespace(norm=lambda *_args, **_kwargs: 0.0)
    np.asarray = lambda *_args, **_kwargs: []
    np.stack = lambda *_args, **_kwargs: []
    np.empty = lambda *_args, **_kwargs: []
    np.matmul = lambda *_args, **_kwargs: []
    np.dot = lambda *_args, **_kwargs: 0.0
    sys.modules["numpy"] = np


if "torch" not in sys.modules:
    torch_mod = types.ModuleType("torch")
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch_mod.mps = types.SimpleNamespace(empty_cache=lambda: None)
    torch_mod.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch_mod.device = lambda name: name
    torch_mod.set_num_threads = lambda *_args, **_kwargs: None
    torch_mod.get_num_threads = lambda: 1
    torch_mod.hub = types.SimpleNamespace(set_dir=lambda *_args, **_kwargs: None, load=lambda *_args, **_kwargs: None)

    class _DummyNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_mod.no_grad = lambda: _DummyNoGrad()
    sys.modules["torch"] = torch_mod

    nn_mod = types.ModuleType("torch.nn")

    class _DummyModule:
        pass

    nn_mod.Module = _DummyModule
    sys.modules["torch.nn"] = nn_mod
    torch_mod.nn = nn_mod

    fn_mod = types.ModuleType("torch.nn.functional")
    fn_mod.normalize = lambda tensor, p=2, dim=1: tensor
    sys.modules["torch.nn.functional"] = fn_mod

if "torchaudio" not in sys.modules:
    torchaudio_mod = types.ModuleType("torchaudio")
    torchaudio_mod.load = None
    torchaudio_mod.save = None
    torchaudio_mod.info = None
    torchaudio_mod.functional = types.SimpleNamespace(resample=None)
    sys.modules["torchaudio"] = torchaudio_mod

import module.labeling as labeling
import module.clustering as clustering


@unittest.skipUnless(HAS_NUMPY, "numpy is required for labeling tests")
class TestLabeling(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.data_dir = os.path.join(self.root, "data")
        self.result_dir = os.path.join(self.data_dir, "result")
        os.makedirs(self.result_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.data_dir, "clustering_slices.csv")

    def tearDown(self):
        self._tmp.cleanup()

    def _write_manifest(self, rows):
        with open(self.manifest_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "index", "timestamp_start", "timestamp_end", "transcript", "cluster_dir"])
            writer.writerows(rows)

    def _touch(self, cluster_name, file_name):
        cluster_dir = os.path.join(self.result_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        file_path = os.path.join(cluster_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(b"wav")
        return file_path

    def test_context_order_and_label_choices(self):
        rows = []
        for idx in range(5):
            rows.append([f"u2_{idx}.wav", str(idx), "1.0", "2.0", "line", "clustering_2"])
            self._touch("clustering_2", f"u2_{idx}.wav")
        for idx in range(2):
            rows.append([f"u10_{idx}.wav", str(idx), "1.0", "2.0", "line", "clustering_10"])
            self._touch("clustering_10", f"u10_{idx}.wav")
        for idx in range(3):
            rows.append([f"la_{idx}.wav", str(idx), "1.0", "2.0", "line", "마키세 크리스"])
            self._touch("마키세 크리스", f"la_{idx}.wav")

        rows.append(["noise_0.wav", "0", "1.0", "2.0", "line", "noise"])
        self._touch("noise", "noise_0.wav")

        self._write_manifest(rows)

        context = labeling.build_labeling_context(self.manifest_path, self.result_dir)
        self.assertEqual(context["unlabeled_clusters"][0], "clustering_2")
        self.assertEqual(context["unlabeled_clusters"][1], "clustering_10")

        next_cluster, remaining = labeling.select_next_unlabeled_cluster(context, skipped_clusters=[])
        self.assertEqual(next_cluster, "clustering_2")
        self.assertEqual(remaining, 2)

        next_cluster, remaining = labeling.select_next_unlabeled_cluster(context, skipped_clusters=["clustering_2"])
        self.assertEqual(next_cluster, "clustering_10")
        self.assertEqual(remaining, 1)

        self.assertEqual(labeling.get_label_choices(context), ["마키세 크리스"])

    def test_recommend_label_for_cluster_with_custom_loader(self):
        rows = [
            ["u1.wav", "0", "1.0", "2.0", "u1", "clustering_1"],
            ["u2.wav", "1", "1.0", "2.0", "u2", "clustering_1"],
            ["a1.wav", "2", "1.0", "2.0", "a1", "label_a"],
            ["a2.wav", "3", "1.0", "2.0", "a2", "label_a"],
            ["b1.wav", "4", "1.0", "2.0", "b1", "label_b"],
            ["b2.wav", "5", "1.0", "2.0", "b2", "label_b"],
        ]
        self._write_manifest(rows)

        for cluster, names in {
            "clustering_1": ["u1.wav", "u2.wav"],
            "label_a": ["a1.wav", "a2.wav"],
            "label_b": ["b1.wav", "b2.wav"],
        }.items():
            for name in names:
                self._touch(cluster, name)

        vectors = {
            "clustering_1": {
                "u1.wav": np.array([1.0, 0.0], dtype=np.float32),
                "u2.wav": np.array([1.0, 0.0], dtype=np.float32),
            },
            "label_a": {
                "a1.wav": np.array([1.0, 0.0], dtype=np.float32),
                "a2.wav": np.array([1.0, 0.0], dtype=np.float32),
            },
            "label_b": {
                "b1.wav": np.array([0.0, 1.0], dtype=np.float32),
                "b2.wav": np.array([0.0, 1.0], dtype=np.float32),
            },
        }

        def loader(cluster_dir, _emb_cache):
            return vectors[os.path.basename(cluster_dir)]

        context = labeling.build_labeling_context(self.manifest_path, self.result_dir)
        recommendation, cache = labeling.recommend_label_for_cluster(
            context=context,
            target_cluster="clustering_1",
            embedding_cache={"clusters": {}},
            embedding_loader=loader,
            embeddings_cache_dir=self.root,
            top_k=3,
            centroid_top_n=2,
        )

        self.assertEqual(recommendation["recommended_label"], "label_a")
        self.assertGreaterEqual(recommendation["recommended_score"], 0.99)
        self.assertFalse(recommendation["confidence_warning"])
        self.assertEqual(len(recommendation["recommended_samples"]), 2)
        self.assertEqual(len(recommendation["target_samples"]), 2)
        self.assertIn("label_a", cache["clusters"])

    def test_recommendation_low_confidence_rule(self):
        rows = [
            ["u.wav", "0", "1.0", "2.0", "u", "clustering_1"],
            ["a.wav", "1", "1.0", "2.0", "a", "label_a"],
            ["b.wav", "2", "1.0", "2.0", "b", "label_b"],
        ]
        self._write_manifest(rows)
        self._touch("clustering_1", "u.wav")
        self._touch("label_a", "a.wav")
        self._touch("label_b", "b.wav")

        vectors = {
            "clustering_1": {"u.wav": np.array([1.0, 0.0], dtype=np.float32)},
            "label_a": {"a.wav": np.array([0.34, 0.94042516], dtype=np.float32)},
            "label_b": {"b.wav": np.array([0.33, 0.94398093], dtype=np.float32)},
        }

        def loader(cluster_dir, _emb_cache):
            return vectors[os.path.basename(cluster_dir)]

        context = labeling.build_labeling_context(self.manifest_path, self.result_dir)
        recommendation, _cache = labeling.recommend_label_for_cluster(
            context=context,
            target_cluster="clustering_1",
            embedding_cache={"clusters": {}},
            embedding_loader=loader,
            embeddings_cache_dir=self.root,
            top_k=2,
            centroid_top_n=2,
        )

        self.assertTrue(recommendation["confidence_warning"])
        self.assertIn("top1=", recommendation["confidence_reason"])
        self.assertIn("top1-top2=", recommendation["confidence_reason"])

    def test_apply_cluster_label_then_refresh_manifest(self):
        rows = [
            ["same.wav", "0", "1.0", "2.0", "same", "clustering_5"],
            ["unique.wav", "1", "1.0", "2.0", "unique", "clustering_5"],
        ]
        self._write_manifest(rows)

        self._touch("clustering_5", "same.wav")
        self._touch("clustering_5", "unique.wav")
        self._touch("라벨A", "same.wav")

        moved = labeling.apply_cluster_label(self.result_dir, "clustering_5", "라벨A")
        self.assertEqual(moved["moved_files"], 2)
        self.assertEqual(moved["overwritten_files"], 1)
        self.assertFalse(os.path.exists(os.path.join(self.result_dir, "clustering_5")))

        clustering.refresh_clustering_manifest_cluster_dirs(
            manifest_path=self.manifest_path,
            destination_folder=self.result_dir,
            append_missing_rows=True,
            manifest_scan_workers=1,
            prefer_process_pool=False,
        )

        with open(self.manifest_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        by_name = {r["filename"]: r for r in rows}
        self.assertEqual(by_name["same.wav"]["cluster_dir"], "라벨A")
        self.assertEqual(by_name["unique.wav"]["cluster_dir"], "라벨A")

    def test_validate_label_name(self):
        self.assertFalse(labeling.validate_label_name("   ")[0])
        self.assertFalse(labeling.validate_label_name("noise")[0])
        self.assertFalse(labeling.validate_label_name("clustering_12")[0])
        self.assertFalse(labeling.validate_label_name("a/b")[0])

        ok, normalized, _msg = labeling.validate_label_name("  오카베 린타로  ")
        self.assertTrue(ok)
        self.assertEqual(normalized, "오카베 린타로")


if __name__ == "__main__":
    unittest.main()
