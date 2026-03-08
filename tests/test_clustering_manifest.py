import csv
import os
import sys
import tempfile
import time
import types
import unittest


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

if "numpy" not in sys.modules:
    sys.modules["numpy"] = types.ModuleType("numpy")

import module.clustering as clustering


class TestClusteringManifest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self._tmp.name, "data")
        self.wav_dir = os.path.join(self.data_dir, "audio_wav")
        self.result_dir = os.path.join(self.data_dir, "result")
        os.makedirs(self.wav_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_csv(self, path, header, rows):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def _touch_wav(self, folder, name):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, name)
        with open(path, "wb") as f:
            f.write(b"wav")
        return path

    def test_export_clustering_manifest_merges_slice_metadata_and_cluster_dir(self):
        whisper_csv = os.path.join(self.data_dir, "whisper_slices.csv")
        subtitle_csv = os.path.join(self.data_dir, "subtitle_slices.csv")
        self._write_csv(
            whisper_csv,
            ["filename", "index", "timestamp_start", "timestamp_end", "transcript"],
            [["ep01_000000.wav", "0", "1.000000", "1.500000", "hello"]],
        )
        self._write_csv(
            subtitle_csv,
            ["filename", "index", "timestamp_start", "timestamp_end", "transcript"],
            [["ep02__000001.wav", "1", "2.000000", "2.700000", "subtitle line"]],
        )

        self._touch_wav(os.path.join(self.result_dir, "clustering_0"), "ep01_000000.wav")
        self._touch_wav(os.path.join(self.result_dir, "noise"), "ep02__000001.wav")
        self._touch_wav(os.path.join(self.result_dir, "clustering_9"), "old.wav")
        self._touch_wav(os.path.join(self.result_dir, "clustering_2"), "ep03_000000.wav")

        manifest_path, rows_count = clustering.export_clustering_manifest(
            wav_folder=self.wav_dir,
            destination_folder=self.result_dir,
            included_filenames=[
                os.path.join(self.wav_dir, "ep01_000000.wav"),
                os.path.join(self.wav_dir, "ep02__000001.wav"),
                os.path.join(self.wav_dir, "ep03_000000.wav"),
            ],
        )

        self.assertTrue(os.path.exists(manifest_path))
        self.assertEqual(rows_count, 3)

        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        by_name = {r["filename"]: r for r in rows}
        self.assertEqual(sorted(by_name.keys()), ["ep01_000000.wav", "ep02__000001.wav", "ep03_000000.wav"])
        self.assertEqual(by_name["ep01_000000.wav"]["cluster_dir"], "clustering_0")
        self.assertEqual(by_name["ep01_000000.wav"]["transcript"], "hello")
        self.assertEqual(by_name["ep02__000001.wav"]["cluster_dir"], "noise")
        self.assertEqual(by_name["ep02__000001.wav"]["transcript"], "subtitle line")
        self.assertEqual(by_name["ep03_000000.wav"]["cluster_dir"], "clustering_2")
        self.assertEqual(by_name["ep03_000000.wav"]["index"], "")
        self.assertEqual(by_name["ep03_000000.wav"]["timestamp_start"], "")
        self.assertEqual(by_name["ep03_000000.wav"]["timestamp_end"], "")
        self.assertEqual(by_name["ep03_000000.wav"]["transcript"], "")

    def test_refresh_clustering_manifest_cluster_dirs_updates_existing_and_appends_missing(self):
        clustering_csv = os.path.join(self.data_dir, "clustering_slices.csv")
        subtitle_csv = os.path.join(self.data_dir, "subtitle_slices.csv")

        self._write_csv(
            clustering_csv,
            ["filename", "index", "timestamp_start", "timestamp_end", "transcript", "cluster_dir"],
            [
                ["ep01_000000.wav", "0", "1.0", "1.5", "hello", "clustering_old"],
                ["ep02_000000.wav", "1", "2.0", "2.5", "bye", "clustering_old"],
            ],
        )
        self._write_csv(
            subtitle_csv,
            ["filename", "index", "timestamp_start", "timestamp_end", "transcript"],
            [["ep03_000000.wav", "2", "3.0", "3.5", "new line"]],
        )

        self._touch_wav(os.path.join(self.result_dir, "clustering_5"), "ep01_000000.wav")
        self._touch_wav(os.path.join(self.result_dir, "noise"), "ep03_000000.wav")

        _, rows_count, changed_rows, added_rows, missing_rows = clustering.refresh_clustering_manifest_cluster_dirs(
            manifest_path=clustering_csv,
            destination_folder=self.result_dir,
            append_missing_rows=True,
            manifest_scan_workers=1,
            prefer_process_pool=False,
        )

        self.assertEqual(rows_count, 3)
        self.assertEqual(changed_rows, 2)
        self.assertEqual(added_rows, 1)
        self.assertEqual(missing_rows, 1)

        with open(clustering_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        by_name = {r["filename"]: r for r in rows}

        self.assertEqual(by_name["ep01_000000.wav"]["cluster_dir"], "clustering_5")
        self.assertEqual(by_name["ep02_000000.wav"]["cluster_dir"], "")
        self.assertEqual(by_name["ep03_000000.wav"]["cluster_dir"], "noise")
        self.assertEqual(by_name["ep03_000000.wav"]["index"], "2")
        self.assertEqual(by_name["ep03_000000.wav"]["timestamp_start"], "3.0")
        self.assertEqual(by_name["ep03_000000.wav"]["timestamp_end"], "3.5")
        self.assertEqual(by_name["ep03_000000.wav"]["transcript"], "new line")

    def test_refresh_clustering_manifest_uses_latest_file_mtime_for_duplicate_filenames(self):
        clustering_csv = os.path.join(self.data_dir, "clustering_slices.csv")
        self._write_csv(
            clustering_csv,
            ["filename", "index", "timestamp_start", "timestamp_end", "transcript", "cluster_dir"],
            [["dup.wav", "0", "0.0", "1.0", "dup", "old_cluster"]],
        )

        old_path = self._touch_wav(os.path.join(self.result_dir, "clustering_1"), "dup.wav")
        new_path = self._touch_wav(os.path.join(self.result_dir, "clustering_2"), "dup.wav")
        now = time.time()
        os.utime(old_path, (now - 100, now - 100))
        os.utime(new_path, (now, now))

        clustering.refresh_clustering_manifest_cluster_dirs(
            manifest_path=clustering_csv,
            destination_folder=self.result_dir,
            append_missing_rows=False,
            manifest_scan_workers=1,
            prefer_process_pool=False,
        )

        with open(clustering_csv, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["cluster_dir"], "clustering_2")


if __name__ == "__main__":
    unittest.main()
