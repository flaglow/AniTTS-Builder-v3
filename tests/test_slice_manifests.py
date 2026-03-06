import csv
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = SimpleNamespace(load=None, save=None)

if "torch" not in sys.modules:
    class _DummyCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    class _DummyMps:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    class _DummyBackends:
        mps = _DummyMps()

    class _DummyNN:
        class Module:
            pass

    sys.modules["torch"] = SimpleNamespace(
        float16="float16",
        cuda=_DummyCuda(),
        backends=_DummyBackends(),
        mps=_DummyMps(),
        nn=_DummyNN(),
    )

if "transformers" not in sys.modules:
    class _DummyProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(tokenizer=None, feature_extractor=None)

    class _DummyModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(to=lambda _device: SimpleNamespace())

    sys.modules["transformers"] = SimpleNamespace(
        pipeline=lambda *args, **kwargs: SimpleNamespace(),
        WhisperProcessor=_DummyProcessor,
        WhisperForConditionalGeneration=_DummyModel,
    )

if "numpy" not in sys.modules:
    sys.modules["numpy"] = SimpleNamespace(
        log10=lambda x: x,
        abs=abs,
        sqrt=lambda x: x ** 0.5,
        mean=lambda x: 0.0,
    )

import module.ass_slice as ass_slice
import module.whisper as whisper


class _FakeWaveform:
    def __init__(self, channels=1, length=5000):
        self.shape = (channels, length)

    def mean(self, dim=0, keepdim=True):
        return _FakeWaveform(channels=1, length=self.shape[1])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            time_slice = key[1]
        else:
            time_slice = key

        if isinstance(time_slice, slice):
            start = 0 if time_slice.start is None else int(time_slice.start)
            end = self.shape[1] if time_slice.stop is None else int(time_slice.stop)
            return _FakeWaveform(channels=1, length=max(0, end - start))

        return _FakeWaveform(channels=1, length=1)


class TestWhisperSliceManifest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self._tmp.name, "data")
        self.wav_output_dir = os.path.join(self.data_dir, "audio_wav")
        self.mp3_dir = os.path.join(self.data_dir, "audio_mp3")
        os.makedirs(self.wav_output_dir, exist_ok=True)
        os.makedirs(self.mp3_dir, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, path):
        with open(path, "wb") as f:
            f.write(b"")

    def test_save_slices_writes_prefixed_names_and_manifest(self):
        mp3_a = os.path.join(self.mp3_dir, "ep01.mp3")
        mp3_b = os.path.join(self.mp3_dir, "ep02.mp3")
        self._touch(mp3_a)
        self._touch(mp3_b)

        info = [
            (
                mp3_a,
                [
                    {"start": 0.1, "end": 0.3, "text": "hello"},
                    {"start": 1.0, "end": 1.2, "text": "world"},
                ],
            ),
            (mp3_b, [{"start": 2.5, "end": 3.0, "text": "bye"}]),
        ]
        saved_files = []

        def _fake_save(path, _wave, _sr):
            saved_files.append(os.path.basename(path))
            with open(path, "wb") as f:
                f.write(b"wav")

        with patch.object(whisper.torchaudio, "load", return_value=(_FakeWaveform(), 100)):
            with patch.object(whisper.torchaudio, "save", side_effect=_fake_save):
                with patch.object(whisper.os, "remove", side_effect=lambda _p: None):
                    whisper.save_slices(info, self.wav_output_dir)

        self.assertEqual(
            saved_files,
            ["ep01_000000.wav", "ep01_000001.wav", "ep02_000000.wav"],
        )

        manifest_path = os.path.join(self.data_dir, "whisper_slices.csv")
        self.assertTrue(os.path.exists(manifest_path))

        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 3)
        self.assertEqual([r["filename"] for r in rows], saved_files)
        self.assertEqual([r["index"] for r in rows], ["0", "1", "0"])
        self.assertEqual([r["timestamp_start"] for r in rows], ["0.100000", "1.000000", "2.500000"])
        self.assertEqual([r["timestamp_end"] for r in rows], ["0.300000", "1.200000", "3.000000"])
        self.assertEqual([r["transcript"] for r in rows], ["hello", "world", "bye"])


class TestSubtitleSliceManifest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = os.path.join(self._tmp.name, "data")
        self.audio_dir = os.path.join(self.data_dir, "audio_wav")
        self.sub_dir = os.path.join(self.data_dir, "transcribe")
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.sub_dir, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _prepare_ass_and_wav(self):
        ass_path = os.path.join(self.sub_dir, "ep01.ass")
        wav_path = os.path.join(self.audio_dir, "ep01.wav")
        self._write(
            ass_path,
            "[Script Info]\n"
            "Title: Test\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            "Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,line1\n"
            "Dialogue: 0,0:00:02.50,0:00:03.50,Default,,0,0,0,,line2\n",
        )
        with open(wav_path, "wb") as f:
            f.write(b"")

    def test_run_ass_slice_writes_manifest_when_not_dry_run(self):
        self._prepare_ass_and_wav()
        saved_files = []

        def _fake_save(path, _wave, _sr):
            saved_files.append(os.path.basename(path))
            with open(path, "wb") as f:
                f.write(b"wav")

        with patch.object(ass_slice, "DATA_DIR", self.data_dir):
            with patch.object(ass_slice, "AUDIO_DIR", self.audio_dir):
                with patch.object(ass_slice, "ASS_DIR", self.sub_dir):
                    with patch.object(ass_slice, "OUT_DIR", self.audio_dir):
                        with patch.object(ass_slice.torchaudio, "load", return_value=(_FakeWaveform(), 100)):
                            with patch.object(ass_slice.torchaudio, "save", side_effect=_fake_save):
                                ass_slice.run_ass_slice(dry_run=False, auto_filter_non_dialogue=True)

        self.assertEqual(saved_files, ["ep01__000000.wav", "ep01__000001.wav"])

        manifest_path = os.path.join(self.data_dir, "subtitle_slices.csv")
        self.assertTrue(os.path.exists(manifest_path))

        with open(manifest_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 2)
        self.assertEqual([r["filename"] for r in rows], saved_files)
        self.assertEqual([r["index"] for r in rows], ["0", "1"])
        self.assertEqual([r["timestamp_start"] for r in rows], ["1.000000", "2.500000"])
        self.assertEqual([r["timestamp_end"] for r in rows], ["2.000000", "3.500000"])
        self.assertEqual([r["transcript"] for r in rows], ["line1", "line2"])

    def test_run_ass_slice_dry_run_does_not_write_manifest(self):
        self._prepare_ass_and_wav()

        with patch.object(ass_slice, "DATA_DIR", self.data_dir):
            with patch.object(ass_slice, "AUDIO_DIR", self.audio_dir):
                with patch.object(ass_slice, "ASS_DIR", self.sub_dir):
                    with patch.object(ass_slice, "OUT_DIR", self.audio_dir):
                        with patch.object(ass_slice.torchaudio, "load", return_value=(_FakeWaveform(), 100)):
                            with patch.object(ass_slice.torchaudio, "save", side_effect=lambda *_args, **_kwargs: None):
                                ass_slice.run_ass_slice(dry_run=True, auto_filter_non_dialogue=True)

        manifest_path = os.path.join(self.data_dir, "subtitle_slices.csv")
        self.assertFalse(os.path.exists(manifest_path))


if __name__ == "__main__":
    unittest.main()
