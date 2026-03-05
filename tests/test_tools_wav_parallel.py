import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _DummyFFmpegError(Exception):
    def __init__(self, stderr=None):
        super().__init__("dummy ffmpeg error")
        self.stderr = stderr


if "ffmpeg" not in sys.modules:
    class _DummyFfmpegStream:
        def output(self, *args, **kwargs):
            return self

        def run(self, *args, **kwargs):
            return None

    def _dummy_ffmpeg_input(*args, **kwargs):
        return _DummyFfmpegStream()

    sys.modules["ffmpeg"] = SimpleNamespace(input=_dummy_ffmpeg_input, Error=_DummyFFmpegError)

if "tqdm" not in sys.modules:
    sys.modules["tqdm"] = SimpleNamespace(tqdm=lambda *args, **kwargs: None)

if "requests" not in sys.modules:
    sys.modules["requests"] = SimpleNamespace(get=lambda *args, **kwargs: None)

import module.tools as tools


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _RecordingExecutor:
    last_max_workers = None
    submitted = 0

    def __init__(self, max_workers):
        type(self).last_max_workers = max_workers
        type(self).submitted = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, *args, **kwargs):
        type(self).submitted += 1
        return _ImmediateFuture(fn(*args, **kwargs))


class TestWavConversionWorkers(unittest.TestCase):
    def test_auto_worker_policy(self):
        with patch.object(tools.os, "cpu_count", return_value=16):
            self.assertEqual(tools._auto_wav_convert_workers(), 4)
        with patch.object(tools.os, "cpu_count", return_value=2):
            self.assertEqual(tools._auto_wav_convert_workers(), 1)
        with patch.object(tools.os, "cpu_count", return_value=1):
            self.assertEqual(tools._auto_wav_convert_workers(), 1)

    def test_worker_resolution_argument_overrides_env(self):
        with patch.dict(os.environ, {"ANITTS_WAV_CONVERT_WORKERS": "3"}, clear=False):
            workers, source = tools._resolve_wav_convert_workers(2)
        self.assertEqual(workers, 2)
        self.assertEqual(source, "argument")

    def test_worker_resolution_uses_env(self):
        with patch.dict(os.environ, {"ANITTS_WAV_CONVERT_WORKERS": "3"}, clear=False):
            workers, source = tools._resolve_wav_convert_workers(None)
        self.assertEqual(workers, 3)
        self.assertEqual(source, "env")

    def test_worker_resolution_invalid_env_falls_back_to_auto(self):
        with patch.dict(os.environ, {"ANITTS_WAV_CONVERT_WORKERS": "bad"}, clear=False):
            with patch.object(tools, "_auto_wav_convert_workers", return_value=2):
                workers, source = tools._resolve_wav_convert_workers(None)
        self.assertEqual(workers, 2)
        self.assertEqual(source, "auto")


class TestBatchConvertToWav(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.input_dir = os.path.join(self._tmp.name, "input")
        self.output_dir = os.path.join(self._tmp.name, "output")
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _create_input(self, file_name):
        path = os.path.join(self.input_dir, file_name)
        with open(path, "wb") as f:
            f.write(b"test")

    def test_single_worker_path_is_sequential(self):
        self._create_input("a.mkv")
        self._create_input("b.mp4")

        with patch.object(tools, "convert_to_wav", return_value=True) as mocked_convert:
            with patch.object(tools, "ThreadPoolExecutor") as mocked_executor:
                tools.batch_convert_to_wav(self.input_dir, self.output_dir, workers=1)

        self.assertEqual(mocked_convert.call_count, 2)
        mocked_executor.assert_not_called()

    def test_parallel_path_uses_executor(self):
        self._create_input("a.mkv")
        self._create_input("b.mp4")
        self._create_input("c.webm")

        with patch.object(tools, "convert_to_wav", return_value=True) as mocked_convert:
            with patch.object(tools, "ThreadPoolExecutor", _RecordingExecutor):
                with patch.object(tools, "as_completed", side_effect=lambda futures: list(futures)):
                    tools.batch_convert_to_wav(self.input_dir, self.output_dir, workers=3)

        self.assertEqual(_RecordingExecutor.last_max_workers, 3)
        self.assertEqual(_RecordingExecutor.submitted, 3)
        self.assertEqual(mocked_convert.call_count, 3)

    def test_parallel_worker_count_is_clamped_to_job_count(self):
        self._create_input("a.mkv")
        self._create_input("b.mp4")

        with patch.object(tools, "convert_to_wav", return_value=True):
            with patch.object(tools, "ThreadPoolExecutor", _RecordingExecutor):
                with patch.object(tools, "as_completed", side_effect=lambda futures: list(futures)):
                    tools.batch_convert_to_wav(self.input_dir, self.output_dir, workers=8)

        self.assertEqual(_RecordingExecutor.last_max_workers, 2)
        self.assertEqual(_RecordingExecutor.submitted, 2)

    def test_output_name_collision_is_skipped(self):
        self._create_input("a.mkv")
        self._create_input("a.mp4")
        self._create_input("b.mkv")

        def _convert_result(input_path, _output_path):
            return not input_path.endswith("b.mkv")

        with patch.object(tools, "convert_to_wav", side_effect=_convert_result) as mocked_convert:
            tools.batch_convert_to_wav(self.input_dir, self.output_dir, workers=1)

        converted_inputs = sorted(os.path.basename(call.args[0]) for call in mocked_convert.call_args_list)
        self.assertEqual(converted_inputs, ["a.mkv", "b.mkv"])


if __name__ == "__main__":
    unittest.main()
