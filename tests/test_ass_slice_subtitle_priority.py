import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace


if "torchaudio" not in sys.modules:
    sys.modules["torchaudio"] = SimpleNamespace(load=None, save=None)


class _FakeSubs:
    def __init__(self, events):
        self.events = list(events)

    def save(self, output_path, format_=None):
        if format_ is not None and str(format_).lower() != "ass":
            raise ValueError("Only ASS output is supported by fake pysubs2")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[Script Info]\n")
            f.write("Title: Fake\n\n")
            f.write("[Events]\n")
            f.write(
                "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
            )
            for idx, text in enumerate(self.events):
                f.write(
                    "Dialogue: 0,"
                    f"00:00:{idx:05.2f},"
                    f"00:00:{idx + 1:05.2f},"
                    "Default,,0,0,0,,"
                    f"{text}\n"
                )


def _clean_tags(text):
    cleaned = re.sub(r"(?i)<br\s*/?>", " ", text)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("&nbsp;", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _parse_srt_events(text):
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    events = []
    for block in blocks:
        rows = [row.strip() for row in block.split("\n") if row.strip()]
        if not rows:
            continue
        time_line_idx = next((i for i, row in enumerate(rows) if "-->" in row), -1)
        if time_line_idx < 0:
            continue
        payload = " ".join(rows[time_line_idx + 1 :]).strip()
        payload = _clean_tags(payload)
        if payload:
            events.append(payload)
    return events


def _parse_smi_events(text):
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(re.finditer(r"(?is)<sync\b[^>]*?start\s*=\s*(\d+)[^>]*>", normalized))
    events = []
    for i, match in enumerate(matches):
        next_match = matches[i + 1] if i + 1 < len(matches) else None
        if next_match is None:
            continue
        segment = normalized[match.end() : next_match.start()]
        payload = _clean_tags(segment)
        if payload:
            events.append(payload)
    return events


def _fake_pysubs2_load(path, encoding=None):
    with open(path, "r", encoding=encoding or "utf-8") as f:
        text = f.read()

    ext = os.path.splitext(path)[1].lower()
    if ext == ".srt":
        events = _parse_srt_events(text)
    elif ext == ".smi":
        events = _parse_smi_events(text)
    elif ext == ".ass":
        events = [line for line in text.splitlines() if line.lstrip().startswith("Dialogue:")]
    else:
        events = []
    return _FakeSubs(events)


if "pysubs2" not in sys.modules:
    sys.modules["pysubs2"] = SimpleNamespace(load=_fake_pysubs2_load)

import module.ass_slice as ass_slice


class TestAssSliceSubtitlePriority(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.transcribe_dir = os.path.join(self._tmp.name, "transcribe")
        os.makedirs(self.transcribe_dir, exist_ok=True)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_file(self, name, content, encoding="utf-8"):
        path = os.path.join(self.transcribe_dir, name)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)
        return path

    def test_utf16_smi_can_be_converted_to_ass(self):
        smi_path = self._write_file(
            "ep01.smi",
            "<SAMI><BODY>\n"
            "<SYNC Start=1000><P Class=KRCC>안녕\n"
            "<SYNC Start=2500><P Class=KRCC>&nbsp;\n"
            "<SYNC Start=4000><P Class=KRCC>마지막\n"
            "</BODY></SAMI>\n",
            encoding="utf-16",
        )
        ass_path = os.path.join(self.transcribe_dir, "ep01.ass")

        converted, reason = ass_slice.ensure_ass_from_smi(smi_path, ass_path)
        self.assertTrue(converted)
        self.assertEqual(reason, "converted")
        self.assertTrue(os.path.exists(ass_path))

        with open(ass_path, "r", encoding="utf-8") as f:
            ass_text = f.read()
        self.assertIn("[Events]", ass_text)
        self.assertIn("Dialogue:", ass_text)

    def test_srt_can_be_converted_to_ass(self):
        srt_path = self._write_file(
            "ep01.srt",
            "1\n00:00:01,000 --> 00:00:02,000\nhello\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nworld\n",
            encoding="utf-8",
        )
        ass_path = os.path.join(self.transcribe_dir, "ep01.ass")

        converted, reason = ass_slice.ensure_ass_from_srt(srt_path, ass_path)
        self.assertTrue(converted)
        self.assertEqual(reason, "converted")

        with open(ass_path, "r", encoding="utf-8") as f:
            ass_text = f.read()
        self.assertIn("[Events]", ass_text)
        self.assertIn("Dialogue:", ass_text)

    def test_priority_prefers_ass_over_smi_and_srt_over_smi(self):
        self._write_file("ep01.ass", "[Events]\n")
        self._write_file("ep01.smi", "<SAMI></SAMI>\n")
        self._write_file("ep02.srt", "1\n00:00:01,000 --> 00:00:02,000\nhello\n")
        self._write_file("ep02.smi", "<SAMI></SAMI>\n")
        self._write_file("ep03.smi", "<SAMI></SAMI>\n")

        jobs = ass_slice._build_subtitle_jobs(self.transcribe_dir)
        selected = {job["base"]: (job["selected_name"], job["selected_ext"]) for job in jobs}

        self.assertEqual(selected["ep01"], ("ep01.ass", ".ass"))
        self.assertEqual(selected["ep02"], ("ep02.srt", ".srt"))
        self.assertEqual(selected["ep03"], ("ep03.smi", ".smi"))

    def test_next_selection_reuses_generated_ass(self):
        smi_path = self._write_file(
            "ep10.smi",
            "<SAMI><BODY>\n"
            "<SYNC Start=1000><P Class=KRCC>line1\n"
            "<SYNC Start=2000><P Class=KRCC>line2\n"
            "</BODY></SAMI>\n",
            encoding="utf-16",
        )
        ass_path = os.path.join(self.transcribe_dir, "ep10.ass")
        converted, _ = ass_slice.ensure_ass_from_smi(smi_path, ass_path)
        self.assertTrue(converted)

        jobs = ass_slice._build_subtitle_jobs(self.transcribe_dir)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["base"], "ep10")
        self.assertEqual(jobs[0]["selected_ext"], ".ass")
        self.assertEqual(jobs[0]["selected_name"], "ep10.ass")


if __name__ == "__main__":
    unittest.main()
