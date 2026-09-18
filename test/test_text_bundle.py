import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "text_bundle.py"


def run_bundle(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        check=True,
        capture_output=True,
        text=True,
    )


def test_pack_and_unpack_round_trip_split_files(tmp_path):
    source = tmp_path / "source"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "main.py").write_text("line1\nline2\nline3\n", encoding="utf-8")
    (source / "README.md").write_text("hello\n", encoding="utf-8")
    (source / "ignored.log").write_text("ignore\n", encoding="utf-8")
    bundle = tmp_path / "bundle"
    restored = tmp_path / "restored"

    run_bundle("pack", source, bundle, "--max-lines", 2)
    parts = sorted(bundle.glob("part_*.txt"))
    assert len(parts) == 3

    run_bundle("unpack", bundle, restored)
    assert (restored / "pkg" / "main.py").read_text(encoding="utf-8") == "line1\nline2\nline3\n"
    assert (restored / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert not (restored / "ignored.log").exists()


def test_pack_and_unpack_single_file(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("z\na\n", encoding="utf-8")
    bundle = tmp_path / "bundle.txt"
    restored = tmp_path / "restored"

    run_bundle("pack", source, bundle, "--single-file", "--max-lines", 1)
    run_bundle("unpack", bundle, restored)
    assert (restored / "main.py").read_text(encoding="utf-8") == "z\na\n"


def test_pack_ignores_output_file_inside_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("print('ok')\n", encoding="utf-8")
    bundle = source / "cachecodec_all.txt"

    run_bundle("pack", source, bundle, "--single-file")
    first_size = bundle.stat().st_size
    run_bundle("pack", source, bundle, "--single-file")
    assert bundle.stat().st_size == first_size


def test_scan_reports_large_files_without_reading_content(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "small.py").write_text("x\n", encoding="utf-8")
    (source / "weights.bin").write_bytes(b"0" * 100)
    report = tmp_path / "scan.txt"

    run_bundle("scan", source, report)
    text = report.read_text(encoding="utf-8")
    assert "weights.bin" in text
    assert "100 B" in text
    assert "small.py" in text


def test_scan_skips_project_result_directories(tmp_path):
    source = tmp_path / "source"
    (source / "cachecodec-output").mkdir(parents=True)
    (source / "local" / "eval").mkdir(parents=True)
    (source / "code.py").write_text("x\n", encoding="utf-8")
    (source / "cachecodec-output" / "result.json").write_text("x", encoding="utf-8")
    (source / "local" / "eval" / "result.json").write_text("x", encoding="utf-8")
    report = tmp_path / "scan.txt"

    run_bundle("scan", source, report)
    text = report.read_text(encoding="utf-8")
    assert "code.py" in text
    assert "result.json" not in text


def test_code_only_excludes_json_text_and_markdown(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "main.py").write_text("x\n", encoding="utf-8")
    (source / "config.json").write_text("{}\n", encoding="utf-8")
    (source / "notes.md").write_text("notes\n", encoding="utf-8")
    bundle = tmp_path / "bundle.txt"

    run_bundle("pack", source, bundle, "--single-file", "--code-only")
    text = bundle.read_text(encoding="utf-8")
    assert "main.py" in text
    assert "config.json" not in text
    assert "notes.md" not in text
