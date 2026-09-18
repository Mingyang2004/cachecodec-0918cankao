#!/usr/bin/env python3
"""Pack a source tree into pasteable text parts and restore it."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path, PurePosixPath


VERSION = "CACHECODEC_TEXT_BUNDLE v1"
BEGIN = "===== BEGIN FILE: "
END = "===== END FILE ====="
TEXT_EXTENSIONS = {
    ".cfg",
    ".ini",
    ".ipynb",
    ".jinja",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
CODE_ONLY_EXTENSIONS = {".jinja", ".py", ".sh", ".toml", ".yaml", ".yml"}
TEXT_NAMES = {".gitignore", "Dockerfile", "Makefile"}
SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
    "dist",
    "output",
    "outputs",
    "data",
    "dataset",
    "datasets",
    "checkpoints",
    "models",
    "cachecodec-output",
}
SKIP_RELATIVE_PATHS = {("local", "eval")}
HEADER_RE = re.compile(r"^" + re.escape(VERSION) + r"\nPART: (\d+)/(\d+)\n\n$")
BLOCK_RE = re.compile(
    r"^===== BEGIN FILE: (.+?) =====\n(.*?)^===== END FILE =====\n",
    re.MULTILINE | re.DOTALL,
)


def is_text_source(path: Path, code_only: bool = False) -> bool:
    extensions = CODE_ONLY_EXTENSIONS if code_only else TEXT_EXTENSIONS
    return path.name in TEXT_NAMES or path.suffix.lower() in extensions


def is_skipped_directory(root: Path, path: Path) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(
        relative_parts[: len(prefix)] == prefix for prefix in SKIP_RELATIVE_PATHS
    )


def iter_source_files(root: Path, excluded_paths=(), code_only: bool = False):
    excluded = [path.resolve() for path in excluded_paths]
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        directory_names[:] = [
            name
            for name in directory_names
            if name not in SKIP_PARTS
            and not is_skipped_directory(root, directory_path / name)
        ]
        directory_names[:] = [
            name
            for name in directory_names
            if not any((directory_path / name).resolve() == item for item in excluded)
        ]
        for name in sorted(file_names):
            path = directory_path / name
            if path.is_symlink() or not is_text_source(path, code_only):
                continue
            resolved = path.resolve()
            if any(resolved == item or item in resolved.parents for item in excluded):
                continue
            yield path


def pack(
    source: Path,
    destination: Path,
    max_lines: int,
    single_file: bool = False,
    code_only: bool = False,
) -> int:
    if max_lines < 1:
        raise ValueError("--max-lines must be at least 1")
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    if single_file:
        destination.parent.mkdir(parents=True, exist_ok=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for old_part in destination.glob("part_*.txt"):
            old_part.unlink()

    chunks = []
    output_path = destination.resolve()
    excluded_paths = [output_path]
    for path in iter_source_files(source, excluded_paths, code_only):
        relative = path.relative_to(source).as_posix()
        content = path.read_text(encoding="utf-8", errors="strict")
        lines = content.splitlines(keepends=True) or [""]
        for offset in range(0, len(lines), max_lines):
            chunks.append((relative, "".join(lines[offset : offset + max_lines])))

    total = len(chunks)
    if not total:
        raise ValueError("no supported text files found")
    if single_file:
        output_paths = [(destination, 1, 1)]
    else:
        output_paths = [
            (destination / f"part_{number:05d}_of_{total:05d}.txt", number, total)
            for number in range(1, total + 1)
        ]
    for part, number, part_total in output_paths:
        with part.open("w", encoding="utf-8", newline="") as output:
            output.write(f"{VERSION}\nPART: {number}/{part_total}\n\n")
            selected = chunks if single_file else [chunks[number - 1]]
            for relative, content in selected:
                output.write(f"{BEGIN}{relative} =====\n")
                output.write(content)
                if content and not content.endswith("\n"):
                    output.write("\n")
                output.write(f"{END}\n")
    return total


def safe_relative_path(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise ValueError(f"unsafe file path in bundle: {value!r}")
    return Path(*candidate.parts)


def format_size(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{size} B"
        value /= 1024
    raise AssertionError("unreachable")


def scan(source: Path, report: Path, top: int, code_only: bool = False) -> int:
    if not source.is_dir():
        raise ValueError(f"source directory does not exist: {source}")
    if top < 1:
        raise ValueError("--top must be at least 1")
    entries = []
    for directory, directory_names, file_names in os.walk(
        source, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        directory_names[:] = [
            name
            for name in directory_names
            if name not in SKIP_PARTS
            and not is_skipped_directory(source, directory_path / name)
        ]
        for name in sorted(file_names):
            path = directory_path / name
            if path.is_symlink():
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            kind = "TEXT" if is_text_source(path, code_only) else "OTHER"
            entries.append((size, kind, path.relative_to(source).as_posix()))
    entries.sort(reverse=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as output:
        output.write("CACHECODEC FILE SIZE SCAN v1\n")
        output.write(f"ROOT: {source.resolve()}\n")
        output.write("Files are only stat'ed; contents are not read.\n\n")
        output.write("SIZE\tTYPE\tPATH\n")
        for size, kind, relative in entries[:top]:
            output.write(f"{format_size(size)}\t{kind}\t{relative}\n")
        output.write(f"\nTOTAL FILES: {len(entries)}\n")
        output.write(f"SHOWING TOP: {min(top, len(entries))}\n")
    return len(entries)


def unpack(parts_path: Path, destination: Path) -> int:
    parts = sorted(parts_path.glob("part_*.txt")) if parts_path.is_dir() else [parts_path]
    if not parts:
        raise ValueError(f"no part_*.txt files found in {parts_path}")
    files = {}
    expected_total = None
    seen_parts = set()
    for part in parts:
        text = part.read_text(encoding="utf-8")
        header_end = text.find("\n\n")
        if header_end < 0:
            raise ValueError(f"invalid bundle header: {part}")
        header = text[: header_end + 2]
        match = HEADER_RE.match(header)
        if not match:
            raise ValueError(f"invalid bundle header: {part}")
        number, total = map(int, match.groups())
        if expected_total is None:
            expected_total = total
        if total != expected_total or number in seen_parts:
            raise ValueError("bundle parts have inconsistent or duplicate numbers")
        seen_parts.add(number)
        body = text[header_end + 2 :]
        blocks = list(BLOCK_RE.finditer(body))
        if not blocks or "".join(block.group(0) for block in blocks) != body:
            raise ValueError(f"invalid file block: {part}")
        for block_index, block in enumerate(blocks):
            relative = safe_relative_path(block.group(1))
            files.setdefault(relative, []).append((number, block_index, block.group(2)))

    if expected_total != len(seen_parts) or seen_parts != set(range(1, expected_total + 1)):
        raise ValueError(f"missing bundle parts: expected 1..{expected_total}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative, chunks in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(chunks, key=lambda chunk: (chunk[0], chunk[1]))
        target.write_text("".join(content for _, _, content in ordered), encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack_parser = subparsers.add_parser("pack", help="create pasteable text parts")
    pack_parser.add_argument("source", type=Path)
    pack_parser.add_argument("destination", type=Path)
    pack_parser.add_argument("--max-lines", type=int, default=500)
    pack_parser.add_argument(
        "--single-file",
        action="store_true",
        help="write one large text file instead of a part directory",
    )
    pack_parser.add_argument(
        "--code-only",
        action="store_true",
        help="include only source code and YAML/TOML configuration",
    )
    unpack_parser = subparsers.add_parser("unpack", help="restore a source tree")
    unpack_parser.add_argument("parts", type=Path)
    unpack_parser.add_argument("destination", type=Path)
    scan_parser = subparsers.add_parser("scan", help="report file sizes without reading contents")
    scan_parser.add_argument("source", type=Path)
    scan_parser.add_argument("report", type=Path)
    scan_parser.add_argument("--top", type=int, default=200)
    scan_parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "pack":
            count = pack(
                args.source,
                args.destination,
                args.max_lines,
                args.single_file,
                args.code_only,
            )
        elif args.command == "unpack":
            count = unpack(args.parts, args.destination)
        else:
            count = scan(args.source, args.report, args.top, args.code_only)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    print(f"{args.command}: {count} {'parts' if args.command == 'pack' else 'files'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
