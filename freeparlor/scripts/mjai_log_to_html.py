#!/usr/bin/env python3
"""mjai json.gz / jsonl ログを log-viewer 用 HTML に変換する。

log-viewer/index.example.html の allActions バッククォート埋め込み方式に従う。
イベント行にバッククォートまたは ${ が含まれる場合は loud FAIL（エスケープ禁止）。
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / "log-viewer" / "index.example.html"
FILES_SRC = REPO_ROOT / "log-viewer" / "files"
DEFAULT_OUT_DIR = Path("/home/gamba/mahjong/runs/viewer_out")

ALLACTIONS_RE = re.compile(
    r"(allActions = `\n)"  # group 1: opening marker
    r".*?"
    r"(\n    `\.trim\(\)\.split\('\\n'\)\.map\(s => JSON\.parse\(s\)\))",
    re.DOTALL,
)

FORBIDDEN_IN_LINE = ("`", "${")


@dataclass(frozen=True)
class LogEntry:
    source: Path
    html_name: str
    player_names: str
    n_lines: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert mjai json.gz/jsonl logs to log-viewer HTML pages.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="mjai log file(s) and/or directory(ies) containing *.json.gz / *.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"output directory (default: {DEFAULT_OUT_DIR})",
    )
    return parser.parse_args()


def collect_log_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")
        if path.is_file():
            if path.suffix == ".jsonl" and path.name.endswith(".script.jsonl"):
                continue
            if path.suffix not in {".gz", ".jsonl"}:
                raise ValueError(f"unsupported log file type: {path}")
            if path not in seen:
                seen.add(path)
                paths.append(path)
            continue
        if not path.is_dir():
            raise ValueError(f"input is neither file nor directory: {path}")
        for candidate in sorted(path.glob("*.json.gz")):
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
        for candidate in sorted(path.glob("*.jsonl")):
            if candidate.name.endswith(".script.jsonl"):
                continue
            if candidate not in seen:
                seen.add(candidate)
                paths.append(candidate)
    if not paths:
        raise ValueError("no mjai logs found in inputs")
    return paths


def read_log_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            raw_lines = fh.readlines()
    else:
        raw_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        if raw_lines and not raw_lines[-1].endswith("\n"):
            raw_lines[-1] += "\n"

    lines: list[str] = []
    for lineno, raw in enumerate(raw_lines, start=1):
        line = raw.rstrip("\n")
        if not line:
            continue
        for token in FORBIDDEN_IN_LINE:
            if token in line:
                raise ValueError(
                    f"{path}:{lineno}: event line contains forbidden {token!r} "
                    f"(cannot embed safely in allActions backtick string)"
                )
        json.loads(line)
        lines.append(line)
    if not lines:
        raise ValueError(f"{path}: no non-empty mjai event lines")
    return lines


def extract_player_names(lines: list[str]) -> str:
    for line in lines:
        event = json.loads(line)
        if event.get("type") == "start_game":
            names = event.get("names")
            if not isinstance(names, list) or not names:
                raise ValueError("start_game event missing names")
            return ", ".join(str(name) for name in names)
    raise ValueError("start_game event not found")


def load_template() -> str:
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"template not found: {TEMPLATE_PATH}")
    return TEMPLATE_PATH.read_text(encoding="utf-8")


SPLIT_NEWLINE_MARKER = ".split('\\n')"


def render_html(template: str, lines: list[str], title: str) -> str:
    body = "\n".join(lines)
    match = ALLACTIONS_RE.search(template)
    if not match:
        raise RuntimeError(
            f"template marker not found in {TEMPLATE_PATH} "
            "(expected allActions backtick block)"
        )
    html = (
        template[: match.start()]
        + match.group(1)
        + body
        + match.group(2)
        + template[match.end() :]
    )
    return html.replace("<title>Mahjong Archive Player</title>", f"<title>{title}</title>")


def ensure_viewer_assets(out_dir: Path) -> None:
    dst = out_dir / "files"
    if dst.exists():
        return
    if not FILES_SRC.is_dir():
        raise FileNotFoundError(f"log-viewer assets not found: {FILES_SRC}")
    shutil.copytree(FILES_SRC, dst)


def write_index_html(out_dir: Path, entries: list[LogEntry]) -> None:
    rows = []
    for entry in sorted(entries, key=lambda e: e.html_name):
        rows.append(
            "<tr>"
            f"<td>{entry.html_name}</td>"
            f"<td>{entry.player_names}</td>"
            f"<td><a href=\"{entry.html_name}\">{entry.html_name}</a></td>"
            f"<td>{entry.n_lines}</td>"
            "</tr>"
        )
    html = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset=\"utf-8\"><title>Mjai Log Index</title></head>\n"
        "<body>\n"
        "<h1>Mjai Log Viewer Index</h1>\n"
        "<table border=\"1\">\n"
        "<tr><th>Log</th><th>Players</th><th>Link</th><th>Events</th></tr>\n"
        + "\n".join(rows)
        + "\n</table>\n</body></html>\n"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def html_name_for_log(path: Path, reserved: set[str]) -> str:
    base = f"{path.name}.html" if path.suffix == ".gz" else f"{path.stem}.html"
    candidate = base
    counter = 2
    while candidate in reserved:
        stem = base[:-5] if base.endswith(".html") else base
        candidate = f"{stem}__{counter}.html"
        counter += 1
    if candidate != base:
        print(
            f"NOTE: duplicate log basename {base!r}; "
            f"using disambiguated name {candidate} for {path}",
            file=sys.stderr,
        )
    reserved.add(candidate)
    return candidate


def convert_log(path: Path, template: str, out_dir: Path, reserved: set[str]) -> LogEntry:
    lines = read_log_lines(path)
    html_name = html_name_for_log(path, reserved)
    title = path.stem
    player_names = extract_player_names(lines)
    html = render_html(template, lines, title=title)
    out_path = out_dir / html_name
    out_path.write_text(html, encoding="utf-8")
    return LogEntry(
        source=path,
        html_name=html_name,
        player_names=player_names,
        n_lines=len(lines),
    )


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_paths = collect_log_paths(args.inputs)
    template = load_template()
    ensure_viewer_assets(out_dir)

    entries: list[LogEntry] = []
    reserved_names: set[str] = set()
    for path in log_paths:
        entry = convert_log(path, template, out_dir, reserved_names)
        html_path = out_dir / entry.html_name
        html_text = html_path.read_text(encoding="utf-8")
        if SPLIT_NEWLINE_MARKER not in html_text:
            raise AssertionError(
                f"{html_path}: generated HTML missing literal {SPLIT_NEWLINE_MARKER!r} "
                "(re.sub escape regression: .split backslash-n was converted to real newline)"
            )
        entries.append(entry)
        print(
            f"WROTE {html_path} "
            f"(source={path}, events={entry.n_lines}, players={entry.player_names})"
        )
        print(f"CHECK {html_path.name}: split-newline marker present")

    write_index_html(out_dir, entries)
    print(f"WROTE {out_dir / 'index.html'} ({len(entries)} logs)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
