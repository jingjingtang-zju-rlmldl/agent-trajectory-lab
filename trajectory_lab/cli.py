import argparse
import json
from pathlib import Path
import sys

from .audit import analyze, load_jsonl
from .report import render
from . import __version__


def main():
    parser = argparse.ArgumentParser(description="Audit agent RL JSONL trajectories on CPU")
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("audit-report"))
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strict", action="store_true", help="exit 1 on audit errors; warnings remain nonfatal")
    args = parser.parse_args()
    try:
        rows, digest = load_jsonl(args.input)
        data = analyze(rows, args.k, args.seed)
        data.update(input_sha256=digest, tool_version=__version__)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.out / "report.html").write_text(render(data), encoding="utf-8")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{data['trajectories']} trajectories | {data['errors']} errors | {data['warnings']} warnings")
    print(f"Report: {args.out / 'report.html'}")
    return 1 if args.strict and data["errors"] else 0
