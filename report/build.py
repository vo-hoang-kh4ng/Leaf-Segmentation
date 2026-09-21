r"""Build the report: regenerate assets from results/, then run XeLaTeX twice.

Two passes are required -- the first resolves \label targets, the second fills in the
\ref cross-references that the first pass reported as undefined.

Usage:
    python report/build.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")


def check_source() -> list[str]:
    """Catch LaTeX commands whose backslash was eaten by a Python string escape.

    Editing report.tex through non-raw Python strings turns \\ref into CR+"ef", \\textbf
    into TAB+"extbf", \\bestacc into BACKSPACE+"estacc". None of these is a LaTeX error --
    the PDF builds cleanly and simply prints "ef{sec:...}" as body text -- so the log
    cannot catch it. It happened three times while writing this report.
    """
    import re

    text = (HERE / "report.tex").read_text(encoding="utf-8")
    problems = [f"control character {m.group()!r} at offset {m.start()}"
                for m in re.finditer(r"[\x07\x08\x0b\x0c\t]", text)]
    for n, line in enumerate(text.splitlines(), 1):
        if re.match(r"\s*(ef\{|egin\{|extbf|odo\{|mph\{)", line):
            problems.append(f"line {n} starts with a headless command: {line.strip()[:40]}")
    return problems


def main() -> int:
    (HERE / "build").mkdir(exist_ok=True)

    problems = check_source()
    if problems:
        print("report.tex has eaten escapes (fix before building):", *problems, sep="\n  ")
        return 1

    result = run([sys.executable, "make_assets.py"])
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode:
        return result.returncode

    for i in (1, 2):
        result = run([
            "xelatex", "-interaction=nonstopmode",
            "-output-directory=build", "report.tex",
        ])
        (HERE / "build" / f"pass{i}.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode:
            errors = [l for l in result.stdout.splitlines() if l.startswith("! ")]
            print(f"xelatex pass {i} failed:", *errors, sep="\n  ")
            print(f"full log: report/build/pass{i}.log")
            return result.returncode

    # Undefined references survive only if the second pass still complains; overfull
    # boxes mean text is spilling into the margin. Both are worth surfacing.
    log = (HERE / "build" / "pass2.log").read_text(encoding="utf-8")
    for warning in ("undefined", "Overfull"):
        hits = [l for l in log.splitlines() if warning in l]
        if hits:
            print(f"{len(hits)} '{warning}' warning(s):", *hits[:5], sep="\n  ")

    pages = log.split("Output written on")[-1].split("(")[-1].split(" pages")[0]
    print(f"report/build/report.pdf -- {pages} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
