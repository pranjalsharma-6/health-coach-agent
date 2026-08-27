"""Contrast and vividness checks for the Kaya palette.

Pastels are where accessibility quietly fails: a colour soft enough to look
calm as a surface is usually too light to carry text. Every pair below is one
that actually appears in the UI, checked against WCAG 2.1 contrast, plus an
OKLCH chroma floor so an accent doesn't drift far enough toward grey to read as
"disabled".

Run: python scripts/check-palette.py
"""

import math
import re
import sys
from pathlib import Path

CSS = Path(__file__).parent.parent / "app" / "globals.css"

AA_TEXT = 4.5          # body text
AA_LARGE = 3.0         # >=18.66px bold or >=24px
UI_COMPONENT = 3.0     # borders, focus rings, chart marks
CHROMA_FLOOR = 0.035   # below this an accent reads grey rather than coloured


def parse_tokens() -> dict:
    text = CSS.read_text(encoding="utf-8")
    return dict(re.findall(r"(--color-[a-z0-9-]+):\s*(#[0-9a-fA-F]{6})", text))


def srgb(hex_colour: str) -> tuple:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))


def linearize(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_colour: str) -> float:
    r, g, b = (linearize(c) for c in srgb(hex_colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: str, b: str) -> float:
    la, lb = relative_luminance(a), relative_luminance(b)
    lighter, darker = max(la, lb), min(la, lb)
    return (lighter + 0.05) / (darker + 0.05)


def oklch_chroma(hex_colour: str) -> float:
    r, g, b = (linearize(c) for c in srgb(hex_colour))
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l_, m_, s_ = (math.copysign(abs(v) ** (1 / 3), v) for v in (l, m, s))
    a_ = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return math.hypot(a_, b_)


def main() -> int:
    t = parse_tokens()
    failures = []

    def check(label, fg, bg, threshold):
        if fg not in t or bg not in t:
            missing = fg if fg not in t else bg
            failures.append(f"{label}: token {missing} not found")
            return
        ratio = contrast(t[fg], t[bg])
        ok = ratio >= threshold
        status = "ok  " if ok else "FAIL"
        print(f"  {status} {label:44} {ratio:5.2f}:1  (needs {threshold})")
        if not ok:
            failures.append(f"{label}: {ratio:.2f}:1 < {threshold}")

    def check_chroma(label, token):
        c = oklch_chroma(t[token])
        ok = c >= CHROMA_FLOOR
        print(f"  {'ok  ' if ok else 'FAIL'} {label:44} chroma {c:.3f}")
        if not ok:
            failures.append(f"{label}: chroma {c:.3f} < {CHROMA_FLOOR}")

    print("LIGHT THEME — text on card (#ffffff) and page")
    check("body text on card", "--color-ink-light", "--color-card-light", AA_TEXT)
    check("secondary text on card", "--color-ink-soft-light", "--color-card-light", AA_TEXT)
    check("muted text on card", "--color-ink-muted-light", "--color-card-light", AA_TEXT)
    check("body text on page", "--color-ink-light", "--color-page-light", AA_TEXT)
    check("link/accent text on card", "--color-brand-700", "--color-card-light", AA_TEXT)
    check("error text on card", "--color-clay-700", "--color-card-light", AA_TEXT)
    check("button label on brand fill", "--color-card-light", "--color-brand-600", AA_TEXT)
    check("border on card", "--color-sand-300", "--color-card-light", 1.2)

    print()
    print("DARK THEME")
    check("body text on card", "--color-ink-dark", "--color-card-dark", AA_TEXT)
    check("secondary text on card", "--color-ink-soft-dark", "--color-card-dark", AA_TEXT)
    check("muted text on card", "--color-ink-muted-dark", "--color-card-dark", AA_TEXT)
    check("accent text on card", "--color-brand-300", "--color-card-dark", AA_TEXT)
    check("error text on card", "--color-clay-300", "--color-card-dark", AA_TEXT)

    print()
    print("CHART / UI MARKS — must read against both themes")
    check("chart series on light card", "--color-chart-series", "--color-card-light", UI_COMPONENT)
    check("chart series on dark card", "--color-chart-series", "--color-card-dark", UI_COMPONENT)

    print()
    print("VIVIDNESS — an accent must not read as grey")
    for token in ("--color-brand-500", "--color-brand-600", "--color-clay-500",
                  "--color-spice-400", "--color-chart-series"):
        check_chroma(token.replace("--color-", ""), token)

    print()
    if failures:
        print(f"FAILED — {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS — every pair meets its threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
