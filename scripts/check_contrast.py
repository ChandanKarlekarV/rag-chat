"""Fail if any required token pair misses WCAG AA contrast, in either theme.

Reads `app/static/css/tokens.css` (the `:root` block with colors = light, `[data-theme="dark"]`
block = dark; dark inherits any token it does not override).

Usage: python scripts/check_contrast.py
"""

import re
import sys
from pathlib import Path

TOKENS = Path(__file__).resolve().parents[1] / "app" / "static" / "css" / "tokens.css"

# (foreground, background, minimum ratio). Text pairs need 4.5:1; borders/UI need 3:1.
PAIRS: list[tuple[str, str, float]] = [
    ("--color-text", "--color-bg", 4.5),
    ("--color-text", "--color-surface", 4.5),
    ("--color-text-muted", "--color-bg", 4.5),
    ("--color-text-muted", "--color-surface", 4.5),
    ("--color-text-muted", "--color-surface-muted", 4.5),
    ("--color-text-inverse", "--color-primary", 4.5),
    ("--color-primary", "--color-surface", 4.5),
    ("--color-primary", "--color-primary-subtle", 4.5),
    ("--color-accent", "--color-accent-subtle", 4.5),
    ("--color-text", "--color-accent-subtle", 4.5),
    ("--color-danger", "--color-danger-subtle", 4.5),
    ("--color-success", "--color-success-subtle", 4.5),
    ("--color-focus", "--color-surface", 3.0),
    ("--color-border-input", "--color-surface", 3.0),  # WCAG 1.4.11: input outlines
    ("--color-text", "--color-sidebar", 4.5),
    ("--color-text-muted", "--color-sidebar", 4.5),
    # study output: white paper and topic colors (identical in both themes)
    ("--color-paper-ink", "--color-paper", 4.5),
    ("--color-paper-muted", "--color-paper", 4.5),
    ("--color-paper-muted", "--color-paper-chip", 4.5),
    *[(f"--color-topic-{i}-ink", f"--color-topic-{i}-bg", 4.5) for i in range(1, 9)],
    *[("--color-paper-ink", f"--color-topic-{i}-bg", 4.5) for i in range(1, 9)],
    *[("--color-paper-muted", f"--color-topic-{i}-bg", 4.5) for i in range(1, 9)],
    *[(f"--color-topic-{i}-ink", "--color-paper", 4.5) for i in range(1, 9)],
    *[("--color-paper", f"--color-topic-{i}-ink", 4.5) for i in range(1, 9)],  # numbers on ink circles
    *[(f"--color-topic-{i}-accent", "--color-paper", 3.0) for i in range(1, 9)],  # lines and borders
]


def parse_blocks(css: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (light, dark) token maps of hex colors."""
    blocks = re.findall(r"([^{}]+)\{([^{}]*)\}", re.sub(r"/\*.*?\*/", "", css, flags=re.S))
    light: dict[str, str] = {}
    dark: dict[str, str] = {}
    for selector, body in blocks:
        values = dict(re.findall(r"(--color-[\w-]+)\s*:\s*(#[0-9a-fA-F]{6})\s*;", body))
        if '[data-theme="dark"]' in selector:
            dark |= values
        elif selector.strip() == ":root":
            light |= values
    return light, light | dark


def luminance(hex_color: str) -> float:
    """WCAG relative luminance."""
    channels = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two hex colors."""
    a, b = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def main() -> int:
    light, dark = parse_blocks(TOKENS.read_text(encoding="utf-8"))
    failures = 0
    print("| Theme | Foreground | Background | Ratio | Needs | Result |\n|---|---|---|---|---|---|")
    for theme, tokens in (("light", light), ("dark", dark)):
        for fg, bg, need in PAIRS:
            if fg not in tokens or bg not in tokens:
                print(f"| {theme} | {fg} | {bg} | — | {need} | MISSING |")
                failures += 1
                continue
            r = ratio(tokens[fg], tokens[bg])
            ok = r >= need
            failures += not ok
            print(f"| {theme} | {fg} | {bg} | {r:.2f} | {need} | {'pass' if ok else 'FAIL'} |")
    print(f"\n{'All pairs pass.' if not failures else f'{failures} pair(s) fail.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
