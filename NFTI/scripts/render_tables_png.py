"""Render the manuscript markdown tables in results/tables as PNG images.

Each table is parsed from its .md file and drawn with matplotlib so the result
is a clean, high-resolution image that can be pasted directly into a Google Doc.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
OUT_DIR = TABLES_DIR / "png"

# Styling
HEADER_BG = "#1f3b57"
HEADER_FG = "#ffffff"
SECTION_BG = "#dce6f1"
ROW_BG_A = "#ffffff"
ROW_BG_B = "#f2f6fa"
GRID = "#b8c4d0"
TITLE_FG = "#1f3b57"
FOOTNOTE_FG = "#444444"

DPI = 220


def strip_bold(text: str) -> tuple[str, bool]:
    """Return (clean_text, is_bold). A cell is bold if fully wrapped in **."""
    t = text.strip()
    bold = bool(re.fullmatch(r"\*\*.*\*\*", t)) and t != ""
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    return t.strip(), bold


def parse_md(path: Path):
    """Parse a markdown file into a list of blocks.

    Blocks are dicts: {'type': 'title'|'subtitle'|'table'|'footnote', ...}
    """
    blocks = []
    rows = []

    def flush_table():
        nonlocal rows
        if rows:
            # rows[1] is the separator (---) row, drop it
            data = [rows[0]] + rows[2:]
            blocks.append({"type": "table", "rows": data})
            rows = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush_table()
            continue
        if line.startswith("###"):
            flush_table()
            blocks.append({"type": "title", "text": line.lstrip("#").strip()})
        elif line.startswith("|"):
            cells = [c for c in line.split("|")[1:-1]]
            rows.append(cells)
        elif line.startswith("**") and line.endswith("**"):
            flush_table()
            blocks.append({"type": "subtitle", "text": strip_bold(line)[0]})
        elif line.startswith("*") and line.endswith("*"):
            flush_table()
            blocks.append({"type": "footnote", "text": line.strip("*").strip()})
        else:
            flush_table()
            blocks.append({"type": "footnote", "text": line.strip()})
    flush_table()
    return blocks


def is_section_row(cells) -> bool:
    """A section header row: first cell bold, remaining cells empty."""
    first_clean, first_bold = strip_bold(cells[0])
    rest_empty = all(c.strip() == "" for c in cells[1:])
    return first_bold and rest_empty and first_clean != ""


def estimate_col_widths(header, body, font_size):
    """Estimate relative column widths from character counts."""
    ncol = len(header)
    maxlen = [len(strip_bold(header[i])[0]) for i in range(ncol)]
    for row in body:
        for i in range(ncol):
            maxlen[i] = max(maxlen[i], len(strip_bold(row[i])[0]))
    # add per-column padding and enforce a minimum so short headers don't collide
    maxlen = [max(m + 3, 6) for m in maxlen]
    total = sum(maxlen)
    return [m / total for m in maxlen], maxlen


def render(path: Path):
    blocks = parse_md(path)

    # Layout constants (in inches)
    char_w = 0.072  # approx width per character at base font
    base_font = 9.5
    title_font = 13
    subtitle_font = 11
    foot_font = 8
    row_h = 0.30
    pad = 0.18

    # Determine figure width from the widest table
    tables = [b for b in blocks if b["type"] == "table"]
    fig_w = 7.0
    for tb in tables:
        header = tb["rows"][0]
        body = tb["rows"][1:]
        _, maxlen = estimate_col_widths(header, body, base_font)
        w = sum(m * char_w for m in maxlen) + len(header) * 0.18 + 0.4
        fig_w = max(fig_w, w)
    fig_w = min(fig_w, 16.0)

    # Estimate total height
    total_h = pad
    for b in blocks:
        if b["type"] == "title":
            wrapped = textwrap.wrap(b["text"], width=int(fig_w / 0.11))
            total_h += 0.34 * len(wrapped) + 0.12
        elif b["type"] == "subtitle":
            total_h += 0.52
        elif b["type"] == "footnote":
            wrapped = textwrap.wrap(b["text"], width=int(fig_w / 0.072))
            total_h += 0.20 * len(wrapped) + 0.08
        elif b["type"] == "table":
            total_h += row_h * len(b["rows"]) + 0.18
    total_h += pad

    fig = plt.figure(figsize=(fig_w, total_h), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    y = total_h - pad

    for b in blocks:
        if b["type"] == "title":
            wrapped = textwrap.wrap(b["text"], width=int(fig_w / 0.11))
            for ln in wrapped:
                y -= 0.34
                ax.text(0.012, y, ln, fontsize=title_font, fontweight="bold",
                        color=TITLE_FG, va="center", ha="left")
            y -= 0.12
        elif b["type"] == "subtitle":
            y -= 0.34
            ax.text(0.012, y, b["text"], fontsize=subtitle_font, fontweight="bold",
                    color=TITLE_FG, va="center", ha="left")
            y -= 0.18
        elif b["type"] == "footnote":
            wrapped = textwrap.wrap(b["text"], width=int(fig_w / 0.072))
            for ln in wrapped:
                y -= 0.20
                ax.text(0.012, y, ln, fontsize=foot_font, style="italic",
                        color=FOOTNOTE_FG, va="center", ha="left")
            y -= 0.08
        elif b["type"] == "table":
            header = b["rows"][0]
            body = b["rows"][1:]
            ncol = len(header)
            fracs, _ = estimate_col_widths(header, body, base_font)
            # convert fracs to x positions within [0.01, 0.99]
            left, right = 0.008, 0.992
            span = right - left
            xs = [left]
            for f in fracs:
                xs.append(xs[-1] + f * span)
            # cell font size shrinks if many columns
            fs = base_font if ncol <= 9 else (8.0 if ncol <= 12 else 7.0)

            # header row
            y_top = y
            y -= row_h
            ax.add_patch(plt.Rectangle((left, y), span, row_h, facecolor=HEADER_BG,
                                       edgecolor=GRID, lw=0.6, zorder=1))
            for i in range(ncol):
                clean, _ = strip_bold(header[i])
                cx = xs[i] + 0.004
                ha = "left" if i == 0 else "center"
                if ha == "center":
                    cx = (xs[i] + xs[i + 1]) / 2
                ax.text(cx, y + row_h / 2, clean, fontsize=fs, color=HEADER_FG,
                        fontweight="bold", va="center", ha=ha, zorder=3)
            # body rows
            shade = 0
            for row in body:
                section = is_section_row(row)
                y -= row_h
                if section:
                    bg = SECTION_BG
                else:
                    bg = ROW_BG_A if shade % 2 == 0 else ROW_BG_B
                    shade += 1
                ax.add_patch(plt.Rectangle((left, y), span, row_h, facecolor=bg,
                                           edgecolor=GRID, lw=0.5, zorder=1))
                for i in range(ncol):
                    clean, bold = strip_bold(row[i])
                    if clean == "":
                        continue
                    ha = "left" if i == 0 else "center"
                    if ha == "left":
                        cx = xs[i] + 0.004
                    else:
                        cx = (xs[i] + xs[i + 1]) / 2
                    weight = "bold" if (bold or section) else "normal"
                    ax.text(cx, y + row_h / 2, clean, fontsize=fs, color="#1a1a1a",
                            fontweight=weight, va="center", ha=ha, zorder=3)
            # vertical grid lines
            for xv in xs:
                ax.plot([xv, xv], [y, y_top], color=GRID, lw=0.5, zorder=2)
            y -= 0.18

    out = OUT_DIR / (path.stem + ".png")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    return out


def main():
    md_files = sorted(TABLES_DIR.glob("*.md"))
    for md in md_files:
        out = render(md)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
