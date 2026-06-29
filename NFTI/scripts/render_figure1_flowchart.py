"""Render Figure 1: horizontal study flow diagram."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

PRIMARY = "#1f3b57"
PRIMARY_FILL = "#eef3f8"
ACCENT = "#7a8a99"
ACCENT_FILL = "#f4f6f8"
MODEL_FILL = "#e3edf6"
EVAL_FILL = "#dceaf3"
TEXT = "#15212e"
ARROW = "#4a5b6b"

DPI = 230
GAP = 1.8
TITLE_BODY = 0.42


@dataclass
class Box:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.cx - self.w / 2

    @property
    def right(self) -> float:
        return self.cx + self.w / 2

    @property
    def top(self) -> float:
        return self.cy + self.h / 2

    @property
    def bottom(self) -> float:
        return self.cy - self.h / 2


def draw_box(ax, box: Box, title: str, body: str | None = None, *,
             border=PRIMARY, fill=PRIMARY_FILL,
             title_size=9.2, body_size=8.2, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (box.left, box.bottom), box.w, box.h,
        boxstyle="round,pad=0.22,rounding_size=0.45",
        linewidth=lw, edgecolor=border, facecolor=fill, zorder=2,
    ))
    if body:
        n_body = body.count("\n") + 1
        title_h = title_size * 0.12
        body_h = n_body * body_size * 0.12 * 1.3
        block_h = title_h + TITLE_BODY + body_h
        block_top = box.cy + block_h / 2
        ax.text(box.cx, block_top, title, ha="center", va="top",
                fontsize=title_size, fontweight="bold", color=PRIMARY, zorder=3)
        ax.text(box.cx, block_top - title_h - TITLE_BODY, body,
                ha="center", va="top", fontsize=body_size, color=TEXT,
                zorder=3, linespacing=1.28)
    else:
        ax.text(box.cx, box.cy, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=PRIMARY, zorder=3)


def h_arrow(ax, x0: float, x1: float, y: float, *, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y),
        arrowstyle="-|>", mutation_scale=12, lw=lw, color=ARROW,
        shrinkA=0, shrinkB=0, zorder=1,
    ))


def v_arrow(ax, x: float, y0: float, y1: float, *, lw=1.3):
    ax.add_patch(FancyArrowPatch(
        (x, y0), (x, y1),
        arrowstyle="-|>", mutation_scale=12, lw=lw, color=ARROW,
        shrinkA=0, shrinkB=0, zorder=1,
    ))


def h_line(ax, x0: float, x1: float, y: float, *, lw=1.3):
    ax.plot([x0, x1], [y, y], color=ARROW, lw=lw, zorder=1, solid_capstyle="round")


def v_line(ax, x: float, y0: float, y1: float, *, lw=1.3):
    ax.plot([x, x], [y0, y1], color=ARROW, lw=lw, zorder=1, solid_capstyle="round")


def main():
    fig = plt.figure(figsize=(13.5, 4.6), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 135)
    ax.set_ylim(0, 46)
    ax.axis("off")

    ax.text(
        2, 43.5,
        "Figure 1. Study flow: cohort definition, data splitting, and model development",
        ha="left", va="center", fontsize=11, fontweight="bold", color=PRIMARY,
    )

    mid_y = 20.5

    # ---- Left: source registry ----
    src = Box(cx=14, cy=mid_y, w=18, h=14)
    draw_box(ax, src,
             "2020 TQIP registry",
             "N = 1,124,671")

    # ---- Exclusion branch (below source) ----
    exc = Box(cx=14, cy=6.5, w=18, h=9)
    draw_box(ax, exc,
             "Excluded",
             "n = 465,420\nNon-EMS; IFT",
             border=ACCENT, fill=ACCENT_FILL, title_size=8.8, body_size=7.8)

    branch_x = src.cx
    v_arrow(ax, branch_x, src.bottom - 0.2, exc.top + GAP * 0.25)

    # ---- Analytic cohort ----
    coh = Box(cx=38, cy=mid_y, w=20, h=16)
    draw_box(ax, coh,
             "EMS analytic cohort",
             "N = 659,251\n74 predictors\n143 features")

    h_arrow(ax, src.right + GAP * 0.3, coh.left - GAP * 0.3, mid_y)

    # ---- Data split (three rows in one column) ----
    split_x = 62
    train = Box(cx=split_x, cy=28, w=17, h=7.5)
    val = Box(cx=split_x, cy=20.5, w=17, h=7.5)
    hold = Box(cx=split_x, cy=13, w=17, h=7.5)

    draw_box(ax, train, "Training", "n = 476,256", title_size=8.6, body_size=7.6)
    draw_box(ax, val, "Validation", "n = 84,046", title_size=8.6, body_size=7.6)
    draw_box(ax, hold, "Holdout", "n = 98,949", title_size=8.6, body_size=7.6, fill=EVAL_FILL)

    bus_x = coh.right + (split_x - 17 / 2 - coh.right) / 2
    h_line(ax, coh.right + GAP * 0.3, bus_x, mid_y)
    v_line(ax, bus_x, hold.bottom - 0.5, train.top + 0.5)
    h_arrow(ax, bus_x, train.left - GAP * 0.3, train.cy)
    h_arrow(ax, bus_x, val.left - GAP * 0.3, val.cy)
    h_arrow(ax, bus_x, hold.left - GAP * 0.3, hold.cy)

    # ---- Model development ----
    dev = Box(cx=86, cy=24, w=19, h=13)
    draw_box(ax, dev,
             "Model development",
             "XGBoost + LR\nGridSearchCV\nLocked threshold",
             fill=MODEL_FILL, title_size=8.8, body_size=7.6)

    merge_x = split_x + 17 / 2 + (dev.left - (split_x + 17 / 2)) / 2
    h_line(ax, train.right + GAP * 0.2, merge_x, train.cy)
    h_line(ax, val.right + GAP * 0.2, merge_x, val.cy)
    v_line(ax, merge_x, val.cy, train.cy)
    h_arrow(ax, merge_x, dev.left - GAP * 0.3, dev.cy)

    # ---- Holdout evaluation ----
    ev = Box(cx=110, cy=mid_y, w=19, h=16)
    draw_box(ax, ev,
             "Holdout evaluation",
             "Prevalence 18.7%\nThresholds 0.50 / 0.11\nAUROC, AUPRC, Brier",
             fill=EVAL_FILL, title_size=8.8, body_size=7.6)

    h_arrow(ax, hold.right + GAP * 0.3, ev.left - GAP * 0.3, hold.cy)
    h_arrow(ax, dev.right + GAP * 0.3, ev.left - GAP * 0.3, dev.cy)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "figure1_study_flow.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white", pad_inches=0.12)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
