"""Shared manuscript plotting style for NFTI figures.

Centralizes the neutral color palette and a consistent light-gray gridline
style so every generated manuscript figure shares one visual language.
"""
from __future__ import annotations

# Neutral manuscript color palette (dark navy / red / gold).
NAVY = "#13274F"
RED = "#CE1141"
GOLD = "#EAAA00"

# Light-gray gridline appearance applied across manuscript figures.
GRID_COLOR = "#B0B0B0"
GRID_ALPHA = 0.35
GRID_LINEWIDTH = 1.0


def apply_manuscript_grid(ax) -> None:
    """Apply the shared light-gray manuscript gridline style to ``ax``.

    Draws major gridlines on both axes behind the plotted data. Intended for
    line/bar/scatter figures where gridlines improve readability; avoid on
    dense heatmaps or library-styled plots (e.g. SHAP).
    """
    ax.grid(
        True,
        which="major",
        axis="both",
        color=GRID_COLOR,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
    )
    ax.set_axisbelow(True)
