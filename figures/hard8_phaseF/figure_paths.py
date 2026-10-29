"""Input and output locations shared by the 8 x 8 and mechanism figure builders.

redraw_all_figures.py sets NRT_FIGURE_AGGREGATE_DIR and NRT_FIGURE_OUTPUT_DIR before it imports the
builders.  Without these variables the repository copy of the figure source data is read
(data/aggregate_full) and the figures are written to outputs/figures/_builder_outputs.
"""
from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AGGREGATE = Path(os.environ.get("NRT_FIGURE_AGGREGATE_DIR", REPOSITORY_ROOT / "data" / "aggregate_full")).resolve()
SOURCE_ROOT = AGGREGATE / "figure_source_data"
PHASEF_FIGURE_DATA = SOURCE_ROOT / "FigureA5"
PHASEF_TABLES = AGGREGATE / "hard8x8_phaseF_followup"
OUT = Path(os.environ.get("NRT_FIGURE_OUTPUT_DIR", REPOSITORY_ROOT / "outputs" / "figures" / "_builder_outputs")).resolve()
