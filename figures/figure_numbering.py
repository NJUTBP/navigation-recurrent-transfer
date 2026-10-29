"""Figure numbers of the article and the builders and source data behind them.

The figure builders and the folders of ``data/aggregate_full/figure_source_data`` keep the
identifiers under which they were created (for example ``FigureA3`` or
``SupplementaryFigureS8``).  This table is the single place that relates those identifiers to
the numbering of the article; ``redraw_all_figures.py`` and ``export_source_data_final.py``
both read it, and ``docs/FIGURE_DATA_MAP.md`` describes it.
"""
from __future__ import annotations

from typing import NamedTuple


class PaperFigure(NamedTuple):
    number: str                   # name in the article and in the redraw output folder
    builder: str                  # module below figures/ that draws the figure
    function: str                 # entry point: build(ctx) or a module-level function
    builder_output: str           # file stem written by the builder
    source_data: tuple[str, ...]  # source data exported for the figure, relative to data/aggregate_full
    content: str


# Build order: the 4 x 4, pretraining and topology builders first, then the 8 x 8 and
# mechanism builders (the same order as in the runs that produced the article figures).
PAPER_FIGURES: tuple[PaperFigure, ...] = (
    PaperFigure("Figure2", "paper_figures.make_figure2_navigation", "build", "Figure2",
                ("figure_source_data/Figure2",), "Navigation pretraining (builder Figure2)"),
    PaperFigure("Figure3", "paper_figures.make_figure3_frozen_transfer", "build", "Figure3",
                ("figure_source_data/Figure3",), "Frozen-core transfer on the 4 x 4 task"),
    PaperFigure("Figure4", "paper_figures.make_figure4_interface", "build", "Figure4",
                ("figure_source_data/Figure4",), "Relation-input transformations"),
    PaperFigure("Figure5", "paper_figures.make_figure5_symbols", "build", "Figure5",
                ("figure_source_data/Figure5",), "Arbitrary relation symbols"),
    PaperFigure("SupplementaryFigureS1", "paper_figures.make_figureA1_control", "build", "FigureA1",
                ("figure_source_data/FigureA1",), "Delayed-reconstruction pretraining"),
    PaperFigure("SupplementaryFigureS4", "paper_figures.make_figureA2_mapping_controls", "build", "FigureA2",
                ("figure_source_data/FigureA2",), "Relation and endpoint mapping controls"),
    PaperFigure("SupplementaryFigureS2", "paper_figures.make_figureA3_mechanism", "build", "FigureA3",
                ("figure_source_data/FigureA3",), "Topology, reduced-dynamics and path-invariance tests"),
    PaperFigure("SupplementaryFigureS3", "paper_figures.make_figureA4_topology", "build", "FigureA4",
                ("figure_source_data/FigureA4",), "Torus-like topology visualization"),
    PaperFigure("SupplementaryFigureS10", "paper_figures.make_figureA6_associations", "build", "FigureA6",
                ("figure_source_data/FigureA6",), "Grid-like unit fraction versus transfer"),
    PaperFigure("Figure6", "hard8_phaseF.make_figure6", "make_figure6", "Figure6",
                ("figure_source_data/FigureA5",), "Extended 8 x 8 task (tables shared with S5 and S6)"),
    PaperFigure("SupplementaryFigureS5", "hard8_phaseF.make_figureS5", "make_s5", "SupplementaryFigureS5",
                ("figure_source_data/FigureA5",),
                "Extended 8 x 8 controls (panel E: all_per_length.csv, held-out splits)"),
    PaperFigure("SupplementaryFigureS6", "hard8_phaseF.make_figureS6", "make_s6", "SupplementaryFigureS6",
                ("figure_source_data/FigureA5", "hard8x8_phaseF_followup/all_dataset_audit.csv"),
                "Optimization, shortcut-baseline and split diagnostics"),
    PaperFigure("Figure7", "hard8_phaseF.make_figure7", "make_figure7", "Figure7",
                ("figure_source_data/Figure7",),
                "Recurrent perturbation and rescaling (C/D renamed to final letters)"),
    PaperFigure("SupplementaryFigureS7", "hard8_phaseF.make_figureS7", "make_s7", "SupplementaryFigureS7",
                ("figure_source_data/SupplementaryFigureS8",),
                "Extended optimization, gain calibration and additive controls"),
    PaperFigure("SupplementaryFigureS8", "hard8_phaseF.make_figureS8", "make_s8", "SupplementaryFigureS8",
                ("figure_source_data/SupplementaryFigureS9",), "Multi-step state maintenance and error accumulation"),
    PaperFigure("SupplementaryFigureS11", "hard8_phaseF.make_figureS9", "make_s9", "SupplementaryFigureS9",
                ("figure_source_data/SupplementaryFigureS10",), "Prespecified source-measure analyses"),
    PaperFigure("SupplementaryFigureS9", "hard8_phaseF.make_figureS10", "make_s10", "SupplementaryFigureS10",
                ("figure_source_data/SupplementaryFigureS11",), "Refitting of navigation-specific mappings"),
    PaperFigure("SupplementaryFigureS12", "hard8_phaseF.make_figureS12", "make_s12", "SupplementaryFigureS12",
                ("figure_source_data/SupplementaryFigureS12",), "Rescaling interventions by path length and source"),
)


def paper_sort_key(number: str) -> tuple[int, int]:
    """Main figures first, then supplementary figures, each by number."""
    if number.startswith("SupplementaryFigureS"):
        return (1, int(number[len("SupplementaryFigureS"):]))
    return (0, int(number[len("Figure"):]))


def in_paper_order(figures=PAPER_FIGURES) -> list[PaperFigure]:
    return sorted(figures, key=lambda figure: paper_sort_key(figure.number))
