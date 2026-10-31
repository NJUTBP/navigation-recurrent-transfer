"""Figure numbers, builders and source-data folders match the article."""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"
AGGREGATE = REPO / "data" / "aggregate_full"
sys.path.insert(0, str(FIGURES))

import figure_numbering as fn  # noqa: E402

EXPECTED_NUMBERS = [f"Figure{i}" for i in range(2, 8)] + [f"SupplementaryFigureS{i}" for i in range(1, 13)]
# Builder output -> article figure, and article figure -> exported source data, as used for the article.
BUILDER_TO_FIGURE = {
    "Figure2": "Figure2", "Figure3": "Figure3", "Figure4": "Figure4", "Figure5": "Figure5",
    "Figure6": "Figure6", "Figure7": "Figure7", "FigureA1": "SupplementaryFigureS1",
    "FigureA3": "SupplementaryFigureS2", "FigureA4": "SupplementaryFigureS3", "FigureA2": "SupplementaryFigureS4",
    "SupplementaryFigureS5": "SupplementaryFigureS5", "SupplementaryFigureS6": "SupplementaryFigureS6",
    "SupplementaryFigureS7": "SupplementaryFigureS7", "SupplementaryFigureS8": "SupplementaryFigureS8",
    "SupplementaryFigureS9": "SupplementaryFigureS10", "SupplementaryFigureS10": "SupplementaryFigureS9",
    "FigureA6": "SupplementaryFigureS12", "SupplementaryFigureS12": "SupplementaryFigureS11",
}
FIGURE_TO_SOURCE_DATA = {
    "Figure2": ("figure_source_data/Figure2",), "Figure3": ("figure_source_data/Figure3",),
    "Figure4": ("figure_source_data/Figure4",), "Figure5": ("figure_source_data/Figure5",),
    "Figure6": ("figure_source_data/FigureA5",), "Figure7": ("figure_source_data/Figure7",),
    "SupplementaryFigureS1": ("figure_source_data/FigureA1",), "SupplementaryFigureS2": ("figure_source_data/FigureA3",),
    "SupplementaryFigureS3": ("figure_source_data/FigureA4",), "SupplementaryFigureS4": ("figure_source_data/FigureA2",),
    "SupplementaryFigureS5": ("figure_source_data/FigureA5",),
    "SupplementaryFigureS6": ("figure_source_data/FigureA5", "hard8x8_phaseF_followup/all_dataset_audit.csv"),
    "SupplementaryFigureS7": ("figure_source_data/SupplementaryFigureS8",),
    "SupplementaryFigureS8": ("figure_source_data/SupplementaryFigureS9",),
    "SupplementaryFigureS9": ("figure_source_data/SupplementaryFigureS11",),
    "SupplementaryFigureS10": ("figure_source_data/SupplementaryFigureS10",),
    "SupplementaryFigureS11": ("figure_source_data/SupplementaryFigureS12",),
    "SupplementaryFigureS12": ("figure_source_data/FigureA6",),
}


def test_every_article_figure_appears_once():
    assert [f.number for f in fn.in_paper_order()] == EXPECTED_NUMBERS
    assert len({f.builder_output for f in fn.PAPER_FIGURES}) == len(fn.PAPER_FIGURES)


def test_mapping_matches_the_article_mapping():
    assert {f.builder_output: f.number for f in fn.PAPER_FIGURES} == BUILDER_TO_FIGURE
    assert {f.number: f.source_data for f in fn.PAPER_FIGURES} == FIGURE_TO_SOURCE_DATA


def test_builders_and_source_data_exist():
    for f in fn.PAPER_FIGURES:
        path = FIGURES / (f.builder.replace(".", "/") + ".py")
        assert re.search(rf"^def {f.function}\(", path.read_text(encoding="utf-8"), re.M), (path, f.function)
        for rel in f.source_data:
            assert (AGGREGATE / rel).exists(), (f.number, rel)
    assert (AGGREGATE / "hard8x8_phaseF_followup").is_dir()


def test_entry_points_share_the_table():
    import export_source_data_final
    import redraw_all_figures
    assert redraw_all_figures.PAPER_FIGURES is fn.PAPER_FIGURES
    for old in export_source_data_final.FIGURE7_RENAME:
        assert (AGGREGATE / "figure_source_data" / "Figure7" / old).is_file()


def test_style_module_copies_are_identical():
    a = (FIGURES / "paper_figures" / "nature_style.py").read_bytes()
    b = (FIGURES / "hard8_phaseF" / "nature_style.py").read_bytes()
    assert a == b
