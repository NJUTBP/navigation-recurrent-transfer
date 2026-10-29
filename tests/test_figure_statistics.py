"""Statistics stored with the figure source data agree with the source-level tables."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "aggregate_full" / "figure_source_data"
CONTRAST_TABLES = [("Figure3", "contrasts.csv", "paired_differences.csv"),
                   ("Figure4", "adapter_contrasts.csv", "adapter_paired_differences.csv"),
                   ("Figure5", "contrasts.csv", "paired_differences.csv"),
                   ("FigureA2", "contrasts.csv", "paired_differences.csv")]


@pytest.mark.parametrize("folder,contrast_file,paired_file", CONTRAST_TABLES)
def test_contrasts_are_means_over_navigation_sources(folder, contrast_file, paired_file):
    contrasts = pd.read_csv(DATA / folder / contrast_file)
    paired = pd.read_csv(DATA / folder / paired_file)
    keys = [k for k in ("family", "contrast") if k in contrasts.columns and k in paired.columns]
    assert (contrasts["n_sources_planned"] == 10).all()
    for _, row in contrasts.iterrows():
        mask = np.logical_and.reduce([(paired[k] == row[k]).to_numpy() for k in keys])
        rows = paired[mask & np.isfinite(paired["difference"].astype(float)).to_numpy()]
        assert rows["source_seed"].is_unique, (folder, row["contrast"])
        assert len(rows) == int(row["n"]), (folder, row["contrast"])
        if len(rows):
            assert np.isclose(rows["difference"].mean(), float(row["mean"]), rtol=1e-9, atol=1e-12)


def test_association_statistics_recompute():
    stats = pytest.importorskip("scipy.stats")
    table = pd.read_csv(DATA / "FigureA6" / "grid_transfer_associations.csv")
    stored = pd.read_csv(DATA / "FigureA6" / "association_statistics.csv")
    for _, row in stored.iterrows():
        z = table[["grid_fraction", row["metric"]]].dropna()
        r, p = stats.pearsonr(z["grid_fraction"], z[row["metric"]])
        rho, ps = stats.spearmanr(z["grid_fraction"], z[row["metric"]])
        assert int(row["n_sources"]) == len(z) == 10
        assert np.allclose([r, p, rho, ps], row[["pearson_r", "pearson_p", "spearman_rho", "spearman_p"]].astype(float),
                           rtol=1e-9, atol=1e-12)
