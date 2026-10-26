# FigureA5 Phase-F final plotting workflow

## What changes

The final FigureA5 no longer plots the old low-competence 1,000-example negative benchmark.
Its four panels now form one result-driven argument:

A. **Input scale gates frozen transfer**
   - fixed gain 1, 3, 10
   - Navigation versus Nav-derived matched control
   - start+length shortcut shown only as a reference
   - answers the causal "why did the old frozen task underperform?" question.

B. **Navigation advantage replicates across sources**
   - primary Phase-F scalar-interface Navigation-minus-matched difference
   - all 10 navigation sources are shown
   - source-level mean and 95% bootstrap CI are shown
   - makes the inferential unit transparent.

C. **Navigation benefit grows beyond the trained horizon**
   - path lengths 1–5: trained horizon
   - path lengths 6–12: temporal extrapolation
   - Navigation / matched / fresh-random
   - this is the headline result panel.

D. **Preserving navigation dynamics supports extrapolation**
   - frozen recurrent core + learned scalar interface versus full recurrent fine-tuning
   - Navigation / matched / fresh-random
   - uses seen-long accuracy
   - shows that the reusable benefit lies in preserving the navigation recurrence.

## Styling

`make_figureA5_hard.py` imports the existing `common.py` and `nature_style.py`.
No new global style contract is introduced:
- manuscript-wide Arial/Arimo/Liberation Sans fallback stack;
- existing 7-pt-scale typography;
- existing 0.75-pt axes and 1.5-pt primary data strokes;
- Navigation keeps the manuscript blue;
- recurrent controls remain neutral grey;
- uppercase panel labels use the existing helper;
- output remains SVG/PDF/PNG/TIFF (+ EPS when pdftops is available).

## aggregate_full

Before the freeze step, Phase-F lives in:
`rerun_best_validation/hard8x8_diagnostics/audit_v1/phaseF_fullscale_v1/aggregate`

The freeze script makes the user's normal local workflow self-contained by creating:

`aggregate_full/hard8x8_phaseF_followup/`
- complete frozen Phase-F aggregate tables.

`aggregate_full/figure_source_data/FigureA5/`
- only the tables required for the final FigureA5.

The previous FigureA5 source folder is preserved once as:
`aggregate_full/figure_source_data/FigureA5_legacy_pre_phaseF/`

The legacy folder is **not read by the new FigureA5 code**. It exists only for provenance.
Thus the old negative result is removed from the figure/story without destructively deleting historical data.

## Local redraw

The user's existing `redraw_all_figures.py` can remain unchanged.
It already imports `paper_figures.make_figureA5_hard` and reads from
`aggregate_full/figure_source_data`.

Do **not** run an old unpatched `prepare_source_data.py` after freezing Phase-F FigureA5.
The patched `prepare_source_data.py` supplied here detects
`PHASEF_FIGURE_SOURCE_MANIFEST.json` and preserves the frozen FigureA5 source data.

## Files in this package

- `freeze_phaseF_figureA5_into_aggregate.py`
- `paper_figures_patch/make_figureA5_hard.py`
- `paper_figures_patch/prepare_source_data.py`
- `preview/FigureA5.{png,svg,pdf,tiff}`
- `CLUSTER_AND_LOCAL_COMMANDS.txt`
