# E6 autonomous-retention intervention v1

Post-E6 preregistered bidirectional intervention derived from C1.

- nav_impair: E6 rho=0 core, W_hh multiplied by c=0.8032013898784912.
- rho01_rescue: E6 rho=0.1 core, W_hh multiplied by 1/c=1.245017765907104.
- W_in is unchanged.
- 10 sources x 3 targets x 2 arms = 60 training cells.
- Training/evaluation is identical to E6/E0.9a: frozen core, learned scalar gain, 32000 updates.
- Existing E6 rho=0 and rho=0.1 cells are paired baselines and are not rerun.

The registered interpretation is in ANALYSIS_PLAN_retention_intervention.md. Do not tune c after observing results.

This experiment requires a GPU-capable queue for practical runtime. The shipped LSF keeps the project's historical queue `<LSF_QUEUE>`; if that queue is unavailable, change only the submission queue when an authorized GPU queue is known. Do not run the 60 full training cells on a CPU-only queue.
