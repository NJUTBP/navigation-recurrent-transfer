# mechanism_v1 判定 v1

按 ANALYSIS_PLAN.md（含 2026-09-20 修订 1）逐条机械判定 E2、E3、E4、E6 的预注册检验，并统一做 Holm。只读两个 collector 的 aggregate 与 E0.9a 的 aggregate/all_metrics.csv，不训练、不前向。

| 输出 | 内容 |
|---|---|
| VERDICTS.md、VERDICTS.json | 锚点、主效应、主推断、S1a 至 S5 与 Holm 调整后的判定 |
| validation.csv | 用独立代码从逐 cell 表重算 12 个量，与 collector 相差超过 1e−9 即中止 |
| exploratory_*.csv | 探索性表，不参与任何判定 |

运行：`bash lsf/submit_absolute.sh`，结束后 `bash lsf/check_status_absolute.sh`。输出目录已完成则跳过，非空但未完成则报错退出。
