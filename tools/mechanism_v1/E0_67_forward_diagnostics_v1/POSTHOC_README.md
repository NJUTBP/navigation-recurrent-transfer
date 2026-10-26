# E0.6 / E0.7 / E0.4 事后分析

`scripts/analyze_forward_posthoc.py` 读 `aggregate/`，只写 `aggregate_posthoc/`。

| 读数 | 内容 |
|---|---|
| A1 | 每种消融下的绝对准确率，nav 减 matched |
| A2 | 理想积分器在 last_k 下的精确准确率（由重建 split 算出），以及各条件超出它的部分；打乱顺序后出格和产生 held-out 双字组的比例 |
| A3 | 平稳性：每步范数保持率、delta 第 12 步与第 3 步之比、第 1 步 delta；同一 origin 内与 seen_long 的 Spearman（探索性） |
| A4 | 误差结构：错误距离随 L 的增长指数、相邻格点错误占比、行列方差比 |

```bash
ROOT=<PROJECT_ROOT>
bsub < "$ROOT/tools/mechanism_v1/E0_67_forward_diagnostics_v1/lsf/posthoc.lsf"
cat "$ROOT/rerun_best_validation/followups/mechanism_v1/E0_67_forward_diagnostics/aggregate_posthoc/FORWARD_POSTHOC.md"
```

本地已在返回的 aggregate 上完整跑通，集群重跑应得到相同数值。
