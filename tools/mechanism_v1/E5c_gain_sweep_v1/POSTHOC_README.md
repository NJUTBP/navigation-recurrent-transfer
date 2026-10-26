# E5c 事后稳健性分析

`scripts/analyze_gain_sweep_posthoc.py` 在看过 E5c 结果之后编写，不替代预注册的 `collect_gain_sweep.py`。它只写 `aggregate_posthoc/`，不碰 `aggregate/`。重跑会重写 `aggregate_posthoc/` 自身。

| 读数 | 内容 |
|---|---|
| P1 | 各 source 在所有增益中的最好值，nav 减 matched；以及逐增益比较 |
| P2 | 平台宽度。标准误取二项误差与 target 间复现误差的较大者 |
| P3 | 阈值穿越增益，在 log2 g 上线性插值，标出左删失和右删失 |
| P4 | 考虑删失的速率面。未达标的 run 记为删失，不从样本里丢掉 |
| P5 | 同一批 target 上学到增益与固定增益的比较 |
| P6 | 过增益敏感度，长程与短程比较 |

```bash
ROOT=<PROJECT_ROOT>
bsub < "$ROOT/tools/mechanism_v1/E5c_gain_sweep_v1/lsf/posthoc.lsf"
cat "$ROOT/rerun_best_validation/followups/mechanism_v1/E5c_gain_sweep/aggregate_posthoc/POSTHOC_RESULT.md"
```

P4 需要逐 cell 的 update_curves，只能在集群上得到真实数值。
