# E0.1 非递归基线 v1（方案 v5 Step 5）

## 做什么

在 Phase F 的 8×8 五个 split 上训练两个没有递归核的基线。它们是算法参照，不是递归记忆对照。

| 基线 | 结构 | 参数量 |
|---|---|---|
| `action_count_linear` | 起点 one-hot 加四种关系的计数，接一个线性分类器 | 4416 |
| `linear_accumulator_2d` | 每个节点一个二维起点向量，关系向量逐步相加，线性读出 | 328 |

## 为什么这样设计

任务是交换群作用，终点等于起点加关系之和。所以起点和关系计数就完全决定答案。

两个基线的函数类都含有精确解。设格点坐标为 p_j，logit_j = 2 p_j·(p_s + D c) − |p_j|² 在目标格点处最大。preflight 把这组精确权重装进两个基线，要求所有 split 准确率等于 1.0。这一步同时验证了关系编码的映射。所以训练后的任何不足都属于学习或外推问题，不属于表达能力问题。

## 与 Phase F 严格配对

- split 由 `build_fullscale_split` 现场重建，并核对 manifest hash，不搬运数据。
- `run_seed`、`loader_seed` 与所有递归条件相同，因此初始化随机流和 minibatch 顺序相同。
- 优化器、学习率、梯度裁剪、只用验证集选模型，全部沿用 Phase F。
- 位移向量到 one-hot 的转换是 `velocity_action_labels` 的向量化版本，preflight 逐条比对全部 85,000 条轨迹。
- 基线没有导航核，所以没有 source 维度。一个 cell 是（target，预算），共 5 × 2 = 10 个。

## 运行

```bash
ROOT=<PROJECT_ROOT>
bash "$ROOT/tools/mechanism_v1/E0_1_nonrecurrent_baselines_v1/lsf/submit_absolute.sh" --smoke
bash "$ROOT/tools/mechanism_v1/E0_1_nonrecurrent_baselines_v1/lsf/check_status_absolute.sh"
bash "$ROOT/tools/mechanism_v1/E0_1_nonrecurrent_baselines_v1/lsf/submit_absolute.sh"
```

smoke cell 写在 `_smoke/` 子目录，永远不会与 array 撞目录。E5c 丢失的那个 cell 就是 smoke 与 array 撞目录造成的。

## 本地已验证

- 五个 target 的 hash 全部匹配，85,000 条轨迹的 one-hot 标签与 `velocity_action_labels` 逐条一致。
- 精确解装入后两个基线在所有 split 上准确率都是 1.0。
- 真实数据 CPU 预演（target 84001）：累加器 8000 步 ID 0.860、seen_long 0.719；32000 步 0.991、0.969。动作计数线性 32000 步 0.582、0.289，仍在上升。
- collector 在真实的 Phase F 与 E0.9a 参照值上跑通。

## 输出

`rerun_best_validation/followups/mechanism_v1/E0_1_nonrecurrent_baselines/aggregate/BASELINE_RESULT.md`
