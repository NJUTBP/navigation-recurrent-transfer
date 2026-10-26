# E0.6 / E0.7 / E0.4 forward diagnostics v1

方案 v5 的 Step 4（递归消融、错误位移、状态诊断）。**纯前向，不训练**，与 E5c 并行跑。

三项共用同一批前向，所以做成一个包。

## 1. 为什么现在做

- **E0.6**：4×4 有递归消融对照（1.000 → 0.101），**8×8 没有**。E0.9a D3 显示学到的增益是 48（@8000）到 74–94（@32000），审稿人必然问在这个工作点上递归还被不被用到。
- **E0.7**：Phase F 显示 matched 在 seen_long 上准确率 0.458、置信度 0.824、ECE 0.366 —— 不像衰减到噪声，像系统性走错。把预测节点映射回格点坐标直接量出来。
- **E0.4**：E5c 要跑很久，而 `δ` 的实测（E5b' 的输入）只是一次前向。顺手产出。

## 2. 这个包解决的唯一难点：冻结核不在 checkpoint 里

Phase F 的 `compact_state_dict` 在 `core != trainable` 时剔除全部 `grid_rnn.*`，`storage_mode = 'frozen_core_omitted_reconstruct_from_origin_and_seed'`。所以必须先重建核再把界面与读出加载上去：

```python
nav_hh, nav_ih, nav_meta = load_navigation_grid_weights(nav_best_pt)
matched_hh, matched_ih, _ = make_matched_random_weights(
    nav_hh, nav_ih, seed=stable_seed32('matched_random', target),
    iterations=8, singular_rank=16, probe_count=64, strict=True)
random_core_seed = stable_seed32('phaseF_fresh_random_core', source, target)
```

`rebuild_model` 做这件事，然后 `load_state_dict(..., strict=False)` 并断言：**unexpected keys 必须为空；missing keys 必须全部是 `grid_rnn.*`**。同时用 checkpoint 自己存的 `navigation_checkpoint_sha256` / `matched_random_seed` / `run_seed` 交叉校验。

### Gate R

重建是否正确**用行为验证，不靠肉眼**：intact 模型在四个测试格上重评一遍，与原始运行写进 `metrics.csv` 的准确率比。超过 `GATE_R_TOLERANCE = 2e-3` 就说明核重建错了，本包所有下游数字都无意义。

**看任何结果之前先看 Gate R。** `COMPLETE.json` 里有 `gate_r_failures` 与 `gate_r_worst_abs_difference`，`check_status_absolute.sh` 会逐 cell 报。

## 3. 格点坐标映射（已从代码确认，不是猜的）

`fullscale_common.BridgeEpisode`：

```python
def coord(node): r, c = divmod(int(node), 8); return [r, c] * 0.02
velocity(a→b) = [row_b - row_a, col_b - col_a] * 0.02
```

所以 `node → (row, col) = divmod(node, 8)`，无需查表，E0.7 的误差向量是精确的。preflight 会断言这个映射铺满 8×8。

## 4. 三项的具体内容

### E0.6 消融模式

| 模式 | 做法 | 测什么 |
|---|---|---|
| `intact` | 原样 | Gate R 的参照 |
| `no_recurrence` | `W_hh ← 0`（上下文管理器，退出时还原） | 递归是否被用到 |
| `last_1/2/3` | 把每行最后 k 个真实步之前的关系输入置零 | 实际用了多长的历史 |
| `shuffle_order` | **每行独立置换真实步的顺序** | 顺序不变性 |

`last_k` 是把早期输入置零而不是缩短轨迹 —— 递归映射仍在那些步上跑，所以隔离的是"早期输入的贡献"，不是"更短的路径"。

`shuffle_order` 是**对计划的新增**：任务是交换群作用，终点等于关系之和，所以顺序置换不改变真值。加性解应当完全不受影响。一次额外前向就能直接量出顺序不变性，与 H4/E3 直接相关。

报告用**按 chance 归一化的落差** `(acc − acc_ablated)/(acc − 1/64)`。原始落差上界就是 `accuracy − chance`，因此主要在跟踪准确率 —— 这是 4×4 那张图掉进去的同一个坑（E1 已实测：原始 0.881/0.550/0.160 归一化后是 0.977/0.962/0.884）。

### E0.7 误差位移

对 `seen_long` 与 `heldout_long` 保存每条轨迹的预测节点与真值，转成整数格点坐标，逐 path_length 报 `E‖e‖`、`E[e]`（带符号的行/列分量）、标准差，另加按 split 聚合的二维位移直方图。

**判定**：`E[e]` 随 L 线性增长且方向一致 → 系统性偏移；`E[e] ≈ 0` 而 `E‖e‖` 增长 → 各向同性扩散。collector 对 `E‖e‖` 与 L 的斜率做 nav vs matched 的配对检验。

### E0.4 状态诊断与 δ

从一次带 `hidden_seq` 的前向逐步记录（`hidden_seq` 是 ReLU 之后的，所以 `h_t > 0` 就是 `1[z>0]`）：

```
hidden_norm          ‖h_t‖
active_fraction      mean(h_t > 0)
signal_vs_zero_input ‖h_t − h_t^{zero-input}‖
drive_ratio_linear   ‖g·W_in u_t‖ / ‖W_hh h_{t-1}‖
delta_rectified      ‖h_t − ReLU(W_hh h_{t-1})‖ / ‖ReLU(W_hh h_{t-1})‖
```

外加 `delta_by_gain`：在模型自己的起始状态上，对 `g ∈ {1,3,10,32,100}` 算

```
delta(g) = E_h ‖ReLU(W_hh h + g·W_in u) − ReLU(W_hh h)‖ / ‖ReLU(W_hh h)‖
```

**这就是 E5b' 需要的量**，在这里是免费的。collector 会算出 `delta_nav / delta_matched`，作为"所需增益比"的预测值，与已观测的 1.46（@8000）、约 1.26（@32000）对照，等 E5c 给出固定增益版本后再做正式检验。

## 5. 运行

```bash
export ROOT=<PROJECT_ROOT>
# 解压到 $ROOT/tools/mechanism_v1/

# 先烟雾跑一个 cell（会先跑 preflight）
bash "$ROOT/tools/mechanism_v1/E0_67_forward_diagnostics_v1/lsf/submit_absolute.sh" --smoke
```

**烟雾 cell 跑完先看 `gate_r.csv`**：20 行（5 条件 × 4 格），`within_tolerance` 应全为 True，`abs_difference` 应在 1e-4 量级。**Gate R 不过就不要提交全量**，先查重建流程。

通过后：

```bash
bash "$ROOT/tools/mechanism_v1/E0_67_forward_diagnostics_v1/lsf/submit_absolute.sh"
bash "$ROOT/tools/mechanism_v1/E0_67_forward_diagnostics_v1/lsf/check_status_absolute.sh"
```

规模：10 sources × 5 targets = **50 个 array job**，每个跑 5 个条件 × 4 格 × 6 模式的前向，外加状态诊断。单 cell 的主要开销是 `make_matched_random_weights`（4096×4096，8 次迭代），只做一次。

默认条件（可用 `--conditions` 覆盖）：

```
navigation_trainable_scalar_gain
matched_random_trainable_scalar_gain
fresh_random_trainable_scalar_gain
navigation_fixed_gain10_frozen
matched_random_fixed_gain10_frozen
```

默认 checkpoint 来源是 Phase F（`--checkpoint-root` 可切到 E0.9a 的 32000 步版本；两者的 `group_*/checkpoints/<cond>/best_validation.pt` 布局相同，group 名由 glob 处理）。

## 6. 本地已验证

```
python3 -m py_compile scripts/*.py                                   ✅
node_to_lattice: node 9 → [1,1]，node 63 → [7,7]，铺满 8×8            ✅
apply_ablation 的掩码与置换逻辑用 numpy 等价复刻验证：
  last_k 每行非零步数 = min(k, L)                                     ✅
  shuffle_order 前 L 步是原前 L 步的置换，padding 仍为 0               ✅
array 索引 50 → 50 个唯一 (source, target)，无重无漏                   ✅
```

**未验证**：GPU 上的实际前向、`rebuild_model` 的 `strict=False` 加载、`ZeroedRecurrence` 的还原、collector 的全部逻辑。烟雾 cell 就是为这些准备的。

## 7. 已知限制

- `state_diagnostics` 与 `delta_by_gain` 默认只用 `seen_long` 的前 512 条（`--state-episodes`），足够估均值，但不要拿它做逐 source 的精细推断。
- `delta_by_gain` 用的是每条轨迹的**第一个**关系向量与 h0，所以它是"起始状态处的一步整流位移"。若要状态平均版本，用 `state_diagnostics.csv` 的 `delta_rectified` 逐步列。
- 误差直方图按 split 聚合（不按 path_length），是为控制文件大小。逐长度的均值在 `error_displacement.csv` 里。
- `shuffle_order` 用 `stable_seed32('e067_shuffle', target, split)` 固定，同一 (target, split) 在所有条件与所有 source 上拿到同一组置换，所以条件之间可配对比较。
