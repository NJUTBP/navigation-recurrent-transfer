# E0.9a extended-budget convergence control

方案 v5 优先级第 2 位。这一项决定全篇的机制表述，排队时间最长，应当尽早提交。

**它要回答的问题**：Phase F 那些结论，是"学得更快"还是"能学会"？

v4 §0 的诊断：8000 更新预算对多数条件是约束性的。`navigation_trainable_scalar_gain` 末 1000 步只涨 0.0018（已收敛），但 `matched_random_trainable_scalar_gain` 涨 0.0084 且仍在升，`navigation_identity_frozen` 涨 0.0497 **而且在加速**（逐窗口 0.0238 → 0.0367 → 0.0309 → 0.0497）。所以主对比是"已收敛 vs 未收敛"，而 gain=1 的 0.333 是陡坡上的快照。

---

## 1. 设计要点

### 1.1 不重写 Phase F，直接 import

`run_extended.py` 把 Phase F 的 `run_group.py` 和 `fullscale_common.py` 作为模块导入，复用它的 `condition_specs` / `configure_origin` / `wrap_model` / `diagnostic_row` / `compact_state_dict` 和全部建模、装权重、评测路径。**只改两样东西**：更新预算 8000 → 32000，条件表裁到四条。

协议不是手抄的，是运行时从 canonical `PROTOCOL.json` 派生的，所以 `expected_manifest_hashes`、`model`、`batch_size`、`split_length_quotas`、`pairing` 等全部与 Phase F 逐字节一致。已验证只有 7 个顶层键变化，全部是有意的。

### 1.2 这是 Phase F 的严格延长，不是独立重跑

Phase F 的种子是

```python
run_seed    = stable_seed32('phaseF_run',    target, protocol['training_examples'])
loader_seed = stable_seed32('phaseF_loader', target, protocol['training_examples'])
```

第三个分量是 `training_examples`（8000 条训练轨迹），**不是更新预算**。本包保持它不变（代码里有 assert），所以模型初始化、minibatch 顺序、数据集全部与 Phase F 相同。**前 8000 步是对 Phase F 轨迹的再执行，不是独立样本。**

`collect_extended.py` 的 **Gate 0** 就是检查这件事：把本次的验证曲线与 Phase F 的 `all_update_curves.csv` 在共同评估网格上逐点对比，报最大与平均绝对偏差。偏差大说明配置有问题，不是发现 —— **先看这个数再看任何结论**。

`eval_every_updates` 也保持 200（不是我在 v5 里写的 400）。评测在 `no_grad` 下用不打乱的 loader，不会扰动训练，但保持与 Phase F 一致就少一个自由度，而且 Gate 0 能在完整网格上比对。160 次评估 << patience 999，早停永远触发不了，`stopped_update == max_gradient_updates` 的硬断言保留。

### 1.3 新增：逐步的增益轨迹

`run_group.py` 只在训练结束后记录一次 gain。本包用一个 `log_gain` 上的 backward hook 记录轨迹（每 50 步一条），**完全不碰 `train_fixed_updates`**。`log_gain` 每次优化更新恰好收到一次梯度，验证在 `no_grad` 下不触发钩子，所以计数就是更新序号。

这是 E0.9a 顺带拿到的最有价值的新数据：能直接看 Nav 与 matched 的增益是先后到达同一水平，还是收敛到不同水平。

### 1.4 规模

10 sources × 3 targets × 2 groups = **60 个 array job**，每个 job 训 2 个条件 × 32000 步 = 120 cell，约 480 个 Phase-F cell 当量。

---

## 2. 部署与运行

```bash
export ROOT=<PROJECT_ROOT>
mkdir -p "$ROOT/tools/mechanism_v1"
# 解压后把 E0_9a_extended_budget_v1/ 整个放进去
bash "$ROOT/tools/mechanism_v1/E0_9a_extended_budget_v1/lsf/submit_absolute.sh"
```

提交脚本**先跑 preflight 再提交**。preflight 不用 GPU、不训练、几秒钟，检查：

- Phase F 模块能否正确 import（并校验 import 到的是 `tools/hard8x8_phaseF_fullscale_v1/scripts` 下那份）
- 协议派生结果（预算、评估次数、早停不会触发）
- 四个条件都在 Phase F 的 spec 表里
- 三个 target 的 run/loader/matched 种子
- `build_fullscale_split` 重建数据并逐个校验 `manifest_hash`
- 10 个导航 checkpoint 都在

结果写到 `PREFLIGHT.json`。**如果 preflight 失败，提交会中止 —— 这是有意的，不要绕过。**

单独跑 preflight：

```bash
PY=<PYTHON>
"$PY" "$ROOT/tools/mechanism_v1/E0_9a_extended_budget_v1/scripts/run_extended.py" --preflight
```

查状态：

```bash
bash "$ROOT/tools/mechanism_v1/E0_9a_extended_budget_v1/lsf/check_status_absolute.sh"
```

输出与日志都是新目录，不碰 `phaseF_fullscale_v1/`，不写 `aggregate_full`：

```
$ROOT/rerun_best_validation/followups/mechanism_v1/E0_9a_extended_budget/
$ROOT/HPC_logs_mechanism_v1/E0_9a_extended_budget_v1/
```

---

## 3. 三条预注册判定

`collect_extended.py` 的 docstring 是预注册文本。**提交前读一遍确认，出结果后不要改。**

**D1 主要** —— 32000 步处 `navigation_trainable_scalar_gain` 减 `matched_random_trainable_scalar_gain` 在 `id_seen_short` 上是否仍显著为正。

- 仍显著 → Phase F 的 +0.0364 升级为渐近能力主张
- 消失 → 改写为样本/计算效率主张，正文与摘要都要改

四个测试格都报，`id_seen_short` 为主，另外三个 Holm 校正。同时输出 `phasef_vs_extended.csv`，直接对照 8000 步与 32000 步的差值变化。

**D2 分水岭** —— `navigation_identity_frozen`（gain 1）训到 32000 步，能否追上 Phase F 里 `navigation_fixed_gain10_frozen` 在 8000 步的 0.9733。

- 追上 → 增益只影响学习**速率**，H5'' 是优化条件数假说
- 追不上 → 增益也影响**可达解**，H5'' 是表示可分辨性假说

**这两条通向完全不同的 Discussion。** 收集器直接给出 verdict 字段和逐 source 的表。

**D3 增益轨迹** —— 两个 scalar 条件的增益在 32000 步处是否已平（用末 10% 相对增长量），以及 Nav 是否仍低于 matched。

若两者都已平且序不变，则 v4 §0.3 给 Phase F 增益比较挂的那个预算限定可以摘掉。注意 Phase D（2000 步）时 Nav 11.28 > matched 9.10，Phase F（8000 步）时 Nav 48.6 < matched 71.0，**方向翻过一次**，所以这条必须由轨迹本身来判，不能靠单点。

---

## 4. 拿到结果后的阅读顺序

1. `PREFLIGHT.json` —— 提交前就该是干净的
2. `aggregate/EXTENDED_RESULT.md` 的 **Gate 0** —— 偏差异常就停下来查配置
3. **D2 verdict** —— 它决定 Discussion 怎么写，比 D1 更影响全篇
4. **D1** 与 `phasef_vs_extended.csv`
5. **D3** 与 `gain_summary.csv`

发回来之后我按结果写 Figure 7（操作点与机制）的作图代码，并接进 `redraw_all_figures.py`。

---

## 5. 已知限制

- **不能从 8000 步的 checkpoint 续训。** Phase F 不保存 optimizer state，用 fresh Adam 续训会改变优化过程。所以本包从 update 0 重训到新预算，这也是 Gate 0 能成立的前提。
- **GPU 非确定性。** `deterministic_algorithms: false`，所以 Gate 0 只能要求小幅漂移，不能要求比特一致。判据是量级：更新 2000 处偏差若超过 0.1，几乎肯定是配置问题而不是数值噪声。
- **E0.9a 只覆盖四个条件、三个 target split。** 若 D1 或 D2 的结论需要扩展，再按 v4 的 E0.9b 加 `*_fixed_gain3/10_frozen` 与三个 `full_core_trainable`，并扩到五个 split。
