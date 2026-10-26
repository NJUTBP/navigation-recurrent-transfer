# E5c fixed-gain operating curve

方案 v5 优先级第 5 位，因 E0.9a 的 D3 结果提前。**E0.9a 已完成**（60/60 cell，Gate 0 通过），本包是下一个长队列项。

## 1. 为什么现在做这个

E0.9a D3 给出了增益的完整轨迹（`all_gain_trace.csv`，source 级均值）：

```
update      1    1000   2000   4000   8000  12000  20000  28000  32000
nav       1.00   4.18  13.09  32.29  48.50  57.40  70.03  77.10  74.35
matched   1.00   3.75  11.91  40.17  72.10  83.14  91.84  95.55  93.71
```

三件事：

1. **update 8000 处的值（nav 48.50 / matched 72.10）几乎精确重现 Phase F 的 48.59 / 71.04**，这是 Gate 0 之外对流水线的第二个独立确认。
2. **两条曲线在 update 3000–4000 之间交叉。** 2000 步时 nav 13.09 > matched 11.91，与 Phase D 记录的 nav 11.28 > matched 9.10 一致；4000 步后 matched 反超并保持。v4 §0.3 那个"方向翻转过、所以增益比较带预算限定"的问题，现在有了精确的机制解释，不再是悬案。
3. **增益一路爬到约 28000 步才转平**（nav 48.5 → 77.1，matched 72.1 → 95.6），且在准确率早已收敛之后仍在爬。

所以**学到的增益不是"这个冻结核需要多大输入幅度"的干净估计**，它被交叉熵的 margin 压力持续往上推。要拿到操作曲线，必须把增益固定住直接扫。

## 2. 设计

- 增益是 `FixedGainPathIntModel` 里的 buffer，只学 start embedding 与 endpoint readout。
- 条件 spec 用 Phase F 的精确形状 `{'origin','kind':'fixed_gain','gain','core':'frozen'}`，所以 `configure_origin` / `wrap_model` / `diagnostic_row` / `compact_state_dict` 原样复用。preflight 会把我们的 spec 键集与 `navigation_fixed_gain3_frozen` 对比，不一致就中止。
- **g = 1 不在网格里**：E0.9a 的两个 identity 条件就是 g=1，同预算、同 target，collector 直接 join 进来，不重跑。
- 网格 `g ∈ {2,4,8,16,32,64,128,256}`（log2），加上 g=1 共 9 点跨两个数量级，覆盖学到的 48–94 区间与两侧。
- 预算 32000，`eval_every` 200，patience 999，`training_examples` 保持 8000（有 assert），与 E0.9a 完全一致，所以 g=1 anchor 可直接比较。
- 规模：8 gains × 2 origins × 10 sources × **2 targets**（84001/84002）= 320 cell，160 个 array job，每个跑 2 个条件。约 2.7× E0.9a。

条件命名用零填充 `navigation_fixed_gain032_frozen`，与 Phase F 的 `navigation_fixed_gain3_frozen` / `gain10` 不冲突，preflight 会检查碰撞。

## 3. 运行

```bash
export ROOT=<PROJECT_ROOT>
# 解压到 $ROOT/tools/mechanism_v1/

# 先烟雾跑一个 cell（会先跑 preflight）
bash "$ROOT/tools/mechanism_v1/E5c_gain_sweep_v1/lsf/submit_absolute.sh" --smoke

# 确认那个 cell 的 COMPLETE.json 与 metrics.csv 正常后，再提交全量
bash "$ROOT/tools/mechanism_v1/E5c_gain_sweep_v1/lsf/submit_absolute.sh"

# 查状态
bash "$ROOT/tools/mechanism_v1/E5c_gain_sweep_v1/lsf/check_status_absolute.sh"
```

烟雾 cell 是 `source 6101 / target 84001 / gain 32`。要确认三件事：`condition_diagnostics.csv` 里两行的 `gain` 都等于 32.0、`grid_any_trainable` 为 False；`metrics.csv` 的 `stopped_update` 等于 32000；`id_seen_short` 的准确率落在 0.9 以上（g=32 介于 gain10 的 0.973 与学到增益的 0.993 之间，应当不低）。

输出与日志：

```
$ROOT/rerun_best_validation/followups/mechanism_v1/E5c_gain_sweep/
$ROOT/HPC_logs_mechanism_v1/E5c_gain_sweep_v1/
```

本地已验证：三个脚本编译通过；用真实 `PROTOCOL.json` 跑 `derive_protocol()` 得 updates 32000 / eval_every 200 / n_evals 160 / training_examples 8000 / n_cells 320；array 索引映射覆盖全部 160 个唯一 cell 无重无漏。

## 4. 五个预注册读数

`collect_gain_sweep.py` 的 docstring 是预注册文本，跑前读一遍，跑后不要改。

**Q1 形状** —— `accuracy(g)` 逐 split 逐 origin 的曲线，含 g=1 anchor。描述性，但 Q2/Q3 用它的峰位。

**Q2 最优点是否分裂** —— 逐 source 求 `g*_short = argmax_g acc(g, id_seen_short)` 与 `g*_long = argmax_g acc(g, seen_long)`，检验 `log2(g*_long) − log2(g*_short)` 是否为零。
**若非零，说明按 `validation_seen_short`（只含长度 1–5）选出的界面对长程不是最优，长程优势是 out-of-sample 的副产品。** 这条对正文表述影响很大。

**Q3 所需幅度** —— 检验 `log2(g*_matched) − log2(g*_nav)`。这是不依赖增益学习动力学的版本。对照已测的学到增益比：8000 步 71.04/48.59 = 1.46，32000 步约 93.7/74.4 = 1.26。差值 +1.0 代表两倍。

**Q4 完成 D2** —— E0.9a 的 D2 发现 g=1 在 32000 步只到 0.8157，且仍以约 +0.009/1000 上升，**所以它没有建立渐近下界**。Q4 同时报每个 g 的末窗斜率：准确率低**且**斜率平的，坐在更低的平台上；准确率低但斜率还陡的，只是慢。`end_of_budget_slope.csv` 就是这一列。

**Q5 速率面** —— 首次达到验证准确率 0.5 / 0.8 / 0.95 所需更新数对 g 的曲线，H5'' 中"增益决定优化条件数"那一半的直接检验。

## 5. 与 E0.9a 的衔接

collector 用 `--e09a-root` join g=1 anchor，只取 `group_identity_pair` 里两个 identity 条件、且 target 在 {84001, 84002} 内的行。若一行都没 join 到会打 WARNING 并让曲线从 g=2 开始 —— **看到这条 WARNING 就停下来查路径**，没有 g=1 的曲线回答不了 Q4。

## 6. 已知限制

- targets 只用了 2 个（E0.9a 用 3 个）。这是为控制规模。若 Q2/Q3 的结论接近临界，补 84003 即可，array 脚本改 `TARGETS` 和 `[1-160]` → `[1-240]`。
- `argmax` 在离散网格上取，所以 `g*` 的分辨率是 2 倍。Q2/Q3 用 log2 单位正是为此。若峰形很平，`g*` 会抖动；届时应改用加权质心或抛物线插值，但**先看数据再改**。
- 大 g 的 cell 在 32000 步上多半浪费（gain10 在 8000 步已收敛），但保持统一预算换来与 g=1 anchor 的可比性，值得。
