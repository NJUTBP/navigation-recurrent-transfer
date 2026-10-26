# E5b′ 整流位移预测所需增益 v1（方案 v4 的 E5b′）

## 做什么

在两个固定状态集上测

```
delta_core(g) = E_h || ReLU(W h + g W_in u) − ReLU(W h) || / || ReLU(W h) ||
```

u 取 8×8 的四个关系向量（±0.02 m）。再检验预测 log2(g_matched / g_nav) = log2(delta_nav / delta_matched)。

## 为什么不用 E0.4 的 delta

模型除读出层外没有偏置，(h, g) → (c·h, c·g) 只把整条轨迹放大 c 倍。所以 delta 只取决于 g/‖h‖。E0.4 在各模型学到的起点状态上算 delta，而起点尺度本身是学出来的。同一个导航核，fixed10 与 scalar 两种起点给出的 delta(g=1) 相差 2.7 倍。所以本包改用学习之前就固定的状态集。

| 状态集 | 定义 | 角色 |
|---|---|---|
| H_nav | 导航验证分布上的轨迹状态。两个核用同一组速度序列、同一个位置细胞起点，各走各的动力学，取第 3 步以后同一批位置 | 正式 |
| H_init | 下游模型初始化时的 64 个起点状态。run_seed 固定，导航与 matched 条件的起点完全相同 | 一致性检查 |

## 判定规则（已在第一次实测前写死在 collect_delta.py 的 docstring 里）

- 观测：E5c 事后分析里 id_seen_short 准确率 0.90 处的阈值穿越右移。
- 误差 e = 观测 − 预测，逐 source 计算，n = 10。
- 95% bootstrap 区间落在 ±0.5 log2 内判为支持；exact p < 0.05 且区间不含 0 判为证伪；其余判为不确定。
- 0.5 是 E5c 网格步长的一半，观测侧分辨不了更小的差异。

## 顺带产出

两个核在 H_nav 上的自治保持率 ‖ReLU(W h)‖/‖h‖，以及范数沿轨迹的对数斜率。F(h) = ReLU(W h) 是一次齐次的，所以保持率只取决于状态方向。任何增益和起点缩放都改变不了它，它是核本身的性质。这是 E0.4 平稳性差异的源侧版本。

## 运行顺序

E5b 的 collector 需要 E5c 事后分析的输出，所以先跑 E5c 事后分析。

```bash
ROOT=<PROJECT_ROOT>
bash "$ROOT/tools/mechanism_v1/E5b_delta_prediction_v1/lsf/submit_absolute.sh" --smoke
bash "$ROOT/tools/mechanism_v1/E5b_delta_prediction_v1/lsf/check_status_absolute.sh"
bash "$ROOT/tools/mechanism_v1/E5b_delta_prediction_v1/lsf/submit_absolute.sh"
```

## 每个 cell 内的自检

- 导航核与 matched 核装进模型后逐元素比对重建权重，不一致就中止。
- H_init 的起点状态直接取自模型自己的前向。各 trajectory batch 重新算出的 h0 与它的最大偏差，不得超过起点状态最大值的 1e-4。实测值记在 COMPLETE.json 的 `h0_max_relative_deviation`。
- 导航 checkpoint 里的递归核权重必须等于 `load_navigation_grid_weights` 读出的权重。

## 本地已验证

用一个随机权重的合成导航 checkpoint 跑通了完整代码路径，包括 H_nav 轨迹生成、matched 重建（strict=True）和上述三项自检。collector 用真实的 E5c、Phase F、E0.9a 观测值加合成 delta 跑通。合成 delta 的数值没有意义。
