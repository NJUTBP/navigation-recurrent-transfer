# Step 6：E2、E3、E4 源侧指标 v1

## 做什么

对每个 (source, target) cell，在 H_nav 上测 Nav 核与该 target 的 matched 核。

| 指标 | 含义 |
|---|---|
| E2：S_opp、S_orth、S_cons、r_eff | 单步位移几何，在中心化位移上计算，附随机方向零分布的 z |
| E3：C_comm、C_inv | 两步组合的可交换性与逆元一致性，参照为 F²(h, 0) |
| E4：R²(L)、R_ret、R_ret_diff | 岭回归只在 L ≤ 5 拟合，L = 6 至 12 为外推 |
| E4 辅助：O(L)、c(t, 12)、信号与范数、激活比例 | 顺序依赖、逐步贡献谱 |

工作点每个核 6 个：A（g = 1）、g3、g10、B8（Phase F 学到的增益）、B32（E0.9a 学到的增益，仅 84001 至 84003）、Bstar（与 E0.4 下游第 1 步 δ 匹配的增益）。全部都算，哪个是主工作点由 ANALYSIS_PLAN.md 规定。

## 依赖

E5b′ 包的 delta_common.py、Phase F 包、Phase F 与 E0.9a 的输出树、E0_67 的 aggregate/all_state_diagnostics.csv。preflight 逐项检查。

## 运行

```
bash lsf/submit_absolute.sh --smoke     # 单个 cell，写入 _smoke
bash lsf/check_status_absolute.sh       # 看到 smoke cell: PASS 再继续
bash lsf/submit_absolute.sh             # 50 个 cell 的阵列，结束后自动汇总
```

每个 cell 只做前向与岭回归，单卡几分钟。

## 输出

`rerun_best_validation/followups/mechanism_v1/E234_source_measures/`，每个 cell 一个目录，汇总在 `aggregate/`，结论表是 `SOURCE_MEASURES_RESULT.md`。

## 两条运行时强制校验

1. matched 核的置换与符号由其记录的 seed 与比例重建，必须逐位复现 matched 权重，否则报错退出。
2. Nav 核手工展开的轨迹必须与导航模型自身前向一致（相对差 ≤ 1e−4）。
