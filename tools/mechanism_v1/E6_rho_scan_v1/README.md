# E6：符号混合 ρ 扫描 v1

## 做什么

用 Q = D P 构造 ρ 核，W′ = Q W Qᵀ，W_in′ = Q W_in。全部线性不变量保持，只改变 ReLU 整流几何。每个 (source, target, ρ) 训练一个学习标量增益的 8×8 迁移模型，训练完全沿用 E0.9a（32000 步、同一种子与小批次顺序）。训练后在同一个核上测下游状态剖面，以及 E2、E3、E4（工作点 A、g3、g10、B、Bstar）。

ρ 取 0、0.05、0.1、0.2、0.35、0.5，另加 0.9 作对称性自检。10 个 source × 3 个 target × 7 档，共 210 个 cell。

## 构造器

rho_core.py 的 signed_similarity_core。原函数 _signed_similarity_candidate 不动（它最少翻转 1 个单元，表达不了 ρ = 0）。同一 seed 下所有 ρ 共用一个置换与一个单元次序。ρ ≤ 0.5 翻转次序的前 round(ρH) 个，翻转集随 ρ 嵌套。ρ > 0.5 翻转 1 − ρ 那一档的精确补集，因此 D_{1−ρ} = −D_ρ 严格成立。

## 运行（必须按顺序）

```
bash lsf/submit_absolute.sh --smoke      # ρ = 0，6101/84001，完整 32000 步
bash lsf/check_status_absolute.sh        # 打印 ρ = 0 与同一 cell 的 Nav 准确率对比
bash lsf/submit_absolute.sh --anchors    # ρ = 0、0.1、0.9，共 90 个
bash lsf/check_status_absolute.sh        # 看锚点表
bash lsf/submit_absolute.sh --rest       # 其余 4 档，共 120 个
```

每一阶段结束后自动运行汇总。每次提交前 preflight 用真实导航权重复核构造器的全部代数锚点，不通过就拒绝提交。

## 输出

`rerun_best_validation/followups/mechanism_v1/E6_rho_scan/`，目录层级为 source、target、rho_千分值（如 rho_0100）。结论表是 `aggregate/RHO_SCAN_RESULT.md`。

## 依赖

E0.9a 包的 extended_common.py（协议与 GainTrace），本包同批部署的 E234 包（源侧指标），E5b′ 包的 delta_common.py。

## 未实现

逐档重学导航读出的可分离性对照。
