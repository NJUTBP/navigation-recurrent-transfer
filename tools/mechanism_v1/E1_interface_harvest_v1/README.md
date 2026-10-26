# E1 interface harvest v1

第一项代码交付，对应方案 v5 第 5 节优先级第 1 位。零 GPU，纯读取冻结产物。

**它要回答的问题**：学到的关系界面，区分迁移能力的是**幅度**还是**形状**？

单 source 的预览（v5 §0）显示形状指标方向是反的，幅度指标方向是对的，且与 8×8 的 gain 排序一致。这个包把那条观察从 1 个 source 扩到 10 个 source 的配对检验。

---

## 1. 放到集群哪里

```bash
export ROOT=<PROJECT_ROOT>
mkdir -p "$ROOT/tools/mechanism_v1"
# 解压后得到 E1_interface_harvest_v1/，整个目录放进去
```

最终路径：

```
$ROOT/tools/mechanism_v1/E1_interface_harvest_v1/
├── README.md
├── scripts/harvest_interface_rows.py
├── scripts/analyze_interface_magnitude.py
└── lsf/e1_harvest.lsf
    lsf/submit_absolute.sh
```

输出与日志（都是新目录，不碰任何既有产物）：

```
$ROOT/rerun_best_validation/followups/mechanism_v1/E1_interface/
$ROOT/HPC_logs_mechanism_v1/E1_interface_harvest_v1/
```

遵守 `env.md` 第 20/21/22/23 条：新实验新包、新日志目录、sidecar 输出，不修改旧 LSF，不写入 `aggregate_full`。

## 2. 怎么跑

```bash
bash "$ROOT/tools/mechanism_v1/E1_interface_harvest_v1/lsf/submit_absolute.sh"
```

这是纯 JSON 读取，几十秒就能跑完，登录节点也完全够用。但按 `env.md` 第 18 条的偏好，默认走 LSF。要在登录节点直接跑就用：

```bash
PY=<PYTHON>
OUT="$ROOT/rerun_best_validation/followups/mechanism_v1/E1_interface"

"$PY" "$ROOT/tools/mechanism_v1/E1_interface_harvest_v1/scripts/harvest_interface_rows.py" \
    --project-root "$ROOT" --out "$OUT" --require-complete

"$PY" "$ROOT/tools/mechanism_v1/E1_interface_harvest_v1/scripts/analyze_interface_magnitude.py" \
    --in "$OUT" --out "$OUT"
```

## 3. 产出什么

| 文件 | 内容 |
|---|---|
| `interface_rows.csv` | 每 (source, target, arm, budget) 一行，预期 10×3×38 = 1140 行 |
| `harvest_manifest.json` | 选中的 30 个输入文件、各自 sha256、每 source 的 target 列表、完整性检查 |
| `candidate_scan.json` | 全部候选路径的扫描日志，含被拒绝的原因 |
| `source_level_interface.csv` | target repeat 在 source 内平均后的表 |
| `interface_contrasts.csv` | 全部声明的配对检验 |
| `recurrence_drop_normalized.csv` | 原始落差与按 chance 归一化的落差 |
| `shape_accuracy_dissociation.csv` | 形状最接近时的准确率差 |
| `INTERFACE_RESULT.md` | 人读报告，可直接贴进讨论 |
| `INTERFACE_RESULT.json` | 机器可读版，供后续作图 |

## 4. 关键实现决定

**目标种子靠 glob 发现，不用公式推。** 遵守 `env.md` 第 26 条。`symbolic_baseline7000_t*` 由精确正则 `^symbolic_t(\d+)$` 排除。

**两种历史布局都搜。** `artifacts/symbolic_t<T>/` 与 `attempts/symbolic_t<T>/try_<NNN>/` 都会被扫到，按 realpath 去重，同一 (source, target) 若仍有多个则优先 `artifacts`，其次取最大的 try 序号。选择过程全部记在 `candidate_scan.json` 里。

**`column_norms` 只在 JSON 里，CSV 没有。** 本次分析的主指标完全建立在它上面，所以必须读 JSON。

**codebook 幅度从 `protocol_manifest.json` 的 `seeds[].action_codebook` 实读，不写死 0.02。** 这样

```
M_rel_codebook = mean_k ||W_rel[:,k]|| / ||codebook 列||
```

与 Phase F 8×8 的 `gain`（它同样乘在同一套 codebook 上）严格同量纲，两个实验可以直接并排比较。

**自带一致性校验。** `adapter_geometry.procrustes_scale` 是把学到的矩阵放大到 codebook 的最优尺度，所以当界面接近 codebook 的旋转加缩放时应有

```
M_rel_codebook ≈ 1 / procrustes_scale
```

脚本把两者的相对差写成 `scale_consistency` 列，并在 manifest 里报 budget ≥ 1000 的中位数。烟雾测试里这个值是 **0.34%**。**如果实跑出来这个中位数不是百分之几的量级，说明字段读错了，先查再看结论。**

**入口门。** `--require-complete` 在没凑齐 10 source × 3 target 时返回非零并写清缺了谁。

## 5. 预注册的检验结构

分析脚本的 docstring 就是预注册文本，**跑之前先读一遍并确认，跑完不要再改**。

- **主要**：最大共同预算处，`M(nav) < M(ctl01)`。这里两者都在准确率天花板上（1.000 与 0.999），所以既是预算匹配也是性能匹配。
- **次要族（Holm，m=4）**：nav 对 random；性能匹配的 `M*`（首次达到 0.9 时的幅度）；nav 的幅度已平；ctl01 的幅度未平。
- **预注册零结果族（Holm，m=3）**：反对称指数、Procrustes 残差、轴角偏离 90° 三项，nav 对 ctl01，**预期不偏向导航**。这一族是故意声明为零的，`exp8` 的 S2 是历史预注册判据，DR 若在形状上更规范而迁移更差，就说明形状是任务强加的。
- **描述性**：形状/准确率解离；归一化递归落差；2-D 累加器（无核）的形状参考线。

统计口径沿用项目惯例：source 为复制单位，target repeat 先在 source 内平均，two-sided 穷举 exact sign-flip，20000 次 source bootstrap（种子 982451653），Holm 分族校正。

## 6. 烟雾测试已做

用你给的那份真实样本复制成 10 source × 3 target 的合成树（一半用 `artifacts` 布局、一半用 `attempts` 布局，另放一个 `symbolic_baseline7000_t1302` 诱饵），两个脚本都跑通：

```
n_sources 10 / n_files 30 / n_rows 1140
sources_missing []
scale_consistency_median 0.0034
诱饵已被排除，两种布局都被正确识别
```

合成树的数值是真实样本加抖动得来的，**那些 p 值没有科学意义**，只用来确认所有代码路径都执行到了。

## 7. 拿到结果之后

1. 先看 `harvest_manifest.json` 的 `scale_consistency_at_budget_ge_1000.median`。不是百分之几就停下来查。
2. 再看 `INTERFACE_RESULT.md` 的第一张表（幅度随预算）。v5 §0 的单 source 预览是 nav 14× / DR 60× / random 108×，看 10 个 source 的平均是否保持这个排序与量级。
3. 主检验通过的话，这一条就可以进稿子，并且和 8×8 的 gain 串成同一条线。**在此之前不要写进稿子。**
4. 把结果发回来，我按数据出 Figure 8 的作图代码，并接进 `redraw_all_figures.py` 的 `BASE_BUILDERS` / `FINAL_MAPPING`。

## 8. 下一步

队列长，建议 E1 提交后立刻并行准备 **E0.9a**（扩展预算收敛对照）。那一项决定全篇机制表述，是排队时间最长的一项，应该尽早排上。E1 的结果不会改变 E0.9a 的设计，两者可以同时推进。
