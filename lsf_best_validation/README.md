# Best-validation checkpoint downstream rerun (LSF)

这套脚本用于在**不重新训练 navigation** 的前提下，对已有的 10 个 navigation sources
重新执行后续认知实验。

## 固定的新主规则

每个 source 固定使用：

```text
sources/nav_seed_<SOURCE>/artifacts/navigation/best.pt
```

`best.pt` 在原始 `exp1_nav_pretrain.py` 中定义为 **minimum lightweight navigation validation task loss**。
预检脚本会读取 `history.json`，确认 `best.pt` 中记录的 epoch 与
`validation_task_loss` 的全训练最小值一致；不一致时直接退出。

为隔离 checkpoint-policy 这一处变化，CTL source 不重新训练，继续复用：

```text
sources/nav_seed_<SOURCE>/artifacts/sequence_control/best_validation.pt
```

## 本轮运行哪些实验

保留并重新运行：

1. `exp5_nav_t*`
   - transfer_freeze
   - transfer_finetune
   - random_init
   - random_frozen
   - transfer_input_only
   - transfer_recurrent_only
2. `exp5_ctl_t*`
3. `mapping_t*`
4. `test_transforms_t*`
5. `adapter_t*`
6. `symbolic_t*`
7. `hard_t*`
8. `factorial_t*`

所有 target seed 块直接从原有 `plan.json` 读取，因此不会重新发明 seed 分配。

## 本轮明确跳过

由于 topology 代码暂时继续修订，本轮**不提交**以下链：

```text
topology_export
formal_topology
source_splits
source_reduced
action_causality
path_invariance
```

因此本轮也**不运行原始 `multinav.aggregate --strict`**。原 aggregator/figure pipeline
期待 topology-dependent Figure A1/A3 产物；在 topology 修复前强行 aggregate 会把
“有意跳过”误判为缺失。

## 放置位置

把整个目录：

```text
lsf_best_validation/
```

复制到原先已完成的 `multinav_paper_v1` run root 下。该目录必须同时包含：

```text
multinav_paper_v1/
├─ plan.json
├─ code/
│  ├─ gpm_cmap/
│  └─ multinav/
├─ sources/
│  ├─ nav_seed_6101/
│  ├─ ...
│  └─ nav_seed_6110/
└─ lsf_best_validation/
```

原 `sources/` 只读，不会被修改。

## 新输出位置

默认：

```text
<run root>/rerun_best_validation/
```

例如：

```text
rerun_best_validation/
└─ sources/
   ├─ nav_seed_6101/
   │  ├─ preflight.json
   │  ├─ source_identity.json
   │  ├─ configs/
   │  ├─ logs/
   │  ├─ state/
   │  ├─ attempts/
   │  ├─ cell_cache/
   │  └─ artifacts/
   │     ├─ exp5_nav_t1000/
   │     ├─ exp5_ctl_t1000/
   │     ├─ mapping_t1000/
   │     ├─ test_transforms_t1000/
   │     ├─ adapter_t1000/
   │     ├─ symbolic_t1000/
   │     ├─ hard_t1030/
   │     ├─ factorial_t1060/
   │     └─ ...
   └─ ...
```

每个 stage 采用 `attempts/try_XXX + artifacts/<stage> symlink` 结构。
已经成功完成且 fingerprint 一致的 stage 会自动 `[reuse]`，便于断点重提。

## 一键提交

先进入原 run root：

```bash
cd /path/to/multinav_paper_v1
mkdir -p HPC_logs_best_validation
bash lsf_best_validation/submit_all_best_validation.sh
```

提交顺序：

```text
preflight
   │
   ├── Exp5/main array       [6101..6110]
   ├── adapter/symbolic      [6101..6110]
   ├── hard benchmark        [6101..6110]
   └── factorial             [6101..6110]
```

四个 GPU arrays 在 preflight 通过后彼此独立，可以并行。

当前每个 array 设置：

```text
[1-10]%2
```

即一个 family 最多同时跑 2 个 source。四个 family 全部提交时理论上最多占用约 8 张 GPU。
如果集群配额不同，可以直接修改四个 `.lsf` 顶部 `%2`。

## 手动提交单个 family

先做预检：

```bash
bsub < lsf_best_validation/run_00_preflight.lsf
```

随后可独立提交：

```bash
bsub < lsf_best_validation/run_10_main_array.lsf
bsub < lsf_best_validation/run_20_adapter_symbolic_array.lsf
bsub < lsf_best_validation/run_30_hard_array.lsf
bsub < lsf_best_validation/run_40_factorial_array.lsf
```

若手工提交，请确认 preflight 已成功完成。

## 覆盖路径/环境

默认 Python：

```text
<PYTHON>
```

可以在提交前覆盖：

```bash
export RUN_ROOT=<CLUSTER_PATH>
export OUTPUT_ROOT=<CLUSTER_PATH>
export PYTHON_BIN=<PYTHON>
export THREADS=8

bash lsf_best_validation/submit_all_best_validation.sh
```

`RUN_ROOT` 必须指向**旧的完整 run root**（含 `plan.json/code/sources`），
`OUTPUT_ROOT` 必须是新的结果目录。

## 资源设置

GPU family 默认：

```text
queue: <LSF_QUEUE>
cores: 8
gpu: num=1
```

与原 `multinav/config.yaml` 的正式队列一致。

预检不申请 GPU。

## 为什么 CTL 不重训

本轮要改变的因素是 navigation source checkpoint rule：

```text
旧：best_grid_balanced.pt
新：best.pt
```

sequence-control source 与该规则无关。复用旧 `sequence_control/best_validation.pt`
能让新的 downstream rerun 与旧队列相比只改变 navigation source checkpoint，
避免无意义地引入第二个 source-training replicate。

## 下游 epoch

这套 LSF **没有把 Exp5 从 100 epoch 改成 300 epoch**。
它严格复用 frozen `code/gpm_cmap/configs/*.yaml` 的正式训练预算。

原因是这次目标是把“best navigation validation checkpoint”确定为新的正式 source policy，
而不是再做 checkpoint-sensitivity experiment。300-epoch 实验已经用于诊断
balanced checkpoint 的慢收敛机制，不应悄悄改变正式下游 protocol。

## 重要：不要覆盖旧结果

不要把：

```bash
OUTPUT_ROOT="$RUN_ROOT"
```

这样会和旧的 `sources/` 混在一起。

推荐始终使用：

```text
rerun_best_validation
```

或另一个全新的目录。

## 完成后检查

```bash
find rerun_best_validation/sources -name 'status_*.json' -print
find rerun_best_validation/sources -path '*/state/*.json' -print
```

查看失败：

```bash
grep -R '"status": "failed"' rerun_best_validation/sources/*/state
```

查看 source checkpoint：

```bash
grep -R '"checkpoint_policy": "best_validation"' \
  rerun_best_validation/sources/*/source_identity.json
```

## 后续 aggregate

等 topology 代码修复后，建议再做一个新的 aggregate/figure pass，
明确把本轮 `best_validation` rerun 作为行为主结果输入，把 topology/mechanism
链作为后补阶段。当前包故意没有提供 aggregate LSF，避免旧 aggregator 把
计划性跳过误报为技术缺失。

## v2 修复：LSF `.lsbatch` 路径

LSF 在执行时会把 `.lsf` 内容复制到 `~/.lsbatch/`。因此不能通过
`BASH_SOURCE[0]` 推断 `run_best_validation_source.py` 所在目录。

v2 使用环境变量 `BEST_VALIDATION_DIR`。一键提交脚本会自动把它设置为实际的
`lsf_best_validation_v2/`（或当前解压目录）并传给所有 batch jobs。

如果手工 `bsub < ...lsf`，建议显式设置：

```bash
export RUN_ROOT=/path/to/multinav_paper_v1
export BEST_VALIDATION_DIR="$RUN_ROOT/lsf_best_validation_v2"
export OUTPUT_ROOT="$RUN_ROOT/rerun_best_validation"
export PYTHON_BIN=<PYTHON>
bsub < "$BEST_VALIDATION_DIR/run_00_preflight.lsf"
```

预检日志现在会打印 `BEST_VALIDATION_DIR` 和 `LAUNCHER`，便于确认实际路径。
