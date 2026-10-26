# E1 addendum 上集群（补齐 Step 1 的 provenance）

`scripts/analyze_interface_matched.py` 就是此前只在本地跑过的 addendum 脚本，内容未改。

`lsf/matched.lsf` 把结果写到 `E1_interface/addendum_cluster/`，与 E1 原有产物分开。若该目录已有结果，作业直接退出，不覆盖。Phase F 数据从 `aggregate_full` 的真实路径（readlink）只读。

```bash
ROOT=<PROJECT_ROOT>
bsub < "$ROOT/tools/mechanism_v1/E1_interface_harvest_v1/lsf/matched.lsf"
```

集群结果应与本地版本 `INTERFACE_ADDENDUM.md` 的数值一致。跨实验相关仍是探索性结果。
