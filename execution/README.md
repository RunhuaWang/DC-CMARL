# Decentralized execution stage

本目录加载训练阶段得到的冻结 Actor，执行论文采用的无中心对偶更新。代码不会更新
Actor 或 critics；每个机器人维护本地 dual，通过四机器人环形图的两轮同步通信恢复
全局 cost，再执行投影更新。

## 固定设置

- 初始 dual：0；dual 区间：`[0, 10]`；步长：0.25；
- cost estimation horizon：64；
- 1024 次 dual updates，共 65,536 个执行步；
- stochastic action sampling；
- evaluation seeds：1000--1019；
- continuing execution，不进行周期 reset。

## 安装

在本目录执行：

```bash
python -m venv .venv
.venv/bin/pip install -e '.[train]'
```

## 八个阈值实验

按需直接运行以下脚本：

```text
src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/run_decentralized_execution_c0_h64.py
src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/run_decentralized_execution_c0p1_h64.py
src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/run_decentralized_execution_c0p2_h64.py
src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/run_decentralized_execution_c0p4_h64.py
src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/run_decentralized_execution_c0p6_h64.py
src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/run_decentralized_execution_c0p9_h64.py
src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/run_decentralized_execution_c1_h64.py
src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/run_decentralized_execution_c1p2_h64.py
```

汇总八组已有日志并重新生成论文图：

```bash
.venv/bin/python src/hrmr/experiments/fixed_responsibility_small_6x6_decentralized_execution/plot_h64_threshold_comparison.py
```

投稿副本只保留生成论文双子图所需的八份 `dual_update_log.csv`、合并后的
`long_run_metrics_summary.csv`，以及最终 PNG/PDF。它们位于：

```text
map_6x6/data/decentralized_execution/
map_6x6/images/decentralized_execution/h64_c0_to_c1p2_comparison/
```

逐阈值轨迹、单独图、额外评价表和诊断结果未纳入本投稿副本。再次运行同一脚本会
覆盖对应阈值的同名结果，不会写入其他阈值目录。
