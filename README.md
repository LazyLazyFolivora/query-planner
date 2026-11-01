# query-planner

李明洋维护的查询与工具路由方法。雷达每完成一个动作后，把真实回执、证据缺口、合法候选和剩余额度交给 `radar_policy.decide(request)`；方法选择下一动作或建议 complete/insufficient。雷达承担执行、权限、模型调用、持久化和恢复。

## 安装与调用

雷达在线策略只需 Python 3.10+ 和 NumPy，不加载模型权重、不联网、不独立持有商业模型密钥：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-policy.txt
python -m unittest discover -s tests -v
```

雷达设置 `BIORADAR_QUERY_PLANNER_PROJECT_DIR` 为本目录。当前雷达使用 external.v2；请求及样例见 [contracts](contracts/)。`document_search`、原文读取、补采、dataset_prepare、analysis_run、professional_assess 和图查询均为候选动作，宿主收到决定后执行一项，再以真实结果重新决策。候选声称能覆盖缺口，不等于已经取得支持。

图谱动作可以查实体关系、邻域与路径，实际 Cypher 在 Fabric 图服务内运行。材料、模型调用在动作间可以由外部程序安排；已发出的单次模型生成本身不由路由器内部干预。`complete` 只表示宿主必要检查满足，不代表研究结论正确或 BioCheck PASS。

prepared/sent/unknown/unconfirmed 回执阻止重复发送；恢复先由宿主核对既有调用。默认使用 `HeuristicSelector` 与 `RuleStop`，标记 `heuristic_untrained`。没有已验证的训练权重，不能声称学习方法已优于 baseline。原图谱 demo 及独立 v1 接口保留供原实验使用，雷达走单一当前合约。

## 独立研究与公开任务

完整图谱 demo 需要 `requirements.txt` 中的 Torch/编码器依赖；`run_demo.py --preset rule --quiet` 为规则基线，模型头及训练脚本用于独立研究。外部路由状态与图谱 demo 状态不同，旧 checkpoint 不能直接混用。

Data Factory 发布的 `public/planner_questions.jsonl` 由 [run_public_questions.py](scripts/run_public_questions.py) 独立求解，不能读隐藏答案、生成轨迹或目标条件：

```sh
python scripts/run_public_questions.py --questions release/public/planner_questions.jsonl   --backend primekg --world public-world.sqlite --environment-ref YOUR_PUBLIC_ENVIRONMENT   --out fresh-solver-trajectories.jsonl
```

当前离线环境支持 PrimeKG SQLite 或 ToyKG；只接受已映射的 relation_access/tool_calls/tokens，CTD/PubMed 的独立预算维度仍未接入。求解结束后另做隐藏答案评测。

## 跑一个完整问题与检查

在 BioRadar 提问“BCMA×CD3 双特异性抗体管线有哪些证据和风险？”，选择 query_planner；查看逐步路由、原文、专业评价、报告核验及追问。完整启动、四项集成检查和逐步示例见 [BioRadar README](../BioRadar/README.md)。本仓库无独立的四项 Makefile 目标；本地单元检查使用上面的 unittest 命令。

## 本轮结果与未完成

2026-11-01 单一运行记录在 `/home/zk_test/zk_projects/.branch-archive-20261101/runs.jsonl`。32 项业务分析共 233 次新模型请求，已知 token 下界 5,863,856、1 次用量未知；A/B/C PASS 为 0/3，四个 baseline 出稿 0/4，D 两个更早判断正确终点均未证明。这是系统结果，不是该策略的独立效果分数。

减负比较重放 341 个策略决定，结果未变；实际选中图谱动作 0。下一步由李明洋和雷达研究维护人修复查询引用、重要缺口与覆盖判断，使用原回执回归，再以独立材料比较当前启发式方法和 baseline。公共 CTD/PubMed 环境与策略训练仍需实现和独立评测。

主线仅保留 main，已正常推送。旧分支在同一归档目录的 query-planner.bundle 与 refs-manifest.tsv。
