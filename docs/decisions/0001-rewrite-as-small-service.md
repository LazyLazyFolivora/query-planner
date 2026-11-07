# 0001 Rewrite as a small, stateless decision service

Status: accepted (2026-11)

## Context

The planner started as a demo that mixed a knowledge-graph explorer, a model client, a v1 external interface and
training code with the one thing BioRadar needs: choosing the next research step.

## Decision

The whole thing became the small service described in the README (BioRadar refactoring spec 3.3): one endpoint,
no model calls, no network, no files, the same input always gives the same output.

What was dropped and what was kept:

整个重写成上面这个小服务（见 BioRadar 重构规格 3.3）。删掉了：旧的 v1 外部接口、与雷达无关的知识图谱演示（解析、实体链接、Torch 网络、`kg.py`、`primekg.py`）及其测试、数据和脚本、1748 行的合约 schema 和样例、源码哈希（`radar_policy.py`）、`run_demo.py`、模型客户端和 `.env` 加载、Torch 依赖，以及只服务旧演示的训练代码。保留并简化了 `kdd.py` 的打分、停止规则、“答完须有当前证据”、语义去重和翻页/关键记录的优先级；Fabric 游标和目录发现的专用校验不再放在这里，翻页由雷达当作普通候选提供。

## Consequences

- The scoring is a fixed-prior heuristic; nothing is trained (see the README).
- Paging and Fabric cursor or catalogue checks are the radar's business; a page is an ordinary `next_page`
  candidate here.
