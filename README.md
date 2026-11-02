# query-planner

## 这是什么

一个很小的 HTTP 决策服务：BioRadar 每做完一步研究，就把“问题、待查问题清单、现在能做的动作、做过的动作、剩余额度”发给它，它回答三件事之一：

- `continue`：下一步做第几个候选动作（返回候选的下标）；
- `complete`：必答的问题都有证据了，可以收尾；
- `insufficient`：做不下去了（额度用完、没有候选、没有预期收益等），带着现有结果收尾。

它**不调用大模型、不联网、不读写文件**，同样的输入永远得到同样的输出。动作由雷达自己执行；“complete”只表示必答检查已有证据，不代表结论正确，也不代表报告核验通过。

维护人：李明洋。研究方法部分（雷达里怎么用这个决策）由孙博文负责。

**当前是启发式方法，没有训练过。** 打分里的先验是固定常数，不是学出来的权重，不能说它已经优于 baseline。基于历史数据学习下一步动作是后续工作，旧的训练代码已随旧演示一起删除。

## 怎么运行

需要 Python 3.10 以上。

```sh
python -m venv .venv
.venv/Scripts/activate        # Linux/macOS 用 . .venv/bin/activate
python -m pip install -r requirements.txt

export SERVICE_TOKEN=换成一个长随机串   # Windows PowerShell: $env:SERVICE_TOKEN="..."
uvicorn planner.service:app --port 8000
```

- `GET /health`：不需要令牌，返回 `{"status": "ok"}`。
- `POST /v1/decide`：请求头必须带 `Authorization: Bearer <SERVICE_TOKEN>`。服务端没设置 `SERVICE_TOKEN` 时拒绝所有 `/v1/` 请求（503），不会放行。
- 所有错误统一为 `{"error_code": "...", "message": "..."}`：`unauthorized`(401)、`token_not_configured`(503)、`invalid_request`(422，message 里写出错的字段)、`not_found`(404)、`method_not_allowed`(405)、`internal_error`(500)。

## POST /v1/decide

请求（`candidates` 的下标从 0 开始，`policy` 可省略，默认 `planner`，也可写 `baseline`）：

```json
{
  "question": "BCMA×CD3 双特异性抗体管线有哪些证据和风险？",
  "checks": [
    {"id": "Q1", "text": "在研管线", "required": true, "importance": 1.0, "status": "open", "evidence": []},
    {"id": "Q2", "text": "安全性风险", "required": true, "importance": 0.8, "status": "answered", "evidence": ["E1"]}
  ],
  "candidates": [
    {"kind": "search", "source": "pubmed", "cost": 1, "target_checks": ["Q1"], "args": {"query": "BCMA CD3 bispecific pipeline"}},
    {"kind": "read",   "source": "pubmed", "cost": 1, "target_checks": ["Q1"], "args": {"ids": ["E3"]}},
    {"kind": "search", "source": "weknora", "cost": 1, "target_checks": ["Q1"], "args": {"query": "BCMA CD3 管线"}}
  ],
  "history": [
    {"kind": "search", "source": "pubmed", "target_checks": ["Q1"], "gain": 0.0, "args": {"query": "bcma cd3"}}
  ],
  "budget": {"steps_left": 8, "reads_left": 15}
}
```

响应：

```json
{"decision": "continue", "candidate": 1, "reason": "best_score", "scores": [0.323232, 0.484848, 0.484848]}
```

意思是“做 `candidates[1]`，也就是读 E3”。`scores` 与候选一一对应；已经做过、重复、付不起的候选得分是 0。

字段说明：

- `checks[].status`：`open` / `answered` / `conflict`。`importance` 在 0 到 1 之间。
- `candidates[].kind`：`search`、`read`、`next_page`、`fetch`、`dataset`、`analysis`、`mechanism`。翻页就是一个普通的 `next_page` 候选，由雷达构造（带上游标等参数）。`cost` 与 `budget.steps_left` 同一单位，默认 1。
- `history[]`：做过的动作；`gain` 是它带来的信息增益（0 到 1），`args` 用来识别“已经做过”。
- 以下是**可选**字段，不写就取默认值，需要雷达额外判断时才用：候选的 `impact`（命中后对答案的影响，默认 1）、`scope_match`（与问题范围的吻合度，默认 1）、`critical`（读的是更正、反证等关键记录）、`names_object`（查的是问题里点名但还没读过的对象）；历史的 `status`（`completed` 默认 / `failed` / `pending`，发出了但还没确认的写 `pending`）。
- 多写了不认识的字段会得到 422，而不是被悄悄忽略。

## 打分和停止规则（大白话）

**什么叫“答完了”**：必答检查的状态是 `answered` 且 `evidence` 不为空。只写 `answered` 没有证据，按未答处理。`conflict` 也不算答完。

**打分**：对每个能做的候选，把它要查的、还没答完的检查逐个算 `重要性 × 后验增益 × impact × scope_match`，加起来，再除以 `1 + 0.25 × 成本压力`。

- 后验增益：同一类动作（kind + source，比如“pubmed 的搜索”）过去的 `gain` 与先验 0.5 的平均，先验相当于 2 次观察。做过几次都没收获，这类动作的分数就降下来；失败的算 0。
- 成本压力：`cost / steps_left`；读取还要和 `1 / reads_left` 比，取大的。越接近用完额度越贵。
- 去重：kind、source、args 都相同就是同一个动作。大小写、多余空格、`terms` 的顺序不算区别。做过的和列表里重复出现的（只保留第一个）都不再考虑。
- 保底分：三类“明摆着没做完的事”不会因为估计增益为 0 就被跳过——翻页（`next_page`）、点名对象的查询（`names_object`）、读取与未答必答检查相关的记录或 `critical` 记录。它们的分数至少是 `0.25 × scope_match / (1 + 成本压力)`，并且排序时先于普通动作：先翻页，再点名对象，再关键记录读取，最后才按分数。

**停止规则，按这个顺序检查，先满足的先生效**：

1. 历史里有 `pending` 的动作 → `insufficient`（先等雷达确认上一步）。
2. 必答检查都答完，且没有“还能做的”翻页、点名对象、关键记录读取 → `complete`。（没有任何必答检查时不会 complete。）
3. `steps_left` 为 0 → `insufficient`（`step_limit`）。
4. 没有候选（含全被去重）→ `insufficient`（`no_candidates`）。
5. 有候选但都付不起（成本超过剩余步数，或读取而 `reads_left` 为 0）→ `insufficient`（`nothing_affordable`）。
6. 最高分 ≤ 0 → `insufficient`（`no_expected_gain`）。
7. 否则 `continue`，选排序第一的。

## baseline

请求里写 `"policy": "baseline"`。固定顺序，不看分数：先把每个不同的 `search` 候选各做一次（按给定顺序），再按给定顺序做 `read` 候选；其它类型不用。`reads_left` 为 0、`steps_left` 为 0 或没有可做的就停，必答检查都有证据则 `complete`，否则 `insufficient`。用来和 planner 对比。

## 代码

- `planner/models.py`：请求和响应（pydantic）。
- `planner/policy.py`：打分、去重、停止规则。
- `planner/baseline.py`：固定顺序的 baseline。
- `planner/service.py`：FastAPI、令牌、统一错误。

## 测试

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

不联网。覆盖：停止规则顺序、打分次序、去重、翻页优先、“答完但没有证据”被降级、baseline 顺序，以及令牌和错误格式。

## 这次改了什么

整个重写成上面这个小服务（见 BioRadar 重构规格 3.3）。删掉了：旧的 v1 外部接口、与雷达无关的知识图谱演示（解析、实体链接、Torch 网络、`kg.py`、`primekg.py`）及其测试、数据和脚本、1748 行的合约 schema 和样例、源码哈希（`radar_policy.py`）、`run_demo.py`、模型客户端和 `.env` 加载、Torch 依赖，以及只服务旧演示的训练代码。保留并简化了 `kdd.py` 的打分、停止规则、“答完须有当前证据”、语义去重和翻页/关键记录的优先级；Fabric 游标和目录发现的专用校验不再放在这里，翻页由雷达当作普通候选提供。
