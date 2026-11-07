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

CPython 3.12，依赖只从锁文件装（见下文「开发环境与测试」）。

```sh
make venv                     # .venv，只按 requirements.lock 安装，每个包都校验 sha256
. .venv/bin/activate

export SERVICE_TOKEN=换成一个长随机串   # Windows PowerShell: $env:SERVICE_TOKEN="..."
uvicorn planner.service:app --host 127.0.0.1 --port 18561   # 18561 是现役规划服务的端口
```

- `GET /health`：不需要令牌，返回 `{"status": "ok", "version": "0.2.0", "kinds": [...], "channels": [...]}`。`version` 只在契约或决策行为变化时才改，每个响应的 `planner_version` 与它相同；`kinds` 是接受的动作种类，`channels` 是接受的检索通道。雷达据此在发送新种类或新通道之前确认这个 planner 认得（旧版只返回 `status`）。
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
{"decision": "continue", "candidate": 1, "reason": "best_score", "scores": [0.323232, 0.484848, 0.484848],
 "reasons": ["scored", "open_required_read", "scored"], "channels": [],
 "checks": [{"id": "Q1", "state": "open", "required": true, "reason": ""},
            {"id": "Q2", "state": "answered", "required": true, "reason": ""}],
 "planner_version": "0.2.0"}
```

意思是“做 `candidates[1]`，也就是读 E3”。`scores` 与候选一一对应；已经做过、重复、付不起的候选得分是 0。`reasons`、`channels`、`checks`、`planner_version` 是决策记录，见下文“来源、通道和决策记录”。

字段说明：

- `checks[].status`：`open` / `answered` / `conflict`。`importance` 在 0 到 1 之间。
- `candidates[].kind`：`search`、`read`、`next_page`、`fetch`、`dataset`、`analysis`、`mechanism`。翻页就是一个普通的 `next_page` 候选，由雷达构造（带上游标等参数）。`cost` 与 `budget.steps_left` 同一单位，默认 1。
- `history[]`：做过的动作；`gain` 是它带来的信息增益（0 到 1），`args` 用来识别“已经做过”。
- 以下是**可选**字段，不写就取默认值，需要雷达额外判断时才用：候选的 `impact`（命中后对答案的影响，默认 1）、`scope_match`（与问题范围的吻合度，默认 1）、`critical`（读的是更正、反证等关键记录）、`names_object`（查的是问题里点名但还没读过的对象）；历史的 `status`（`completed` 默认 / `failed` / `pending`，发出了但还没确认的写 `pending`）。
- 多写了不认识的字段会得到 422，而不是被悄悄忽略。
- `series[]`（可选，缺省为空）：问题所依据的数值序列点和它们现在的质量状态，只读输入，不改变任何决策——规划器没有办法把一个值放进序列，只能选择重读它的来源记录。每个点带 `source`、`record_id`、`current`（是否在雷达当前有效序列里：被现行核验器验过且仍可读）、`extraction_check`（BioCheck 抽取核验结果：`status` / `gate` / `checker_version`）和 `comparability`（`provisional` / `comparable` / `not_comparable` 及原因）。只要某个点现在不在有效序列里，或雷达现在不能拿它比较（被更正撤回、没有已知领域），读取它的来源记录就是一次“核验读取”，响应的 `reason` 为 `verify_series_value`；不发 `series` 的调用方行为完全不变，`baseline` 忽略它。

## 来源、通道和决策记录

**来源和通道是两回事。** 来源是这次调查可以使用的资料（如 `who_don`、`gdelt_doc`）；通道只是找候选的办法：`fabric_literal`（Fabric 字面检索）、`weknora_semantic`（WeKnora 语义检索）、`gdelt_leads`（GDELT 线索清单）、`graph`（图谱查询）。允许某个通道，不等于允许它找到的任何资料。三个**可选**字段，都不写时行为与以前完全一样：

- 请求的 `sources`：这次调查的来源集合。写了以后，`source` 不在集合内的候选不会被选，得分 0，原因 `source_out_of_scope`。WeKnora 是通道，不再写进 `source`：`source` 写真实来源，`channel` 写 `weknora_semantic`。
- 候选和历史的 `channel`：候选是经哪条通道找到的。同一来源、同样的词，换一条通道是另一个动作；去重和后验都按（kind、source、channel）分开。候选和历史要一起带，只带一边，历史就认不出做过的动作。
- 请求的 `channels`：雷达对各通道本轮的说明，`{"channel": ..., "available": true|false, "reason": ...}`，每个通道最多一项。`available` 为 false 时必须写原因（未配置、能力不支持、额度用完、`03_not_run` 等）。声明不可用的通道，经它来的候选一律不会被选（`channel_unavailable`）。

**响应里的决策记录。** 雷达把请求和响应存下来，事后说明为什么做了这一步、为什么停。同样的请求（含 `history`）永远得到同样的响应，所以存下来就能重建当时的决定；请求怎么存、怎么哈希是雷达的事。在 `decision`、`candidate`、`reason`、`scores` 之外：

- `reasons`：每个候选一项，与 `scores` 同序。能选的写排名靠前的最强理由：`next_page`、`named_object`、`follows_up`、`critical_read`、`verify_series_value`、`open_required_read`（读与未答必答检查相关的记录）、`scored`（只凭分数）；baseline 写 `baseline_search`、`baseline_read`、`kind_not_used`。不能选的写原因：`source_out_of_scope`、`channel_unavailable`、`already_done`、`duplicate_in_list`、`unaffordable`。
- `channels`：请求里出现过的每个通道（声明过、有候选或做过）各一项，状态为 `offered`（有候选能选）、`chosen`（选中的候选走这条通道）、`not_needed`（它能提供的已经做过，或这个策略用不上）、`unavailable`（雷达声明不可用，或它的候选都做不了：超出来源集合、付不起），带原因。没有任何通道信息的老请求得到空列表。
- `checks`：每个检查项按 planner 实际的理解。`state` 为 `open` / `answered` / `conflict`，只有带证据的 `answered` 才算 `answered`；发来 `answered` 但没有证据的写 `open`，`reason` 为 `answered_without_evidence`。
- 所选动作是 `candidate`（连同 `reasons[candidate]` 和 `chosen` 的通道）；停止时 `reason` 就是停止原因。`planner_version` 是作出这个决定的版本。

例（同样的词经两条通道是两个动作；`nmpa` 不在来源集合内；图谱通道由雷达声明不可用）：

```json
{"question": "某地是否出现不明原因肺炎聚集？",
 "checks": [{"id": "Q1"}, {"id": "Q2", "status": "answered", "evidence": ["E1"]}],
 "sources": ["who_don", "gdelt_doc"],
 "channels": [{"channel": "weknora_semantic"}, {"channel": "graph", "available": false, "reason": "03_not_run"}],
 "candidates": [
   {"kind": "search", "source": "who_don", "channel": "fabric_literal", "target_checks": ["Q1"], "args": {"terms": ["pneumonia", "cluster"]}},
   {"kind": "search", "source": "who_don", "channel": "weknora_semantic", "target_checks": ["Q1"], "args": {"terms": ["pneumonia", "cluster"]}},
   {"kind": "search", "source": "nmpa", "channel": "fabric_literal", "target_checks": ["Q1"], "args": {"terms": ["pneumonia"]}}],
 "history": [{"kind": "search", "source": "who_don", "channel": "fabric_literal", "target_checks": ["Q1"], "args": {"terms": ["Cluster", "pneumonia"]}}],
 "budget": {"steps_left": 8, "reads_left": 15}}
```

```json
{"decision": "continue", "candidate": 1, "reason": "best_score", "scores": [0.0, 0.484848, 0.0],
 "reasons": ["already_done", "scored", "source_out_of_scope"],
 "channels": [{"channel": "fabric_literal", "status": "not_needed", "reason": "already_done"},
              {"channel": "weknora_semantic", "status": "chosen", "reason": "scored"},
              {"channel": "graph", "status": "unavailable", "reason": "03_not_run"}],
 "checks": [{"id": "Q1", "state": "open", "required": true, "reason": ""},
            {"id": "Q2", "state": "answered", "required": true, "reason": ""}],
 "planner_version": "0.2.0"}
```

## 状态回流

planner 只读检查项，不写。雷达把读阶段和 05、06、07 的状态折成 `checks[].status` 发来：

| 上游状态 | `status` |
|---|---|
| 读阶段有原文依据 | `answered`，带 `evidence`（E#）。意思是“有依据的候选回答”，不是核验通过 |
| 05：可比较的冲突 | `conflict` |
| 05：open、insufficient、不可比较 | `open` |
| 06：追问或判断 E 中上一轮待核的命题 | 对应检查项 `open` |
| 07：机制缺口 | 机制检查项 `open` |

所以上游把一个读阶段已有依据的检查项退回 `open` 或 `conflict`，planner 就不会 `complete`：有预算和候选时继续找，没有就 `insufficient`，响应的 `checks` 列出没解决的。`complete` 只表示必答检查都有带证据的 `answered`、可以停止调查，不等于报告核验通过，发布由 06 决定。

## 打分和停止规则（大白话）

**什么叫“答完了”**：必答检查的状态是 `answered` 且 `evidence` 不为空。只写 `answered` 没有证据，按未答处理。`conflict` 也不算答完。

**打分**：对每个能做的候选，把它要查的检查逐个算 `重要性 × 后验增益 × impact × scope_match`，加起来，再除以 `1 + 0.25 × 成本压力`。已答的检查在还有必答检查没答完时按 0.25 计（一篇记录常常只答了一部分，比如三家公司里的一家），全部必答答完后按 0 计。

- 后验增益：同一类动作（kind + source + channel，比如“pubmed 的搜索”）过去的 `gain` 与先验 0.5 的平均，先验相当于 2 次观察。做过几次都没收获，这类动作的分数就降下来；失败的算 0。
- 成本压力：`cost / steps_left`；读取还要和 `1 / reads_left` 比，取大的。越接近用完额度越贵。
- 去重：kind、source、channel、args 都相同就是同一个动作（同样的词换一条通道是另一个动作）。大小写、多余空格、`terms` 的顺序不算区别。做过的和列表里重复出现的（只保留第一个）都不再考虑。
- 保底分：三类“明摆着没做完的事”不会因为估计增益为 0 就被跳过——翻页（`next_page`）、点名对象的查询（`names_object`）、读取与未答必答检查相关的记录或 `critical` 记录。它们的分数至少是 `0.25 × scope_match / (1 + 成本压力)`，并且排序时先于普通动作：先翻页，再点名对象，再关键记录读取，最后才按分数。

**停止规则，按这个顺序检查，先满足的先生效**：

1. 历史里有 `pending` 的动作 → `insufficient`（先等雷达确认上一步）。
2. 必答检查都答完，且没有“还能做的”翻页、点名对象、关键记录读取 → `complete`。（没有任何必答检查时不会 complete。）
3. `steps_left` 为 0 → `insufficient`（`step_limit`）。
4. 没有候选（含全被去重、超出 `sources`、通道声明不可用）→ `insufficient`（`no_candidates`）。
5. 有候选但都付不起（成本超过剩余步数，或读取而 `reads_left` 为 0）→ `insufficient`（`nothing_affordable`）。
6. 最高分 ≤ 0 → `insufficient`（`no_expected_gain`）。
7. 否则 `continue`，选排序第一的。

## baseline

请求里写 `"policy": "baseline"`。固定顺序，不看分数：先把每个不同的 `search` 候选各做一次（按给定顺序），再按给定顺序做 `read` 候选；其它类型不用。`reads_left` 为 0、`steps_left` 为 0 或没有可做的就停，必答检查都有证据则 `complete`，否则 `insufficient`。用来和 planner 对比。

## 代码

- `planner/models.py`：请求和响应（pydantic）。
- `planner/policy.py`：打分、去重、停止规则。
- `planner/baseline.py`：固定顺序的 baseline。
- `planner/record.py`：响应里的决策记录（候选理由、通道状态、检查项状态），planner 和 baseline 共用。
- `planner/service.py`：FastAPI、令牌、统一错误、`/health`。

## 开发环境与测试

唯一的入口是 `Makefile`。需要 uv 0.13.0（在 PATH 上，例如 `~/.local/bin/uv`；或者 `make UV=<路径>`）。

```sh
make venv            # .venv：只按 requirements.lock 安装，每个包都校验 sha256
make test            # 全部测试（不联网）
make lint            # ruff（pyproject.toml）
make metrics         # 函数长度 80 行 / 圈复杂度 20 的门槛（scripts/code_metrics.py；例外清单 scripts/metrics_exceptions.txt 只许变短）
make check-install   # 一个用完即删的新环境：按锁文件安装、逐个导入模块、再跑一遍测试
make check-lock      # requirements.lock 是否仍与 requirements-dev.txt 一致（要访问包索引）
```

`make lint` 的规则：所有仓库共用一份基础规则 `E4, E7, E9, F, W, I`（devtools `README.md` 的“共用的代码门槛”）。圈复杂度只由 `make metrics` 管，不开 ruff 的 C90。在基础规则之上多开规则的仓库：

| 仓库 | 多开的规则 |
|---|---|
| BioRadar | `E`（含行宽 E501，110） |
| BioCheck | `E`（含行宽 E501，120） |

`make check-lock`、`make lock` 要访问 PyPI（`make venv`、`make check-install` 在 uv 缓存里没有的包也要下载）。zktest 上直连 PyPI 有时超时，这时只给这几条命令临时加上本机已有的 HTTP 代理（mihomo，监听 127.0.0.1:7890），例如 `HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 make check-lock`；不要把代理写进全局配置。`check-lock` 的结果有两种失败：解析本身失败（包索引或网络不可用、依赖解不出来）时报 `check-lock did not run`，make 显示 `Error 2`，锁文件没有比较过，不能因此重做锁；只有解析成功而内容不同时才报 `requirements.lock is stale`（make 显示 `Error 1`）。两种情况 make 都以非零状态结束。

- 直接依赖写在 `requirements.txt`（运行）和 `requirements-dev.txt`（测试，含前者）；`requirements.lock` 是由它解析出的完整锁（运行 + 测试），每个包都带 sha256，解析目标是 CPython 3.12、x86_64-unknown-linux-gnu（锁文件头一行写着解析命令）。改了依赖后运行 `make lock`，再提交锁文件。
- 已验证的组合：uv 0.13.0、CPython 3.12.15；通过数量以 `make check` 的输出为准，这里不写数字。

不联网。覆盖：停止规则顺序、打分次序、去重、翻页优先、“答完但没有证据”被降级、baseline 顺序、决策记录、来源与通道分开、上游状态退回检查项、老请求不变，以及令牌和错误格式。

## 设计来由

这个服务是从旧演示整个重写出来的，删了什么、留了什么，见 [docs/decisions/0001-rewrite-as-small-service.md](docs/decisions/0001-rewrite-as-small-service.md)。来源与通道分开、响应里带决策记录的理由见 [docs/decisions/0002-source-channel-and-decision-record.md](docs/decisions/0002-source-channel-and-decision-record.md)。
