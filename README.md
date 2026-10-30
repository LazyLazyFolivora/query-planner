# query-planner（D 课题 demo：全流程策略化）

预算约束查询规划的循环骨架。**主流程的每一步都是一个可替换的策略**，默认全部走模型实现，规则实现作为基线保留。

配套文档：
- `docs/D_步骤×可参考项目_对照表.md`（每步去哪个项目抄什么）
- `docs/D_相关论文×主流程步骤_四分类对照矩阵.md`（每步的传统/小模型/大模型先例）
- `docs/D_系统心智模型_查询循环.md`（循环视角的心智模型）
- `docs/D_状态编码重做_v2_变量定义.md`（状态变量 v2：保留、删除与维度语义）
- `docs/D_三态停止实现记录.md`（③ 分层三态停止、可负担动作数与训练接口）
- `docs/D_深度学习策略模型实现.md`（BGE-small / PubMedBERT + MLP 停止与动作策略）

---

## Radar 逐步调用接口

`radar_policy.py:decide(request)` 是纯 Python 策略入口，默认仅依赖
`requirements-policy.txt` 中的 NumPy；不会读取密钥、访问网络、下载嵌入模型或执行候选动作。
可通过 `importlib.util.spec_from_file_location` 按文件加载，不会占用宿主的 `src` 包名。
输入/输出版本及完整样例在 [`contracts/radar-policy-v1.json`](contracts/radar-policy-v1.json)。

宿主每步提供真实剩余预算、合法候选、已有证据 ID、必要取证检查及每次尝试的最新回执。
接口复用 `HeuristicSelector.select` 和 `RuleStop.decide`：候选 `covers` 与未通过检查的交集
进入既有 18 维动作特征，实际预计成本进入成本特征；已通过且引用现有证据的检查进入
`observable_sufficiency_status`。没有构造假三元组或拿候选覆盖意图充当已取得证据。
反馈更新后重新选择一项，`complete` 仅表示宿主给定的必要取证检查全部通过，不表示医学结论成立。
缺少必要检查不能 `complete`。宿主负责权限、检索/图查询、付费模型、硬预算、持久化和恢复。

`document_search` 每项仅代表一个 Fabric 或 WeKnora provider 的一次查询；
`entity_relations` 代表已落地实体的关系种类查询，`graph_neighbors` 为邻居展开，
`graph_path` 为路径查询。后者是否用 Cypher 由宿主执行器决定，本接口不连接 Neo4j。
候选的 `covers` 必须指向声明的检查；预计成本未知且该维度有限额时排除该候选。
预算省略维度只表示策略不限制该维度，宿主仍保留硬保护。未知成本不会在返回候选中改成零。

`prepared/sent/unknown/unconfirmed` 回执会阻止新动作，等待宿主对账；
`completed/failed/not_sent` 都是已尝试动作，本接口不自动重试。
历史每个 action ID 只传最新回执；按动作种类和参数语义去重，因此换 ID 不能绕过已发送回执。
候选按 ID 稳定排序，同一快照恢复会得到同一结果。宿主在真正派发前须先持久化选择和回执状态。

`create_policy(selector="linear"|"model", stop="rule"|"model")` 可显式复用已有本地模型头，
其中 `model` 需要调用方另装 PyTorch。外部状态编码为 `external_observations_v1`（12 维），
动作特征也有外部语义；旧图谱状态 checkpoint 不可直接混用。没有随附训练权重，默认结果
明确标记 `heuristic_untrained`；接通链路不等于已验证学习方法。

工厂离线入口 `load_public_questions` 只读取
`release/public/planner_questions.jsonl` 的 `task-construction-v2`，保留环境标识和预算，
不读取隐藏目标/manifest，不接受旧出题探索轨迹，`requirement_groups` 始终为空。
`scripts/run_public_questions.py` 使用现有 rule/feature/heuristic pipeline 重新执行检索，输出独立
solver transitions、实际成本、输入及世界文件 SHA256。必须显式绑定 `--environment-ref`：

```bash
python scripts/run_public_questions.py --questions release/public/planner_questions.jsonl \
  --backend primekg --world public-world.sqlite --environment-ref YOUR_PUBLIC_ENVIRONMENT \
  --out fresh-solver-trajectories.jsonl
```

这个离线 runner 仅支持兼容 PrimeKG `edges` 表的 SQLite，以及原有 ToyKG JSON。
要求显式 `relation_access/tool_calls` 预算；ToyKG 文段检索另需 `tokens`。
PrimeKG 仅执行本地图操作，无模型 token 开销。`ctd_access/pubmed_access/primekg_access`
等未映射计量返回 `unsupported_budget`，不会擅自换单位；没有声称接通这些后端。
停止后的隐藏目标评测应由独立评测进程进行。

离线验证：`python -m unittest discover -s tests -v`。

## KDD 多方向路由（external.v2）

`radar_policy.decide(request)` 按 `schema_version` 分派；旧 v1、原图谱 demo 和 Radar
原 baseline 行为保留。v2 的正式请求 schema 与可直接运行的样例分别在
[`radar-policy-v2.schema.json`](contracts/radar-policy-v2.schema.json) 和
[`radar-policy-v2.json`](contracts/radar-policy-v2.json)。

v2 在原四种取证/图动作之外支持宿主注册的 `dataset_prepare`、`analysis_run`、
`professional_assess`。新动作须声明 capability ID/版本、方向和时间范围、输入产物
ID/版本、资格与预计费用；当前输入不齐、范围不符或能力未注册时排除，返回具体原因。
`capabilities.input_requirements` 是所需 `artifact.kind` 列表。缺数据时，宿主可提供
补查候选或范围明确的替代分析；商业、公卫不会被策略强制发送到 MechEvidence。

默认 `information_gain_v1` 用尚未满足检查的重要性、预计信息贡献、范围匹配以及同类
动作已完成回执的观测收益排序，每步重新决策。回执更新使用固定强度 2 的未训练先验；
成本用于可执行性筛选和较小的效用折扣。返回每项候选的评分组成及未满足缺口。
`observed_information` 只有已完成且带当前证据/产物引用时可形成正收益；没有收益测量
不会补成零。期望信息和 `covers` 永远不能将检查标记通过。

`decide_kdd(request, selector="coverage_cost_baseline_v1")` 提供单独命名的同候选空间
覆盖/成本对照，不冒充原 baseline。两种方法都保持 `heuristic_untrained`，没有下载模型
或训练权重。方法版本 `kdd-information-gain-1.1` 与代码内容 hash 同时返回。
停止方法如实记录为 `observed_checks_reading_v2`：先处理待对账回执，再检查真实观察覆盖及未读材料、步数、可执行候选和信息收益；它不是学习得到的停止模型。
所有执行、权限、许可、硬预算、账本和恢复由宿主负责；不确定回执总是先对账。
`complete` 仅指宿主声明的观察检查已满足，科学有效性、业务价值与人工采用分别评估。

### 可见材料补读与主动补采（2026-10-31）

完整请求见 [`radar-policy-v2-reading-fetch.example.json`](contracts/radar-policy-v2-reading-fetch.example.json)。
该示例是合成契约资料，没有隐藏答案、凭据或实际执行回执。新增能力不改变 `external.v1` 或原 baseline。

`reading_manifest` 只描述当前权限与时点内可见的 `(source_id, source_record_id, version_id)`，
状态为 `read / unread / unavailable`。宿主从已公开更正关联、当前分析缺口或声明研究范围构造它；
不得读取评测 oracle 或把人工必读答案塞入。`critical` 必须给 `basis.kind/reference_ids`：
`visible_source_relation` 引用当前 evidence/manifest record，`observed_gap` 引用声明 check，
`declared_scope` 引用 check 或 `question_id`。策略校验引用存在，来源关系的真实性由宿主验证。

`document_read` 是一次 Fabric 精确版本批量请求（1–32 个 records，预计 `tool_calls=1`）。
参数严格为 `provider/source_id/records/scope`；record 必有 `source_record_id/version_id`，可带 `window_start`（Unicode 整数偏移 >=0）。
候选必须在 manifest 的同版本/同窗口中且仍为 unread、适用性 eligible、方向/时点/mode/is_demo 与请求一致。
合法 critical 补读优先于先验分数；普通未读点查得到显式 `visible_reading_floor`，不是伪造的观测收益。
可选窗口只能由宿主从已保存 end/真实 Fabric locator 产生，模型不能编坐标；省略与 0 为同一首窗，后续窗口参与去重。
`unread_window_count` 与 `unread_version_count` 分开统计，续读片段不得算新研究。
未读或 unavailable 项保留 `unmet_reading`，阻止仅凭已有检查完成就返回 complete；
预算不足、达到步数或没有合法动作仍如实不足，由宿主生成带限制的可核草稿。

`source_fetch` 使用现有 capability 形状：注册 ID/版本、input_refs、scope、parameters，
例如 Fabric 已开放的受支持 source fetch。它不是任意 URL 请求或隐藏 Agent。
宿主负责服务许可、参数白名单、固定操作 ID、HTTP/后台任务回执与采用结果。
异步提交不等于材料已读；prepared/sent/unknown/unconfirmed 必须先对账，不能换 ID 再发。
临时 WeKnora 与图谱继续使用现有动作；不合法 demo/权限/版本候选不能变成 eligible。

可选 `action_limits` 给每种动作独立的最大尝试数（包括 failed/not_sent），避免一种动作耗尽
所有步骤；宿主的 max_steps、真实统一预算继续优先。这里不提高任何预算或默认常量。
只有候选 covers/expected_information 及回执 check_updates/observed_information 中的未知
coverage ID 降为 `coverage_issues` 与 warning；原动作/回执不改，未知标签没有奖励。
重复 checks/updates、非法工具、错误结构与 scope/version gate 不会被吞掉。

[`business-checks-v1.json`](contracts/business-checks-v1.json) 提供可选四方向词表；宿主按问题适用性选取，
不强制每题执行所有方法。公卫风险判断与传播判断独立，保留假设支持、反证和来源独立性；
传言、否认或删帖本身不能当结论。商业区分订单、更正、可比期间、收入确认、现金与利用率。

## 一、七个步骤 × 七个策略接口

```
①  解析      QuestionParser   问题文本 -> 证据需求（种子/目标/必需关系/算子）
②  状态编码  StateEncoder     PolicyState -> 定长向量
③  终止判断  StopRule         三态：continue / complete / insufficient
④  动作选择  ActionSelector   在合法动作里按 效用 − λ·成本 选一个
⑤  执行      ActionExecutor   动作 -> 证据 + 真实成本（工具层）
⑥  记账      BudgetLedger     多维成本记账（关系访问数/调用/token）
⑦  更新证据  EvidenceUpdater  证据入列 + 查询历史维护
```

每一步都有**模型实现（默认）**与**规则实现（基线）**：

| 步骤 | 模型实现（默认） | 规则实现（基线） | 其他 |
|---|---|---|---|
| ① 解析 | `model` 词典召回 + 嵌入消歧 + 滑窗模糊 + 嵌入原型算子分类 | `rule` 纯词典/别名匹配 | `hybrid` PrimeKG 匹配 + LLM 补齐 |
| ② 编码 | `model` BGE-small-zh/PubMedBERT + 问题/证据投影 + 42 维手工特征 | `feature` **五块 42 维**手工归一化特征 | |
| ③ 终止 | `model` **分层三态停止**：`p_stop` + `p_complete|stop` + 硬规则兜底 | `rule` 充分性检查 | `flat` 扁平三分类基线；`naive_target` 过早停止基线 |
| ④ 选择 | `model` 状态-动作联合 MLP：效用头 − λ(剩余预算)·成本预测头（λ 自适应） | `heuristic` 缺口覆盖+关系先验+成本惩罚 | `linear` 线性策略基线 |
| ⑤ 执行 | — | `toy_kg` 玩具 KG（4 个动作 + 合法动作空间） | |
| ⑥ 记账 | — | `budget` 多维预算账本 | |
| ⑦ 更新 | — | `standard` 标准更新器 | |

**关键设计**：
- `runner.py` 只依赖策略接口，不含任何具体算法；
- 每步的实现通过注册表按名字构建，新增实现不需要改 runner 与 CLI；
- 分层停止头和动作选择器使用 PyTorch MLP 残差头；输出头零初始化，未训练时以启发式初始化为准，训练后生效。

---

## 二、目录结构

```
projects/query-planner/
├── run_demo.py              入口（可逐步指定实现 + 消融）
├── requirements.txt         依赖（numpy）
├── data/
│   ├── toy_kg.json          玩具医学 KG：29 实体 / 32 三元组 / 5 文段
│   └── questions.json       4 个问题（含证据需求组与预算）
└── src/
    ├── schema.py            数据模型
    ├── constants.py         领域常量与关系先验
    ├── kg.py                玩具 KG 后端（被 executor 包裹）
    ├── state.py             PolicyState + 可观测充分性信号
    ├── evaluator.py         评测（唯一能看 gold 的地方）
    ├── runner.py            调度器：只依赖策略接口
    └── strategies/          ★ 策略模式的核心
        ├── base.py          7 个 Protocol + EvidenceStore
        ├── embedder.py      SentenceEmbedder（BGE-small-zh 默认）/ HashEmbedder（消融）
        ├── llm.py           LLMClient 协议 / NullLLM（默认关闭）/ OpenAICompatClient（预留）
        ├── parser.py        ① RuleParser / ModelParser / HybridParser（PrimeKG 匹配 + LLM 补齐）
        ├── encoder.py       ② FeatureEncoder / ModelEncoder（BGE-small-zh / PubMedBERT）
        ├── nets.py          PyTorch MLP trunk / zero-init heads
        ├── stop.py          ③ RuleStop / HierarchicalModelStop / FlatModelStop / NaiveTargetStop
        ├── selector.py      ④ HeuristicSelector / LinearModelSelector / ModelSelector
        ├── executor.py      ⑤ ToyKGExecutor
        ├── ledger.py        ⑥ BudgetLedger
        ├── updater.py       ⑦ StandardUpdater
        └── __init__.py      注册表 + Pipeline + build_pipeline
```

---

## 三、环境要求

| 项目 | 值 |
|---|---|
| Python | **>= 3.10**（assumption_env：3.10.21） |
| 依赖 | `numpy>=1.24`、`torch==2.6.0+cu118`（CUDA 11.8 GPU 版）、`transformers>=4.40`、`sentence-transformers>=2.7` |
| 默认语义编码器 | `BAAI/bge-small-zh-v1.5`；可用环境变量 `QRAG_EMBED_MODEL` 覆盖 |
| 其他 | 首次运行会下载 HuggingFace 模型；纯规则消融不需要 PyTorch |

```bash
python -m venv .venv
.venv\Scripts\activate                     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py --quiet
```

> ⚠️ **注意解释器**：本机 `python` 当前解析到 `D:\Anaconda\python.exe`（**Python 3.7**），低于要求。
> `run_demo.py` 开头有版本守卫会给出明确提示。请显式使用 3.10+ 的解释器，例如：
> ```bash
> C:\Python312\python.exe run_demo.py --quiet
> ```

---

## 四、运行与消融

```bash
# 默认：全模型
python run_demo.py --quiet

# 全规则基线
python run_demo.py --preset rule --quiet

# 逐步替换（每一步都能单独指定）
python run_demo.py --parser rule --quiet
python run_demo.py --encoder feature --quiet
python run_demo.py --stop-rules rule,flat,model,naive_target --quiet  # ③ 消融
python run_demo.py --selectors heuristic,linear,model --quiet       # ④ 消融
```

---

## 五、当前结果

2026-10-30 验证的**全规则基线**：`python run_demo.py --preset rule --quiet`。
当前分支与原始提交 `6ef6453` 的四题输出逐字一致：

| 问题 | 决策 | 停止原因 | ra | 调用 | 步 | gold | 判定 |
|---|---|---|---|---|---|---|---|
| q1 二甲双胍机制（可达） | complete | sufficient_evidence | 9 | 7 | 8 | 0.75 | 错误完成 |
| q2 华法林相互作用 | complete | sufficient_evidence | 3 | 1 | 2 | 1.00 | 正确完成 |
| q3 致癌机制（图谱中不存在） | insufficient | no_legal_action | 22 | 22 | 23 | 0.00 | 正确不足 |
| q4 布洛芬（预算 ra=8） | insufficient | budget_exhausted | 8 | 3 | 4 | 0.25 | 正确不足 |

q1 暴露了旧 demo 的不足：可观测取证检查已通过，但独立 gold 证据组仅覆盖 0.75，
仍有一次错误完成。该问题在基线已存在，当前接入没有修改旧图谱充分性算法。
因此运行结束与接口接通不能证明证据充分性或学习方法有效。

默认全模型与终止规则消融尚未针对当前版本重新验证，需固定权重、配置与环境后另行评测。

---

## 六、模型实现里哪些是"占位"、怎么换

| 组件 | 现状 | 替换方式 |
|---|---|---|
| `HashEmbedder` | 字符 n-gram 哈希（无语义，仅作消融） | 默认改为 `SentenceEmbedder("BAAI/bge-small-zh-v1.5")`；英文 BioHopR 可用 `pritamdeka/S-PubMedBert-MS-MARCO` |
| 模型头权重 | MLP 停止头 + MLP 策略头：启发式初始化 + 零初始化学习残差 | `HierarchicalModelStop.fit(...)` / `ModelSelector.fit(...)` 先做行为克隆；后续换 PPO/GRPO |
| `ModelParser` 的算子分类 | 嵌入原型 + 关键词兜底 | 接真实编码器后自动提升 |
| 成本预测头 | 恒等映射到预计成本（玩具环境成本已知且精确） | 接真实 KG 后改为分位数回归（P50/P90） |
| 动作集 | 4 个（实体/关系/路径/文段） | executor 换真实工具即可，上层不动 |

---

## 七、下一步

1. `executor` 换成真实 KG（KG-R1 的 `kg_retrieval_server` 接口），验证"关系访问数"口径；
2. 用规则策略批量跑 BioHopR，产出第一批轨迹；
3. `HierarchicalModelStop.fit` / `ModelSelector.fit` 用这些轨迹做行为克隆（监督），再对比未训练版本；
4. `evaluator` 补预算超限率、充分后追加成本、停止校准 ECE。
---

## 八、解析层：PrimeKG 匹配 + LLM 补齐（`HybridParser`）

**不引入任何外部词表**，解析只用两层：

```
① PrimeKG 模式匹配（词典最长匹配 + 别名 + 滑窗模糊）
      命中  →  直接采用（零成本、可复现）
      未命中 ↓
② LLM 补齐（仅在存在"可疑未链接片段"时触发）
      · 先按字面相似度给 LLM 一个 8 条候选（bounded reranking）
      · LLM 只允许从候选中选一个，或回答 NONE
      · 产出必须【再过一次 PrimeKG 匹配】才被采信
      命中  →  补入 linked
      未命中 →  记入 unlinked_mentions
③ 输出三态：linked / partial / unlinked
```

**为什么必须"回图校验"**：LLM 若生成图中不存在的实体名，后续关系访问成本无法计量，
所以 `HybridParser._resolve_with_llm()` 把 LLM 的输出**再喂一次 `_best_primekg_match()`**，
落不到图上就明确记为 `unlinked`。

**unlinked 的后果**：若起点实体全部 unlinked，`seeds` 为空 → `legal_actions()` 返回空 →
策略直接得到 `Selection(None, "no_legal_action")` → **判 insufficient**，不会浪费预算乱查。

### 启用方式

默认仍是 `model`（不依赖 LLM）。要启用 hybrid，需在代码中注入一个 LLM 客户端：

```python
from src.strategies import build_pipeline
from src.strategies.llm import OpenAICompatClient

llm = OpenAICompatClient(base_url="http://127.0.0.1:8000/v1", model="Qwen/Qwen2.5-7B-Instruct")
pipe = build_pipeline(kg, parser="hybrid", llm=llm)
```

API key 从环境变量 `OPENAI_API_KEY` 读取，**不硬编码**。
`NullLLM` 是默认实现——未注入时，hybrid 的行为等价于 `model`（LLM 那一步永远返回 None）。
---


## 九、状态编码（②）：五块 42 维

状态 `s_t` 的五块划分对应 CLAUSE 的 `s_t=(q,G_t,F_t,P_t,b_t)`。这里区分两件事：

- **状态变量**：语义上讨论的一个量，例如“未覆盖关系组”；
- **向量维度**：这个量如何编码，例如 6 个关系组用 6 维 multi-hot。

| 块 | 索引 | 维度 | 回答的问题 |
|---|---|---|---|
| **① 问题块** | 0–15 | **16** | 我在解什么题、要覆盖什么 |
| **② 证据块** | 16–18 | **3** | 已拿到多少结构化/文本证据 |
| **③ 缺口块** | 19–33 | **15** | 还差什么、是否已无路可走 |
| **④ 历史块** | 34–38 | **5** | 最近做过什么、是否卡住 |
| **⑤ 预算块** | 39–41 | **3** | 还剩多少预算 |

### ① 问题块（16 维）

| 索引 | 特征 | 编码 | 维度 |
|---|---|---|---|
| 0–4 | 推理模式（mechanism / chain / intersection / counting / aggregation） | one-hot（互斥） | 5 |
| 5–10 | **必需关系组**（causal / property / interaction / adverse / associative / other） | **multi-hot（集合）** | 6 |
| 11–14 | 目标实体类型（drug / disease / effect / gene） | multi-hot（集合） | 4 |
| 15 | 有无否定约束 | 0/1 | 1 |

**为什么合并推理算子和问题类型**：二者在旧版里会重复编码“是不是机制型问题”。现在只保留一个互斥的 `reasoning_mode`：机制型优先，其余使用解析层的算子分类。

**为什么目标实体类型保留、起点实体类型删除**：目标类型直接参与“答案集合是否完整”的判断；起点类型只是当前节点属性，执行器和候选动作已经能看到它，单独进状态对停止决策没有独立价值。

### ② 证据块（3 维）

| 索引 | 特征 | 说明 |
|---|---|---|
| 16 | `n_triples_norm` | 结构化证据条数 |
| 17 | `n_passages_norm` | 文本证据条数；未启用文本检索时为 0 |
| 18 | `max_causal_chain_norm` | 最长因果链；非机制型问题恒为 0 |

这里的计数都做 log 归一化，不编码“证据密度”“唯一实体数”等动态分母量。

### ③ 缺口块（15 维）

**四项检查状态（8 维）**

| 索引 | 检查 | 编码 |
|---|---|---|
| 19–20 | seed→target 是否存在证据路径 | `applicable, passed` |
| 21–22 | 必需关系组是否全部覆盖 | `applicable, passed` |
| 23–24 | 是否存在 ≥2 跳因果链 | `applicable, passed` |
| 25–26 | 因果链是否有文本证据支撑 | `applicable, passed` |

每项检查用两个 bit：

```
not_applicable -> (0, 0)
failed         -> (1, 0)
passed         -> (1, 1)
```

这样非机制型问题的因果链检查不会被误当成 failed。

**未覆盖关系组（6 维）**

| 索引 | 特征 |
|---|---|
| 27–32 | `uncovered_relgroup_*`：必需关系组中尚未被证据覆盖的组 |

这是比“覆盖率标量”更有信息量的表示：模型直接知道“还缺哪一类关系”，而不是只知道缺了 2/5。

**可负担动作数（1 维）**

| 索引 | 特征 |
|---|---|
| 33 | `affordable_action_count_norm`：当前合法、尚未尝试且剩余预算足以执行的动作数 |

它让停止头能识别“证据不全，但已经没有动作可做”的死路状态。非法动作和合法但付不起的动作要区分开，否则停止头会把“有动作但做不起”误判成还可以继续。

### ④ 历史块（5 维）

| 索引 | 特征 | 说明 |
|---|---|---|
| 34–37 | 最近 3 步动作类型频率 | entity_lookup / relation_expand / bounded_path / passage_retrieve |
| 38 | `no_progress_streak_norm` | 连续无新增证据步数 |

这里不再存“实体访问覆盖率”“关系扩展覆盖率”“路径尝试比例”。这些量的分母会随图扩展而变，数值不单调，也没有稳定的决策语义。动作历史使用最近 3 步，不再出现“状态里存了 3 个动作但只使用 1 个”的不一致。

### ⑤ 预算块（3 维）

| 索引 | 特征 | 说明 |
|---|---|---|
| 39 | `rem_relation_access_ratio` | 剩余关系访问预算比例 |
| 40 | `rem_tool_calls_ratio` | 剩余工具调用预算比例 |
| 41 | `rem_tokens_ratio` | 剩余 token 预算比例 |

不编码消耗速率、预计剩余步数、预算压力、步数进度：前三个是早期不稳的派生量或自造指标，最后一个与剩余预算重复。

### 被删除的变量

以下变量不再进入状态向量；其中有些仍会保存在证据元数据或运行日志里，只是不喂给策略模型：

- 必需关系组数：由关系组 multi-hot 推出；
- 起点实体类型 / 起点实体数 / 目标实体数：不改变停止或动作选择；
- 必需关系组覆盖率：由未覆盖关系组推出；
- 证据密度、来源多样性、高置信占比：分母主观或当前数据不适用；
- 充分性总分、失败比例：由四项检查状态推出；
- 实体访问覆盖率、关系扩展覆盖率、路径尝试比例：动态分母，非单调；
- 消耗速率、预计剩余步数、预算压力、步数进度：派生量、早期不稳或没有稳定语义。

如果后续接入多源检索、校准置信度或大规模路径搜索，再以**可消融的附加特征**形式加回，而不是默认塞回状态。

### 与规则版的一致性

`ModelEncoder` = `SentenceEmbedder(BGE-small-zh/PubMedBERT)` ⊕ `tanh(W·问题嵌入)` ⊕ `证据池化向量` ⊕ `FeatureEncoder(state)`，

手工特征部分与规则版**完全一致**，差异只在语义部分——这样 `--encoder feature|model`
的消融可以干净地归因到语义嵌入。

维度：`16(问题投影) + 16(证据池化) + 42(手工) = 74`。

### 训练接口

- `ModelEncoder.fit(embeddings, targets)` —— 更新问题投影矩阵 `W`；实际语言模型编码器用 `SentenceEmbedder`。
- `HierarchicalModelStop`、`ModelSelector` 使用 PyTorch MLP 残差头；未训练时输出头零初始化，
  **初始为 0**：未训练时行为等价于启发式，训练后才生效。
