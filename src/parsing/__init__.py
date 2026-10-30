"""Step 1 解析阶段：把用户原始问题，变成一份"可执行的证据需求"。

    用户问题
       │
       ├─① 语义改写 ──► IntentDraft
       │    LLM（主路径）或规则（兜底）把整句自然语言归一化：
       │    抽出实体提及、关系提示、答案变量、问题模板。
       │    全程只做一次表层规范化。
       │
       ├─② 图谱链接 ──► QueryIntent
       │    用加载期就建好的词表（精确 → 别名倒排 → 模糊匹配）
       │    把实体提及映射成 KG 规范名，关系提示归一成规范关系名，
       │    并判定链接质量：全部命中 / 部分命中 / 起点未命中。
       │
       └─③ 证据需求 ──► EvidenceSpec
            按问题模板查表，得到"这类问题要检查哪些项"，
            并把关系约束按强弱分成硬约束和软约束。

    输出 ParsingResult = 意图 + 证据需求 + 诊断信息

技术分工：只有 ① 的 LLM 改写用模型，其余全是传统算法——
规范化是字符串规则，链接是倒排索引 + 模糊匹配，模板映射是查表。

注意两点现状：
  * 解析层的输入只有用户问题；知识图谱提供的是受控词表（资源），
    不是被处理的输入数据。解析层不读取任何 triples / passages。
  * 目前默认配置走的是另一条解析链路，EvidenceSpec.checks 也还没有
    被第三阶段（终止判断）读取——这一段的产出现在只声明、不消费。

核心契约见 ..common.contracts：

    ParsingResult {
        intent: QueryIntent {
            canonical_question 改写后的规范问题
            template_id        问题模板（问题分类）
            start_entities     链接后的起始实体（KG 规范名）
            target_entities    链接后的终止实体（KG 规范名）
            answer_slots       未知答案变量，如 ?drug
            relation_constraints 关系约束（方向 + required/preferred）
            reasoning_mode     推理模式
            unlinked_mentions  未能链接到图谱的提及
            parse_status       linked | partial | unlinked
        }
        evidence_spec: EvidenceSpec {
            shape            需要什么形状的证据：path/subgraph/relation_set/single_fact/count
            hard_constraints 必须满足的边
            soft_constraints 期望满足的边
            checks           机器可验证的检查项
            answer_slots     答案槽
        }
        diagnostics
    }
"""
