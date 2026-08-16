请基于工具结果输出最终 Agent JSON。

## 输出要求

1. 只输出一个 JSON 对象，不要输出任何其他文字。
2. 禁止 Markdown，禁止代码块（不要使用 ```），禁止列表，禁止解释性文字。
3. 不要再调用任何工具。
4. 文本字段长度限制：
   - possible_cause 不超过 40 个字符；
   - evidence_summary 每项不超过 40 个字符，最多 3 项；
   - reasoning_summary 不超过 100 个字符；
   - recommended_action 不超过 60 个字符。
5. 必须包含字段：event_id、agent_assessment、risk_level、agreement_with_rule、
   possible_cause、evidence_summary、reasoning_summary、recommended_action、
   uncertainty、tools_used、agent_confidence。
5.1 memory_used / similar_event_count / memory_summary 为可选字段：
   如果本轮调用了 get_similar_events，必须在输出中如实反映；
   未调用时保持 memory_used=false、similar_event_count=0、memory_summary=""。
6. 字段取值必须精确（不要缩写、不要改动）：
   - agent_assessment：必须是 NORMAL / SUSPICIOUS / CONFIRMED_RISK / UNCERTAIN
     之一，原样输出，禁止任何变体；
   - risk_level：必须是 low / medium / high 之一；
   - uncertainty：必须是 low / medium / high 之一；
   - agreement_with_rule：必须是 true 或 false；
   - agent_confidence：0.0 ~ 1.0 之间的数字。
7. agent_assessment 与事件状态（status）无关：不要把 status=confirmed 当作
   agent_assessment 取值；CONFIRMED_RISK 必须完整输出，禁止缩写成 CONFIRMED，
   也禁止输出 CONFIRMED_RISK_RISK、confirmed_risk、Confirmed_Risk 等变体。

## 目标 JSON 示例（紧凑形式）

{"event_id":"FE-001","agent_assessment":"SUSPICIOUS","risk_level":"medium","agreement_with_rule":true,"possible_cause":"疑似非火灾烟雾","evidence_summary":["持续烟雾无火焰"],"reasoning_summary":"烟雾特征明显但火焰证据不足，历史相似事件以短时烟雾为主","recommended_action":"继续观察并人工核查","uncertainty":"medium","tools_used":["get_event_summary","get_similar_events"],"agent_confidence":0.78,"memory_used":true,"similar_event_count":3,"memory_summary":"检索到3个相似历史事件，其中2个为短时烟雾/烟花场景"}

只输出 JSON，不要输出示例之外的任何内容。
