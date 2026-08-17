# 测试说明

## 自动化测试

```bash
python -m unittest discover -s tests -p "test*.py"
```

当前基线：**186/186 通过**（全部 Mock / 离线，不消耗真实 API）。

覆盖：

- M6 / M7 回归（union area、秒制时长）；
- Agent Core / Tool 校验 / Schema；
- Hybrid 安全规则（M7 High 不可降级）；
- Agent Loop（工具调用 + 最终 JSON + Fallback）；
- Memory（Normalizer / Retriever / Tool / Agent 集成）；
- M10 报告（Agent Analysis、旧事件兼容）；
- M11 GUI（offscreen 渲染、信号、历史、关闭安全）。

## 真实 API 冒烟（本地手动执行）

以下脚本需要真实 DeepSeek API Key，仅用于手动验证：

```text
tools/v2_phase1_api_smoke.py
tools/v2_phase1_agent_loop_smoke.py
tools/v2_phase2_agent_smoke.py
tools/v2_phase3_memory_smoke.py
tools/v2_phase4a_agent_report_smoke.py
tools/v2_phase4b_agent_gui_smoke.py
tools/v2_phase6_joint_test.py
tools/v2_final_demo.py
```

## GUI 人工验收

自动化测试不能替代人工 GUI 验收；请按《Phase 4-B 报告》中的验收清单人工检查
图片 / 视频 / Agent 状态 / Memory / Rule-Agent-Final / M10 PDF / 暂停停止 / 关闭安全。
