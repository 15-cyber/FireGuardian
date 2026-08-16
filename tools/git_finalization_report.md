# FireGuardian V2 本地 Git 封板报告

> 日期：2026-08-16
> 说明：仅本地 commit + tag；未 push、未创建 GitHub Release、未修改远程仓库。

---

## 1. 当前分支

`v2-deepseek-agent`

## 2. 封板前 HEAD

`0eca3ad`（docs: document baseline v1 audit and model promotion，tag `v0.1.0-baseline`）

## 3. 新增 commit 列表

| Commit | 消息 | 内容 |
| --- | --- | --- |
| `bb37bb2` | feat: finalize FireGuardian V2 agent pipeline | Agent/Memory/Hybrid/M10/M11 代码、UiController Worker、PresentationModel、prompts、配置、Phase 1~4 测试与报告（77 文件） |
| `acbe417` | feat: add V2 YOLO optimization and joint evaluation | Phase 5 数据/训练/评估/conf 工具与结果、Phase 6 联合测试、V2.7 闭环演示脚本与报告（20 文件） |
| `<final_commit>` | docs: finalize FireGuardian V2 technical report | 研发技术总结报告 md/pdf、PDF 构建脚本、本封板报告 |

## 4. 每个 commit 的目的

- Commit 1：固化 V2 Agent 主线（Phase 1~4）全部未提交代码与测试；
- Commit 2：固化 Phase 5/6/V2.7 的模型优化与联合评估代码、结果与报告；
- Commit 3：固化最终研发技术总结报告与封板文档。

## 5. 最终 HEAD

见 `git log --oneline --decorate -20`（最终 commit 为 Commit 3）。

## 6. v0.1.0-baseline

已存在（指向 `0eca3ad`，V1 稳定基线）。

## 7. v0.2.0-agent

本任务创建：`git tag -a v0.2.0-agent -m "FireGuardian V2 Agent Stable Demo"`，
指向最终稳定 commit。

## 8. 最终测试数量

`python -m unittest discover -s tests -p "test*.py"` → **186/186 通过**。

## 9. compileall 结果

`python -m compileall agent ui reports aggregator video_detect simulator tools tests` → **通过（exit 0）**。

## 10. git diff --check 结果

**通过（exit 0）**，仅 LF→CRLF 提示，无空白错误。

## 11. 敏感信息扫描

- `.env` 已忽略（`git check-ignore .env` 命中）；
- `git grep -n -I -E "DEEPSEEK_API_KEY|Authorization: Bearer|sk-[A-Za-z0-9]"`（已跟踪内容）→ 无命中；
- 候选入库文件扫描：`DEEPSEEK_API_KEY`/`api_key` 命中均为**环境变量名/参数名的正常引用**；
  `sk-` 密钥样式、`Bearer sk-`、真实 Key 片段 → **零命中**。

## 12. 大文件扫描

- 候选入库文件最大为 `docs/FireGuardian_V2_研发技术总结报告.pdf`（约 934KB，16 页技术报告，判定适合入库）；
- 其余入库文件均 ≤ 25KB；
- 模型（.pt）、视频、截图、负样本图片、缓存、日志均未入库。

## 13. 被排除的模型/视频/日志

- `models/*.pt`、`models/*.jsonl`
- `me.mp4`、`烟火.mp4`、`测试视频/`（烟花1/2）
- `runs/`、`logs/`（含 `logs/thinking_eval/`）、`screenshots/`
- `tools/v2_phase3_fireworks/representative_frames_*/`
- `tools/v2_phase4a_output/`、`tools/v2_phase4b_screenshots/`、`tools/v2_final_demo/`（运行产物）
- `datasets/v2_negatives/`（250MB 负样本图片）、`datasets/.v2_scan_cache.json`
- `tools/baseline_v1_visual_audit.pdf`（11MB）
- 根目录任务书 docx 与测试图片（历史仓库从未跟踪 docx，保持一致）

## 14. git status 最终结果

工作区干净（`git status --short` 无输出）。

## 15. 是否 push

**NO**（未执行 `git push` / `git push --tags`，未创建 GitHub Release）。

---

**FireGuardian V2 已完成本地 Git 封板，v0.2.0-agent 已指向最终稳定版本。**
