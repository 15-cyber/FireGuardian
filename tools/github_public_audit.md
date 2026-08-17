# FireGuardian GitHub 公开版安全审计

> 日期：2026-08-17
> 工具：`tools/github_public_audit.py`（可复现）
> 结论：**PASS**

## 结果

| 检查项 | 结果 |
| --- | --- |
| Secrets | PASS |
| .env（已跟踪） | PASS（仅 `.env.example`） |
| .env.example | PASS（已入库，占位无 Key） |
| Model weights | PASS（无 `.pt` 入库） |
| Videos | PASS（无 `.mp4` 入库） |
| Large files | PASS（无 >1MB 文件入库） |
| Absolute paths | PASS（无 `Codex项目` / `火焰数据集` / 盘符路径 / 用户目录） |
| History secrets | PASS（全历史 `git grep` 排除审计工具后无密钥） |

## 处理记录

- `me.mp4`（V1 早期误跟踪的测试视频）已从索引移除（本地文件保留，`/*.mp4` 已忽略）；
- 巨型生成报告（`smoke_subset_report.*`、`smoke_e2e_verify_result.json`）已从公开树移除；
- `config/config.yaml`、`datasets/fire.yaml` 及 19 个工具/报告/文档中的本机绝对路径已替换为占位符；
- 审计脚本中的扫描模式采用运行时拼装，避免自匹配误报。

## 说明

- 历史 commit 中仍保留早期开发产物（含 me.mp4 旧 blob），为保留真实开发历史未重写历史；
- 如需彻底清除历史中的视频 blob，需重写历史（与「保留历史真实性」原则冲突，未执行）。
