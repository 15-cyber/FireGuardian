# FireGuardian GitHub 公开版版权表述更新记录

> 日期：2026-08-17

## 修改内容

### 1. README 版权表述

原表述（易引起误解）：

> 项目依赖的第三方开源库见 `requirements.txt`，本项目不声称拥有这些库的源代码版权。

修改为：

> 本项目原创代码版权归项目作者所有，并以 MIT License 授权。项目依赖的第三方开源组件分别遵循其各自许可证，相关依赖及许可证信息见 `requirements.txt`。

### 2. 非商业用途说明

README 增加：

> 本项目仅用于学习、研究与个人求职展示，不得用于任何商业用途。

**提示**：MIT License 允许商业使用；如严格要求非商业用途，建议后续将 LICENSE 更换为非商业许可（例如 CC BY-NC 系列或自定义许可），由项目作者确认后执行。

### 3. LICENSE 检查

- 年份：`Copyright (c) 2026` 正确；
- 版权持有人：当前为 `FireGuardian contributors`，**待项目作者确认真实公开姓名后更新**；
- 未将第三方依赖版权归属于本项目；
- 未修改任何第三方依赖的许可证。

### 4. 展示照片

将「展示照片」文件夹中的 5 张照片按顺序加入 README「项目展示」章节：

1. 过程1 → `docs/assets/showcase_process_1.png`
2. 过程2 → `docs/assets/showcase_process_2.png`
3. 模型结果 → `docs/assets/showcase_model_result.png`
4. agent分析结果 → `docs/assets/showcase_agent_analysis.png`
5. 报告 → `docs/assets/showcase_report.png`

原始「展示照片」文件夹已加入 `.gitignore`，不入库；副本仅保留在 `docs/assets/`。

## 一致性检查

- README 版权说明与 LICENSE（MIT）一致：✅（持有人待更新）；
- `requirements.txt` 第三方依赖保留：✅；
- 未复制第三方依赖源码进仓库：✅；
- 未修改项目功能代码：✅；
- 未修改 Git 历史：✅；
- 未 push、未创建 GitHub Release：✅。

## 复检结果

- Secret audit：PASS
- Absolute path audit：PASS
- Large file audit：PASS
- `git diff --check`：通过
- 全量测试：186/186
- `git status`：干净
