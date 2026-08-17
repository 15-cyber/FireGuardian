# Security Policy

## 安全使用准则

- 不要提交 API Key 或任何敏感凭据（本项目 `.env` 已被 `.gitignore` 忽略）；
- 不要提交敏感数据、原始数据集、模型权重或大体积视频到仓库；
- 配置 DeepSeek API 时请使用本地 `.env`，并通过环境变量读取。

## 报告漏洞

本项目为研究 / 求职 Demo，无专职安全团队。如发现安全问题，请在 GitHub Issues 中描述问题与复现方式；涉及敏感信息时请不要在 Issue 中粘贴任何密钥。

建议仓库开启 GitHub 的 Secret scanning、Push protection 与 Dependabot alerts 以降低公开代码暴露风险。
