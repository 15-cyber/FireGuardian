# 部署与运行说明

## 本地运行

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

## 配置

1. 复制 `.env.example` 为 `.env` 并填写 `DEEPSEEK_API_KEY`；
2. 将模型权重放入 `models/`（参见 [models/README.md](../models/README.md)）；
3. 如需训练 / 数据集功能，将 `config/config.yaml` 中的数据集路径替换为本地数据集路径（原始数据集不随仓库发布）。

## 启动 GUI

```bash
python main.py
```

## 无模型 / 无 Key 时的行为

- 无模型权重：检测模块会报告找不到模型文件，需要放置权重后才能检测；
- 无 DeepSeek Key：Agent 不可用，系统回退到 M7 规则决策，基础检测与报告链路仍可运行。

## GPU 建议

训练与视频检测建议使用 NVIDIA GPU（CUDA）；推理亦支持 CPU（较慢）。

## CI

GitHub Actions（`.github/workflows/tests.yml`）只运行 Mock / 离线单元测试，不调用真实 DeepSeek API，不使用 Secrets。
