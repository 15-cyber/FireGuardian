# FireGuardian 优化备忘

> 记录后续需要优化的方向，第一阶段先跑通全流程

## 🔧 训练速度优化（M2）

- [ ] 数据集从 F 盘移到 SSD（C 盘），更新 `config.yaml` 中 `dataset.path`
- [ ] 启用 `cache: true` 将图片缓存到内存（约需 10GB 空闲内存）
- [ ] 启用 `workers=2` 或 `workers=4` 多进程加载（需要在终端直接运行，而非 Codex 调用）
- [ ] 调大 `batch` 从 16 → 32（需显存足够）
- [ ] 预估：以上优化后 50 epoch 可在 ~4-6 小时完成

## 🎯 精度优化

- [ ] 训练更多 epoch（当前默认 100，YOLO 通常 30-50 收敛）
- [ ] 调参：learning rate、optimizer、数据增强参数
- [ ] 尝试更大的模型：yolo11s.pt → yolo11m.pt（精度更高，但更慢）
- [ ] 交叉验证、超参搜索

## 🚀 后续扩展

- [ ] DeepSeek/OpenAI API 集成（Agent 更自然的事件分析）
- [ ] 企业微信/邮件/Webhook 告警
- [ ] 多摄像头输入
- [ ] Docker 部署
- [ ] FastAPI 后端 + Vue3 前端
- [ ] 历史事件检索与统计分析
