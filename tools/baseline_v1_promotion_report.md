# Baseline V1 模型晋升记录

- **执行时间**：2026-08-05T16:39:39（Asia/Shanghai）
- **候选模型路径**：`models/baseline_v1_best.pt`
- **候选模型类别**：`{0: "fire", 1: "smoke"}`，`nc = 2`
- **晋升前默认模型 SHA-256**：`716cb01d9eed4a88551666b1ec07bb5fbf3d274a3c994bf951d53703e1dafddd`
- **备份模型路径**：`models/best.backup_20260805_163939.pt`
- **备份模型 SHA-256**：`716cb01d9eed4a88551666b1ec07bb5fbf3d274a3c994bf951d53703e1dafddd`（与晋升前一致）
- **晋升后默认模型 SHA-256**：`5b84350ebdfaf489904e43515f83096c1d6060f84183f5c0b97e1b51fbdf6dcb`
- **是否与候选模型一致**：是（字节级一致）
- **晋升命令**：

  ```powershell
  python tools/promote_model.py --candidate models/baseline_v1_best.pt --execute
  ```

- **最终状态**：成功。`models/best.pt` 已替换为 Baseline V1（YOLO11n，fire/smoke）；旧冒烟模型已备份；`models/smoke_v1_best.pt` 保持不变；版本记录写入 `models/model_versions.jsonl`。
- **回滚方式**（如需）：将备份文件恢复为默认模型：

  ```powershell
  Copy-Item models/best.backup_20260805_163939.pt models/best.pt -Force
  ```

- **说明**：模型权重 `.pt` 文件不进入普通 Git；`model_versions.jsonl` 为本地运行记录，不入库。
