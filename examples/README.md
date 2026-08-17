# Examples

The original demo videos are not included due to repository size and data
distribution considerations.

测试视频（me.mp4、烟花1.mp4、烟花2.mp4、烟火.mp4）仅用于本地实验和评估，不随仓库发布。

你可以使用自己的图片或视频作为输入：

```text
python main.py          # GUI：选择图片 / 视频文件作为检测源
```

相关演示工具：

```text
tools/v2_final_demo.py                # 视频 → YOLO → M6 → M7 → Agent → Hybrid → 报告/GUI 闭环演示
tools/v2_phase4b_agent_gui_smoke.py   # Agent → GUI 展示冒烟（offscreen）
```
