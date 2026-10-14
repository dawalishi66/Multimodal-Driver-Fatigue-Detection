# G3 疲劳 Video＋CAN 集成预检（2026-10-09）

本记录是李坤洋在 G3 的代码与真实数据接口预检，不是 G3 最终合并结论，也不是正式 test。

- 从 PR #9 的远端 head `6735b1f30ea3d51b7f61856e05ae7188e3793c89` 建立干净副本，合入 G1 完成后的 `main`。没有将旧本地集成分支中的分心六分类模型修改推入疲劳 PR。
- 同名文档冲突涉及 G1 视频特征 handoff 与 G2 同集合视频基线。两份文档分别保留为 `docs/baselines/fatigue_video/HANDOFF_V2.md` 和 `README.md`，避免覆盖历史交付。
- 冻结共同集合为 train 1,272 个、validation 328 个 30 秒窗口；视频 `[6,96]`、CAN `[300,9]`；疲劳标签三分类。G2 通过的视频基线、CAN 基线、SimpleFusion 和 DualModalMulT 的 train/val 结果均可由各自 artifact 索引追溯。
- 完整 CPU 测试执行通过，286 个测试项被收集；仓库内容策略检查通过。PR 相对 `main` 的差异未包含分心模块文件、原始数据、NPZ、权重或 ZIP。
- 新建的真实特征小批量冒烟在干净代码提交 `5db59b7cd51aee5d6bb546a145c9646a0d1367a2` 上运行，SimpleFusion 和 DualModalMulT 的真实特征读取、train-only 标准化、前后向、尾部 padding 不变性及 checkpoint 保存重载全部 PASS。
- 冒烟报告位于外部 `Paired_Video_CAN_v1/preflight/fatigue_video_can/g3_recheck_publish_20261009/preflight_report.json`；SHA-256 为 `0cb8a0536d59ea965768fb2f15ea3e824a990a4ebc9c8a46e81ee9d8b82d2ebc`，见 `artifact_index/fatigue_video_can_g3_recheck_20261009.json`。仅执行一次优化器更新，不计算性能指标；`test_manifest_accessed=false`。

仍需：将本集成提交推送到 PR #9，确认该 head 的 CI 全绿，并由邓金祥审核视频输入/配对、饶棋涛审核两个融合模型。非作者审核与最终负责人检查通过前，不合并到 `main`。
