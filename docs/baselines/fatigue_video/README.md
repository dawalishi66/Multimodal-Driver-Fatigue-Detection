# UL-DD 疲劳视频交付 v2

## 交付范围

本页登记邓金祥已有的 UL-DD IR 视频单模态成果及其 30 秒特征交付。完整视频实验合同已经冻结，人工可读版本为源视频仓库的 `docs/experiment_contract_v1.md`，机器可读版本为 `config/experiment_v1.json`，对应代码提交为 `f28ca77a228d20f9aec6fd9c05d96a8299961b91`。

本次只登记代码、合同、脱敏结果和外部 artifact 的版本信息，不上传原始视频、完整特征、模型权重、预测明细或本机路径。

## 数据和标签

- 数据集：UL-DD，IR 视频。
- 标签：保留原始 KSS；`low < 4`、`4 <= medium < 7`、`high >= 7`。
- split：train 为 D/F/G/J/K/L/N/Q/R/S，validation 为 A/E/O，test 为 C/H/P。
- 源视频实验使用 5 秒非重叠窗口、每窗 25 帧；主评测单位为 240 秒 KSS 父区间。
- 30 秒 handoff 由六个连续、真实的 5 秒子窗口组成，不复制特征凑长度。

## 特征接口

`video_handoff_v2.zip` 包含 1,909 个 train/validation 视频候选，每个 NPZ 恰好包含：

| 数组 | dtype 和 shape |
| --- | --- |
| `x` | `float32[6,96]` |
| `time_s` | `float64[6]` |
| `valid_mask` | `bool[6]` |
| `support_s` | `float64[6,2]` |
| `observed_fraction` | `float32[6]` |

96 维向量是冻结蒸馏学生 `distilled_prefix10_r160_mean_seed20260833` 的分类器前输入。独立验证重新应用冻结分类器，11,454 个五秒向量的概率重构最大绝对误差为 `4.76837158203125e-7`。

包的 SHA-256、计数和限制见 `artifact_index/fatigue_video_handoff_v2.json`。大文件通过团队私有渠道获取，使用前必须核对 SHA-256。

## 时间接口迁移

源 handoff v2 的 `time_s` 和 `support_s` 使用 session 相对时间；公共 v0.2 batch 要求 30 秒样本相对时间。消费者必须在进入公共 Dataset 前执行：

```python
video_time_s = video_time_s - window_start_ms / 1000.0
video_support_s = video_support_s - window_start_ms / 1000.0
```

转换后应再次检查时间严格递增、中心位于支持区间内部，且支持区间落在 `[0,30]` 秒。目标仓库的 Video/CAN 配对分支已经按此规则适配；不能把源时间数组未经转换直接传入公共 batch。

## 证据等级和限制

- 1,909/1,909 个特征文件通过结构、dtype、有限值、mask、时间、哈希和概率重构检查。
- 交付只包含 train/validation，test 窗口为 0。
- 源视频实验的三个训练种子是 `20260831/20260832/20260833`，不是当前项目固定的 `11/22/33`，因此旧成绩只能标为 legacy。
- 该交付提供视频候选和特征血缘，不单独证明 Video/CAN 同步。共同集合、同步证据和 complete-8 审计由配对流程负责。
- 该交付不等于 PR #9 所需的同共同集合视频单模态新基线；若要完成四模型公平比较，仍须在冻结 paired cohort 上按种子 11/22/33 训练并审计。

旧实验结果的可公开摘要见 `results/baselines/fatigue_video/legacy_v1.md`。
