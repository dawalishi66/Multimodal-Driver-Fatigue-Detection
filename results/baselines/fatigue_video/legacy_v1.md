# UL-DD 疲劳视频 legacy v1 结果

## 定位

这些结果来自已冻结的独立视频实验合同，用于说明视频子系统已有的研究证据。它们不使用当前多模态仓库固定的训练种子 11/22/33，也不是 `fatigue_video_can_complete8_train_val_v1` 共同集合上的对照，因此不得放入 PR #9 的四模型公平主表。

## 独立驾驶员 test

主评测为 240 秒父区间概率平均后的 Macro-F1：

| 模型 | 角色 | Macro-F1，均值 ± 样本标准差 |
| --- | --- | ---: |
| Mean + Linear | Primary | 0.372465 ± 0.016007 |
| Lightweight TCN | Challenger | 0.418415 ± 0.018811 |

- test 驾驶员：C、H、P。
- test 窗口：2,400；父区间：50。
- 原视频实验种子：20260831、20260832、20260833。
- test 在模型、ROI、聚合方式和角色冻结后执行一次；后续学生模型没有继承这些 test 分数。

## 部署候选

部署 handoff 使用蒸馏学生 `distilled_prefix10_r160_mean_seed20260833` 的 96 维分类器前特征。该学生按 validation 中位 seed 规则选择，属于 validation 级 CPU 候选，不拥有上表 Mean/TCN 的独立 test 成绩。

## 使用边界

- 这些数字只作为 legacy 视频研究证据，不是当前多模态项目的正式公平对比结果。
- 不能根据旧 test 结果为新融合模型选择结构、阈值或数据处理规则。
- 当前项目若需要视频、CAN、SimpleFusion 和 DualModalMulT 的公平比较，必须使用同一 complete-8 train/validation cohort、种子 11/22/33 和当前统一评测器重新训练视频单模态头。
- 原始媒体、完整特征、checkpoint 和预测明细不进入 Git。
