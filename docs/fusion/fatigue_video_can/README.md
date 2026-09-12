# UL-DD 疲劳视频＋CAN融合前置检查

本模块只负责融合模型的公共数据接入与真实特征 G2 冒烟，不代表正式训练或模型成绩。

## 已冻结输入

- cohort：`fatigue_video_can_complete8_train_val_v1`
- train：1,272 个30秒窗口、159个240秒父区间
- validation：328个30秒窗口、41个240秒父区间
- 视频：`float32[6,96]`
- CAN：`float32[300,9]`
- 两个模态在 batch 中都使用当前30秒样本的相对时间
- test 在此入口中不可访问

视频和CAN分别使用仅由train有效token拟合的 `masked_zscore_v1`。统计量不能跨模态共享，也不能使用validation或test拟合。

## G2运行

```powershell
python tools/fusion/fatigue_video_can/prepare.py `
  --dataset-root <UL-DD根目录> `
  --pair-root <Paired_Video_CAN_v1目录> `
  --device auto
```

默认从train和validation各选择16个覆盖三类的真实窗口，依次对 `SimpleFusion` 与
`DualModalMulT` 完成一次前向、交叉熵反向、优化器更新、尾部padding不变性检查，以及
checkpoint保存重载。输出位于外部配对目录的
`preflight/fatigue_video_can/g2_v1/<run_id>/`，不会写入Git仓库。

输出中的 `SMOKE_ONLY_DO_NOT_USE.pt` 只有一次优化器更新，不是训练权重。G2不计算
Accuracy、Macro-F1或其他性能指标，也不访问test。

## 当前模型边界

MulT v1 接收并校验 `time_s`，但当前位置编码仍为序列正弦位置，不使用真实时间值。
这不妨碍基础MulT训练，但必须在报告中明确，后续真实时间编码应作为独立改进版本和消融项。

## Train/validation开发入口

G2通过并将代码提交后，两个模型分别使用独立冻结配置启动；入口不接受test路径：

```powershell
python tools/fusion/fatigue_video_can/train.py `
  --dataset-root <UL-DD根目录> `
  --pair-root <Paired_Video_CAN_v1目录> `
  --config configs/fusion/fatigue_video_can/simple_fusion_v1_train_val.json `
  --device cuda

python tools/fusion/fatigue_video_can/train.py `
  --dataset-root <UL-DD根目录> `
  --pair-root <Paired_Video_CAN_v1目录> `
  --config configs/fusion/fatigue_video_can/mult_v1_train_val.json `
  --device cuda
```

默认按11、22、33三个随机种子训练；每轮只使用窗口级交叉熵，使用validation的240秒
父区间Macro-F1早停和选模。输出保存在外部配对目录的
`fusion/fatigue_video_can/<model_id>/runs/<experiment_id>/`。当前配置和入口仍属于
train/validation开发实验，最终方案冻结前不得解锁test。
