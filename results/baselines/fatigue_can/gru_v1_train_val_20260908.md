# UL-DD CAN GRU v1：train/val 开发实验

## 实验身份

- 实验 ID：`can_gru_v1_trainval_20260908T100437Z`
- 代码提交：`0f9367383fee08f89aa2747ea1d50bc4ae0c177b`
- 协议／接口：`0.2`／`0.2.0`
- 配置版本：`1.1.0-train-val`
- 特征版本：`uldd_can_10hz_boxcar_1.0.0`
- cohort：`can_single_modality_complete8_v1`
- 设备：NVIDIA GeForce RTX 5060 Laptop GPU
- 状态：PASS；`formal_result=false`；`test_manifest_accessed=false`

本实验运行时视频+CAN 共同清单尚未冻结，因此原运行包保守标记为不可公平对比。
后续独立审计已证明该运行实际使用的 complete-8 train/val 集合与双模态共同集合
完全一致，因此这些结果可作为同一 train/val 集合上的 CAN 单模态公平对照，无需重训。

## 数据与训练规则

| Split | 驾驶员 | Session | 30秒窗口 | 240秒父区间 | Low | Medium | High |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 10 | 18 | 1272 | 159 | 296 | 528 | 448 |
| Validation | 3 | 5 | 328 | 41 | 112 | 112 | 104 |

标签固定为 KSS `<4 / [4,7) / >=7`。模型为9维输入投影至64维、单层64维
GRU和三分类头。使用 CrossEntropyLoss、AdamW、学习率3e-4、weight decay
1e-4、batch size 32；最多100轮，以 validation 父区间 Macro-F1 选模，连续
15轮无提升早停。未使用 AMP、调度器、类别权重、过采样或 focal loss。

## 三种子验证结果

| Seed | 最佳轮／实际轮数 | 窗口 Macro-F1 | 窗口 BAcc | 父区间 Macro-F1 | 父区间 BAcc | 父区间 Accuracy | KSS>=7 Recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 6 / 21 | 0.2886 | 0.3274 | 0.3008 | 0.3590 | 0.3659 | 0.0769 |
| 22 | 71 / 86 | 0.3552 | 0.3677 | 0.4433 | 0.4652 | 0.4634 | 0.5385 |
| 33 | 12 / 27 | 0.3099 | 0.3402 | 0.3299 | 0.3883 | 0.3902 | 0.3077 |
| 均值 | — | 0.3179 | 0.3451 | 0.3580 | 0.4042 | 0.4065 | 0.3077 |
| 样本标准差 | — | 0.0340 | 0.0206 | 0.0753 | 0.0549 | 0.0508 | 0.2308 |

train 标签确定的多数类为 Medium。其 validation 父区间 Macro-F1 为
0.1697，三个种子相同。CAN GRU 的三种子父区间平均 Macro-F1 高于该多数类
参照，但种子波动和 KSS>=7 召回波动较大，当前不据此宣称模型已经稳定或最优。

父区间各类真实支持数固定为 Low 14、Medium 14、High 13。完整的每类
Precision／Recall／F1、混淆矩阵、每名驾驶员两种 Macro-F1 口径及窗口预测
均保存在外部运行包中。

## 验收与限制

- 三个种子均从最佳检查点重新载入；重复载入的 validation 概率最大差为0。
- 每个种子均导出328条唯一窗口预测和41条完整父区间预测。
- 已从预测 CSV 独立重算窗口及父区间指标，与 `metrics.json` 一致。
- 已核对运行包38个文件及三个 checkpoint 的 SHA-256。
- 训练入口没有 test 清单路径，运行包没有 test 预测或 test 指标。
- 后续 G0 审计已确认共同 train/val 集合等价；审计中样本、父区间、身份/标签/时间
  及 CAN 特征引用差异均为 0，未重训模型。
- 本结果仍不是规范 G3 的正式 train/val/test 主实验；test 仍未解锁。

外部运行包相对 `Processed_CAN_v1` 的位置：

```text
baselines/fatigue_can/gru_v1/runs/can_gru_v1_trainval_20260908T100437Z
```

文件大小和 SHA-256 见 `artifact_index/fatigue_can_baseline_v1.json`。
共同集合等价性报告相对 `Processed_CAN_v1` 的位置：

```text
baselines/fatigue_can/gru_v1/audits/can_gru_v1_paired_cohort_equivalence_v1.json
```
