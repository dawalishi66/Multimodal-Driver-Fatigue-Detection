# UL-DD CAN 基线训练前置准备 v1

本模块落实实验总规范 v0.2 的 CAN 单模态训练前置条件。它不执行正式训练，
不生成 validation/test 成绩，也不宣称完成视频与 CAN 的公平配对主实验。

## 固定范围

- 标签为 KSS < 4、4 <= KSS < 7、KSS >= 7，对应 0、1、2。
- 公共样本是 30 秒窗口，主评测父区间为 8 个固定窗口组成的 240 秒区间。
- 首版模型是 9 维输入投影到 64 维、单层 64 维 GRU、三分类 raw logits。
- 模型会在进入 GRU 前删除 mask=False 的内部位置，因此无效值和尾部 padding
  不影响预测。全无效样本直接拒绝。
- 标准化为全部 9 个通道的 masked z-score，只使用 complete-8 train
  样本中的有效 token 拟合；无效位置变换后固定为零。
- 不默认启用类别权重、过采样或 focal loss。

## 数据范围与限制

预检严格读取冻结的 CAN complete-8 清单：

- train：159 个父区间，1272 个窗口。
- val：41 个父区间，328 个窗口。
- test：预检入口禁止访问。

这是 CAN 单模态 complete-8 集合，不是最终视频+CAN共同配对集合。它可以用于
CAN 管线和单模态初步实验，但在视频30秒清单完成前不能作为多模型公平主对比
结果。正式公平实验必须重新冻结两个模态共同合格的父区间。

## 预检内容

tools/baselines/fatigue_can/prepare.py 将执行：

1. 校验配置中没有占位符，并核对 train/val 清单 SHA-256。
2. 按清单和 parent_id 连接数据，不依赖文件排序。
3. 拟合仅属于 train 的 mask-aware 标准化统计量。
4. 在真实 train batch 上运行一次前向、反向和 AdamW 参数更新。
5. 在真实 val batch 上只检查前向接口和有限 logits，不计算指标。
6. 检查追加无效尾部 padding 后 logits 不变。
7. 保存并重载 smoke checkpoint，检查重载输出一致。
8. 保存解析配置、normalizer、环境、Git状态、哈希和预检报告。

运行包中的 SMOKE_ONLY_DO_NOT_USE.pt 只有一次优化器更新，仅用于验证保存和
重载，禁止作为训练结果或后续模型初始权重。

## 运行命令

在已激活且安装项目的 fatigue_detect 环境中，从仓库根目录执行：

    python tools/baselines/fatigue_can/prepare.py ^
      --processed-root "<Processed_CAN_v1目录>" ^
      --config configs/baselines/fatigue_can/gru_v1.json ^
      --device auto

未指定 output-root 时，新运行写入：

    <Processed_CAN_v1>/baselines/fatigue_can/gru_v1/preflight/<run_id>/

每次生成新目录，不覆盖旧运行。PASS 只表示 G1 和部分 G2 前置检查完成；
跨模态时间同步、共同配对 cohort、正式三种子训练与最终 test 均不在本步骤内。

## 正式训练前仍需冻结

- 视频+CAN共同合格的完整父区间清单，用于公平主对比。
- 正式训练代码提交和无未记录改动的运行状态。
- 训练/验证循环、早停、窗口及父区间预测导出。
- 三个种子 11、22、33 的正式运行计划。
- 模型和选择规则冻结后的 test 一次性评价入口。
