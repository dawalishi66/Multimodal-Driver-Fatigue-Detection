# UL-DD CAN 基线与训练前置准备 v1

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

运行时视频+CAN共同清单尚未冻结，因此原运行包保守标记为不可公平对比。
共同清单冻结后的独立审计已证明：训练器按当时绑定的 complete-8 父区间
过滤后，有效 CAN train/val 集合与视频+CAN共同集合逐条相同。因此现有
CAN train/val 结果可用作同一共同集合上的单模态对照，无需重训。这不解锁 test。

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

## CAN-only train/val 开发实验

`gru_v1_train_val.json` 和 `train.py` 用于当前 CAN complete-8 集合上的
单模态开发训练。它严格只读取 train/val，先完成 seed 11 的完整性关卡，再继续
seed 22、33；三个种子分别用 validation 父区间 Macro-F1 早停和选择检查点。

    python tools/baselines/fatigue_can/train.py ^
      --processed-root "<Processed_CAN_v1目录>" ^
      --config configs/baselines/fatigue_can/gru_v1_train_val.json ^
      --device cuda

未指定输出目录时，完整运行包写入
`<Processed_CAN_v1>/baselines/fatigue_can/gru_v1/runs/<experiment_id>/`。
运行包保存每个种子的最佳参数、训练曲线、validation 窗口和父区间预测、指标、
环境、配置及哈希。它不读取或评价 test，并明确标为 `formal_result=false`。

后续共同集合审计结果为 PASS：CAN 有效 train 1,272 窗口/159 父区间和
val 328 窗口/41 父区间与双模态清单完全一致，所有样本 ID、身份、时间、
标签及 CAN 特征引用差异为 0。原运行包不改写，通过新审计报告补充证据。

## 共同集合审计

审计入口：

    python tools/baselines/fatigue_can/audit_paired_cohort.py ^
      --processed-root "<Processed_CAN_v1目录>" ^
      --pair-manifest "<complete8-train-val.csv>" ^
      --run-dir "<CAN train-val运行包>" ^
      --report "<审计报告.json>" ^
      --code-version "<Git提交>"

审计报告保存在 `baselines/fatigue_can/gru_v1/audits/` 下。它只证明 train/val
集合等价；最终仍需在融合模型和选择规则冻结后，对共同 test 集合执行一次评价。
