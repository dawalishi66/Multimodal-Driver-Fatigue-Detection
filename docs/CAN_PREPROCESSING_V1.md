# UL-DD CAN 预处理 v1

本模块只处理疲劳任务的 Driver telemetry，不修改原始数据，也不执行训练。

## 输入与候选范围

- 输入：UL-DD 根目录下的 `CSV_Files.zip`、`Labels.csv`、`Info.xlsx`。
- 原始 Telemetry 表头必须与数据集公开字段完全一致。
- 只把团队固定 train/val/test 名单中的驾驶员写入标准 metadata。
- 不在固定名单中的 B、I、M 只进入审计汇总，不擅自分配 split。
- 有标签但缺少 Telemetry 的 session 仍保留 80 个失败窗口，不能静默跳过。

## 时间、标签与特征

- 共同候选网格从每个 session 的采集零点开始，使用 `[start,end)`。
- 每个 240 秒标签区间产生 8 个 30 秒窗口，步长 30 秒。
- KSS 标签遵守 `<4`、`[4,7)`、`>=7` 三分类并保留原始数值。
- 以首个 Telemetry capture timestamp 作为 CAN 内部零点。该零点在正式融合前仍需视频元数据交叉确认。
- timestamp 回退或单步跳变超过 1 秒时，不按 60 Hz 行号修复。首个异常之后的候选窗口写为 `PENDING`，不生成误导性特征。
- 对可信区间使用 100 ms 时间桶聚合到 10 Hz。桶均值同时起到低通作用，避免直接抽帧的混叠。
- heading、pitch、roll 的单位是度，分别转换为 sin/cos；speed 和 rpm 取均值；gear 取桶内最后观测值。
- 输出特征顺序固定为 9 维，未做标准化。标准化参数必须只从 train split 拟合。

## 输出结构

每个可分析窗口保存一个 NPZ：

- `x: float32[300,9]`
- `time_s: float64[300]`
- `valid_mask: bool[300]`
- `support_s: float64[300,2]`
- `observed_fraction: float32[300]`

真实输出默认放在 UL-DD 下的独立 `Processed_CAN_v1/` 目录。该目录不应提交 GitHub。
`metadata/can_parents_240s_v1.csv` 额外标记哪些父区间具备完整 8 个有效窗口，供 240 秒主指标使用。

## 运行

在仓库根目录安装项目后运行：

```powershell
$ulddRoot = "<UL-DD 数据集目录>"
python tools/build_can_metadata.py `
  --dataset-root $ulddRoot `
  --output-root (Join-Path $ulddRoot "Processed_CAN_v1") `
  --config configs/can_preprocessing_v1.json
```

验证：

```powershell
$processedRoot = Join-Path $ulddRoot "Processed_CAN_v1"
python tools/validate_can_metadata.py `
  --metadata (Join-Path $processedRoot "metadata/can_windows_30s_v1.csv") `
  --feature-root $processedRoot `
  --report (Join-Path $processedRoot "reports/can_validation_v1.json")
```

验证器 PASS 只代表 metadata 和特征结构、标签、split、时间网格及 CAN 特征约束通过，不代表视频与 CAN 已完成同步验收。

## 冻结训练划分

在预处理与验证完成后生成训练清单，不复制 NPZ：

```powershell
python tools/split_can_metadata.py `
  --metadata (Join-Path $processedRoot "metadata/can_windows_30s_v1.csv") `
  --output-root $processedRoot
```

训练、验证和最终测试分别读取 `manifests/can_train_windows_v1.csv`、
`can_val_windows_v1.csv` 和 `can_test_windows_v1.csv`。被排除的行单独保存在
`can_excluded_windows_v1.csv`，不得进入训练。
