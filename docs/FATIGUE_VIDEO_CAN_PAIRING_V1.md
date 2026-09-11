# UL-DD 视频与 CAN 配对 v1

## 用途

该流程冻结疲劳任务的 train/val 双模态共同集合。它不读取 test，不复制原始视频、CAN
特征或视频特征，只生成可追溯清单与验证报告。

配对只使用：

```text
subject_id + session_id + window_start_ms + window_end_ms
```

同时要求 split、KSS、类别编号和 240 秒标签区间完全一致。正式主实验仅使用每个父区间
恰好包含八个连续 30 秒窗口的 complete-8 集合。

## 同步证据

UL-DD 论文第 4.2 节说明发布数据已经按人工记录的 session 时间窗对齐：视频经过裁剪，
Telemetry 通过观察其在视频中的启动时刻进行对齐并裁掉多余数据。本地审计进一步比较
IR 容器时长与 Telemetry 的 60 Hz 行数时长。

项目使用 `provider_documented_aligned`，表示发布者记录的 session 相对对齐，不表示两个
采集设备共享硬件时钟。对共同 train/val session，本地时长差最大为 59 ms，配对偏移设为
0 ms。

来源：<https://pmc.ncbi.nlm.nih.gov/articles/PMC13039290/>

## 冻结结果

| split | 驾驶员 | session | 240 秒父区间 | 30 秒窗口 |
| --- | ---: | ---: | ---: | ---: |
| train | 10 | 18 | 159 | 1,272 |
| val | 3 | 5 | 41 | 328 |

1,622 个窗口完成直接配对，其中 22 个因父区间不足八个窗口而不进入主实验。视频侧其余
287 个候选没有可用 CAN 窗口。test 未读取。

## 时间接口修正

视频 handoff v2 的时间数组是 session 相对时间，CAN v1 是 30 秒样本相对时间。加载器
必须对视频执行：

```python
video_time_s -= window_start_ms / 1000
video_support_s -= window_start_ms / 1000
```

转换后两个模态都使用 0–30 秒样本相对坐标。不得直接把原始视频时间数组与 CAN 时间数组
传入同一个 batch。

## 生成命令

```powershell
python tools/build_fatigue_video_can_pairs.py `
  --dataset-root <UL-DD根目录> `
  --video-package <视频handoff-v2.zip> `
  --can-root <Processed_CAN_v1目录> `
  --raw-video-archive <Video_Data.zip> `
  --output-root <Paired_Video_CAN_v1目录>
```

输出目录已存在时工具拒绝覆盖。重新生成新版本时应使用新的版本目录，保留旧清单和哈希。
