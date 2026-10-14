# DCPT 分心音频交付说明 v1（胡煦轩 | G4 整理）

本文件是 G4「音频模块整理」的交付入口：固定代码状态、音频接口、质量摘要、
给陈星宇的核对清单与哈希、实测结果及未决事项。它只做**字段与结构**交付，
不代表 P04 人工同步签署，也不代表 6c 标签或 24/8/8 划分已冻结。

- 处理说明：[docs/audio_processing_v1.md](audio_processing_v1.md)
- 工件索引：[artifact_index/audio_dcpt_v1.md](../artifact_index/audio_dcpt_v1.md)
- 入口分支/PR：`huxx07:feat/distraction-audio-baseline` -> PR #6（基准 `main`）
- 真实特征与权重不进入仓库，受控存储本地交付（见「哈希清单」）。

## 1. PR #6 与 PR #7 的提交关系

- 两个 PR 的 head 完全一致：`huxx07:feat/distraction-audio-baseline`，
  head SHA `2594d3fcabc1c7e1fbae20b9de36b76e85af8663`（PR #6 基准 `main`，
  PR #7 基准 `feat/distraction-audio-baseline`）。
- 由于同一 head，#7 相对 #6 **没有任何额外提交**；#7 想表达的音频改动在 #6 中
  已经全部包含。**结论：#6 是音频的最终入口，#7 是冗余分支。**
- 起始审计提交 `26cd018` 是 `origin/feat/distraction-audio-baseline` 的祖先
  （`git merge-base --is-ancestor 26cd018 origin/feat/distraction-audio-baseline`
  返回 0），说明 #7 的来源提交也已并入 #6 所在分支。

## 2. 同步最新 main

- 固定 main：`0056e916755fd8ec33e9c30622bb512dd10c631d`（合并前需再次确认）。
- 音频分支在保留全部历史的前提下合并该 main：合并点 `80b7a21`
  （`Merge remote-tracking branch 'origin/main' into feat/distraction-audio-baseline`）。
  全程 `merge`，未 rebase、未 squash，历史线性可追。
- 冲突处理原则：公共模型与疲劳接口（`src/driver_state/schemas`、
  `driver_state.validation.metadata`、`driver_state.models` 公共部分、
  `manifests/` 公共约定）以 main 为准；音频侧只在其上叠加。

- 整理后的**固定提交** = 本交付文档所在的提交（即 push 后的 PR #6 head）；
  固定 SHA 在推送后记录到 PR #6 审核评论，供陈星宇按 SHA 复核。

## 3. 最终差异范围（对比 main：31 文件，+4034 / -3）

音频主体（新增）：

- 预处理与契约：`src/driver_state/preprocessing/distraction_audio/`
  下的 `naming.py`、`qc.py`、`build_metadata.py`、`fusion_labels.py`、
  `build_audio_6c_metadata.py`、`validate_audio_6c.py`、`check_fusion_pairs.py`、
  `extract_audio_features.py`、`panns_loader.py`、`build_audio_evaluation.py`、
  `__init__.py`（docstring 整理）。
- 单模态基线：`src/driver_state/models/unimodal/train_audio_baseline.py`。
- 薄 CLI：`scripts/build_distraction_audio_metadata.py`、
  `scripts/build_audio_6c_metadata.py`、`scripts/validate_distraction_audio_6c.py`、
  `scripts/validate_distraction_audio_metadata.py`、`scripts/check_fusion_pairs.py`、
  `scripts/build_audio_evaluation.py`。
- 测试：`tests/test_distraction_audio_naming.py`、`test_distraction_audio_qc.py`、
  `test_distraction_audio_metadata.py`、`test_distraction_audio_6c.py`、
  `test_train_audio_baseline.py`、`test_build_audio_evaluation.py`、`tests/wav_utils.py`。
- 文档与索引：`docs/audio_processing_v1.md`、`artifact_index/audio_dcpt_v1.md`。

**公共依赖的必要修改（逐项）**：

1. `pyproject.toml`（+7）：新增 `audio` 可选依赖
   （`scipy`、`scikit-learn`、`soxr`、`torch`、`torchlibrosa`）。音频预处理/特征/
   基线所需，缺失则无法安装运行。**已知不一致**：`audio` 的 `torch>=2.0`
   无上界，而 `train` extra 为 `torch>=2.7,<3`；建议后续统一，本阶段不改。
2. `.gitignore`（+3）：忽略本地完整规范工作副本
   `驾驶员状态检测项目_实验总规范_v0.2.md`（含机器路径，不入库）。
3. `tests/test_dual_modal_mult.py`、`tests/test_simple_fusion.py`（各 1 行）：
   将 `import torch` 改为 `torch = pytest.importorskip("torch")`，仅让无 torch
   环境优雅跳过，不改融合行为与断言。

## 4. 音频接口声明

- `label_scheme = dcpt_video_6c_v1`，与分心视频模块逐字一致（fusion 可 join）。
- 类别顺序 0-5：`No task, Playing game, Messaging, Phone call, Reading, Eating`。
- 原始任务映射：`01->0, 03->1, 04->2, 05->3, 07->4, 08->5`
  （原 02/06/09 不在本协议范围内）。
- 特征候选：`x float32[5,2048]`，feature_version `panns_cnn14_16k_v1`；
  抽取器/权重版本另记 `cnn14_16k_mAP0.438_v1`。
- NPZ 恰好保存团队规定的五个数组：`x [T,D] float32`、`time_s [T] float64`、
  `valid_mask [T] bool`、`support_s [T,2] float64`、`observed_fraction [T] float32`。
- metadata 列顺序与视频一致：`sample_id, modality, subject_id, session_id, split,
  source_file, window_index, window_start_ms, window_end_ms, duration_ms,
  label_class, label_id, label_scheme, valid, valid_ratio, mask, feature_path,
  feature_shape, feature_dtype, extractor_name, extractor_version, error`。

**实际实现与标准的差异**：

- 公共验证器 `scripts/validate_distraction_audio_metadata.py` 硬编码 9 类；6c 音频
  另用 `validate_distraction_audio_6c.py`（scope
  `structure_and_features_only_6c`），未改动公共验证器。
- 私有历史报告 `reports/audio_metadata_validation_6c_v1.json`（2026-09-18）仍记录
  旧 scheme 串 `dcpt_audio_6c_v1`，与当前 CSV/验证器的 `dcpt_video_6c_v1` 不一致；
  以当前重算结果为准（见「实测结果」）。
- `pyproject.toml` 两处 torch 版本约束不一致（见上）。

## 5. 质量 / 特征 / 实验历史

- metadata：`metadata/audio_windows_10s_v1.csv`，714 行（表头 + 714），全部
  `valid=true`；以现有 **714 条候选记录**为基准，音频与视频候选数一致，无单侧缺失。
  以候选划分（24/8/8 人，seed 2026，`dcpt_subject_24_8_8_seed2026_v1_provisional`）
  填写 `split`；**split 仍为候选，正式签署属于 G5**。
- 特征：`processed/audio_feature_index_v1.jsonl` 714 行，`status=ok` 714、error 0，
  `[5,2048]`；`reports/audio_feature_extraction_summary_v1.json` 记录
  `device=cuda`，feature_version `panns_cnn14_16k_v1`。
- 基线：BiGRU hidden 192、dropout 0.25、mean+max+last pooling、AdamW 3e-4/5e-4、
  batch 32、max_epochs 100、patience 15、seed 11/22/33，val Macro-F1 选点。
  test Macro-F1 逐 seed 0.5953 / 0.5456 / 0.5637，均值 **0.5682 +/- 0.0206**，
  test 样本 144（视频-only v4 同 cohort 为 0.6910 +/- 0.0601）。
- **test 使用历史**：test 标签与预测**仅**用于评测与复现（`evaluation.json`
  重算 Macro-F1 `0.5682041464 +/- 0.0205567034`，`test_sample_count=144`），
  从未用于调参、选点或选择划分。选点依据是 val Macro-F1。
- 分心主指标为原始片段 Macro-F1；报告同时含 balanced accuracy、混淆矩阵、
  每类召回、每驾驶员指标与 majority baseline（majority 类 Messaging，
  test Macro-F1 0.0510）。

## 6. 给陈星宇的核对清单与哈希

受控存储根：`<local>/distraction_audio/`（真实特征/权重私有交付）。以下哈希均为
**本次实际重算**（powershell `Get-FileHash -Algorithm SHA256`）：

| 工件 | 路径（相对受控根） | 大小 | SHA-256（实测） |
| --- | --- | --- | --- |
| 6c metadata CSV | `metadata/audio_windows_10s_v1.csv` | 251,506 B | `A990E403B6558979FA4C2170D65B27FA5CE151FEE216D8CC03BDCF7BA4E57D43` |
| 特征索引 | `processed/audio_feature_index_v1.jsonl` | 201,348 B | `221990FB404925B372B3F1AE8BA49B40C304F1B4D8897278DE03FB27C14C02DD` |
| 标签映射 | `metadata/audio_label_scheme_6c_v1.json` | 567 B | `84CFCA5005ED8AE1D90DCA63789D5AADAB57512C11B24CD047044CA717E0ECA4` |
| 划分清单 | `metadata/audio_subject_splits_6c_v1.json` | 1,219 B | `EE38DB3D325438E859F62862CB73B69FE0624F1EFDF7780C14500D611B0C22EE` |
| 基线评测汇总 | `artifacts/audio_baseline_v1/v1/evaluation.json` | 10,857 B | `9DEC3FC01B81C23012555ABCAF7D66AA3BF4AD23F22FB25F5DA476894740055F` |
| 基线运行汇总 | `artifacts/audio_baseline_v1/v1/summary.json` | 12,503 B | `F34B963021E9BA3B8461885EDF7FA875C1EC272803783C753C8C3382C9A0CC6D` |
| 特征抽取报告 | `reports/audio_feature_extraction_summary_v1.json` | - | `64484A558EE82D003D739087F1BB837AB8DEDB666A25D8E34B031E1C8D5E60A9` |
| 基线结果说明 | `reports/audio_baseline_v1_result.md` | - | `36C4656B37CF66F844510D746E007A7ACBAE9CFC6233FE9A7CCBE38EF233C4E8` |
| PANNs 权重 | `artifacts/pretrained/Cnn14_16k_mAP=0.438.pth` | 358,668,570 B | `E2EE543A27919542C2EA03EABAA70B24DCD4E6C8E05621DE6B67A94E4C5058E6` |

校验方法：`Get-FileHash -Algorithm SHA256`（与 `docs/audio_processing_v1.md` 中
声明的哈希对比一致）。特征 NPZ 逐文件哈希在 `audio_feature_index_v1.jsonl` 内，
抽取时已核对。

## 7. 实测结果（本次提交时重跑）

- 公共测试套件：`python -m pytest` -> **361 passed, 0 failed, 0 error, 0 skipped**。
- 音频子集（6 个测试文件）：**62 passed**。
- 6c 验证器：
  `python scripts/validate_distraction_audio_6c.py --metadata <csv> --label-scheme
  <json> --subject-splits <json> --feature-root <processed>`
  -> `status=PASS`，`scope=structure_and_features_only_6c`，`schema_version=0.2.0`，
  `label_scheme=dcpt_video_6c_v1`，`row_count=714`，`checked_feature_count=714`，
  `min_valid_ratio=0.95`，`errors=[]`，1 条「候选划分待冻结」warning。
- 融合配对：
  `python scripts/check_fusion_pairs.py --audio-csv <audio csv> --video-csv
  metadata/video_reference_v1/video_windows_10s_v1.csv`
  -> `status=PASS`，audio/video 各 714 行、common 714、only_audio 0、only_video 0，
  `label_scheme/split/session/source` 失配全 0，`errors=[]`。

## 8. 接口与字段匹配 vs 时间同步（边界）

- 本节全部结论仅为**字段与结构**匹配：`sample_id`、标签、`subject_id`、`session_id`、
  `split`、`source_file` basename 双侧一致。
- **不是**原始音视频时间同步验证。DCPT 原始时间对齐仍为 pending，P04 人工同步
  **未签署**（见 `docs/video_p04_sync_audit_v1.md`，状态 `pending_manual_signoff`）。
  本文件不修改视频分支、不声称 P04 通过、不用 test 指标调参。

## 9. 未决事项

1. 6c 标签与 24/8/8(seed 2026) 划分待李坤洋冻结；冻结前所有结果标注“候选”。
2. P04 原始音视频同步人工签署仍待完成（G5）。
3. `pyproject.toml` 中 `audio` 与 `train` 的 torch 版本约束需统一。
4. 私有 `reports/audio_metadata_validation_6c_v1.json` 的旧 scheme 串
   (`dcpt_audio_6c_v1`) 建议在下次重算时刷新。
5. 最终合并由李坤洋执行；本阶段不启动新训练。
