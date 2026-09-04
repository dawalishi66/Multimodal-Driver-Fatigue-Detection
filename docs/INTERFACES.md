# 公共接口（schema 0.2.0）

## CSV metadata

每个文件只包含一个任务中的一个模态，UTF-8 编码。疲劳文件推荐名 `metadata/<modality>_windows_30s_v1.csv`；分心使用 `windows_10s`。固定列由 `driver_state.schemas` 提供，允许附加列。

| 字段 | 约定 |
| --- | --- |
| sample_id | 在本模态 CSV 内唯一；同一公共样本的两个模态共享 ID |
| modality | 疲劳 video/can；分心 video/audio |
| subject_id、session_id | 非空；UL-DD 使用 A 等原始字母；DCPT 使用 P01–P40 |
| split | train/val/test，UL-DD 必须匹配固定名单 |
| source_file | 原始相对资源标识，多个资源为 JSON 字符串数组；不能写绝对路径 |
| window_index | 疲劳在父区间内为 0–7；DCPT 原始片段为 0 |
| window_start_ms、window_end_ms、duration_ms | 整数毫秒、左闭右开；duration 等于起止差 |
| label_start_ms、label_end_ms | 仅疲劳必需，所属 240 秒父区间 |
| kss_score | 仅疲劳必需，原始小数，不取整 |
| label_class、label_id | 固定字符串及整数编号；分心字符串使用配置中的英文原名 |
| valid | true/false，兼容 1/0；禁止把字符串 false 当作真 |
| valid_ratio | 可用特征支持区间的并集长度 / 名义样本时长 |
| mask | `相对特征路径.npz::valid_mask`，指向 feature_path 的同一文件 |
| feature_path | 相对于命令参数 feature-root 的 NPZ 路径 |
| feature_shape、feature_dtype | JSON 整数数组如 `[6,128]`，以及 `float32` |
| extractor_name、extractor_version | 可追溯的名称和版本，不能填 latest 或 TO_BE_FILLED |
| error | 有效行为空；失败行保留稳定错误码和简短原因 |

失败且没有特征文件的行：`valid=false`、`valid_ratio=0`，feature_path、mask、feature_shape、feature_dtype 为空，error 必须非空。行保留不代表该样本可以进入训练。

疲劳 CSV 中同一场次的已列窗口必须连续相差 30 秒；中间失败窗口应保留无效行，不能直接删去。小样本冒烟导出连续窗口段，不用过滤后的配对 cohort 代替原始单模态 metadata。验证器仍不能发现未列出的整段开头/尾部，因此候选数守恒审计不可省略。

DCPT 公共 metadata 固定名义 `[0,10000)` 毫秒，音视频不足部分不能伪造为真实观测。用支持范围、observed_fraction 和另行交付的质量 sidecar 记录真实有效起止、补齐量及来源。

## NPZ 特征

标准化输出恰好保存以下五个数组；其他元数据写入 JSON sidecar，不使用 pickle 对象。

| 数组 | 类型 | 形状 |
| --- | --- | --- |
| x | float32 | `[T,D]` |
| time_s | float64 | `[T]` |
| valid_mask | bool | `[T]` |
| support_s | float64 | `[T,2]` |
| observed_fraction | float32 | `[T]` |

- T、D 必须大于 0，不硬编码未知的真实特征维度。
- time_s 是样本相对时间，严格递增且位于对应支持区间内部；support_s 在当前样本范围内。
- valid_mask=True 表示 token 可用。覆盖率按 True token 的支持区间并集计算，重叠不重复计数。
- observed_fraction 表示每个 token 的原始有效比例，范围 0–1；它与特征可用率不同，不因插值或补齐自动变成 1。
- 所有数组有限，不含 NaN/Inf；所有 x（包括无效位置）均须有限。零不是缺失标记。
- CAN 导出器可以从已核验且有说明的目标时间网格推导 support_s，但交给本验证器前必须将结果保存进 NPZ。
- 疲劳视频的 6 个子窗口来源须额外提供索引；本验证器不凭 T=6 宣布六个真实子窗口已对齐。

## 验证器

```bash
python -m driver_state.validation.metadata --task fatigue --metadata path/to/metadata.csv --feature-root path/to/features --report path/to/report.json
```

返回 JSON 包括 status、scope、schema_version、row_count、checked_feature_count、errors、warnings 和 limitations。数据验证通过退出 0，失败退出 1，命令参数错误退出 2。

`--min-valid-ratio` 默认 0.95，只能使用负责人批准的质量版本调整，不得为了过测试随意调低。

PASS **只表示已实现的结构和特征检查通过**。未覆盖真实同步、缺失候选行守恒、六个视频子窗口血缘、complete-8 cohort、跨文件同人划分、train-only 拟合记录、test 使用历史和模态专属质量限制。DCPT 尚无正式名单时只检查文件内被试隔离，并输出警告。完整验收由后续模态验证器和负责人完成。

## 模型与批处理（预留接口，尚未实现）

输入为按模态命名的字典，例如 fatigue 的 video/can；每个模态含 `x[B,T,D]`、`valid_mask[B,T]`、`time_s[B,T]`。右侧补齐 mask=False，内部缺失不能简化为长度。全无效或缺失整模态样本在主实验入口拒绝。

标签为 int64 `[B]`，由训练器使用，不传入特征字典。subject、sample_id、文件名及标签只用于追溯和评测。

模型返回 `{"logits": tensor[B,C]}`，疲劳 C=3、分心 C=9。公共评测接收 sample_id、label、probabilities[C]。现有非 PyTorch 基线可用适配器导出同一概率列顺序，不强制重写模型。
