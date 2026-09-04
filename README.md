# Multimodal Driver State Detection

面向本科生大创项目的多模态驾驶员状态检测工程。仓库同时支持两个相互独立的任务：

- **疲劳检测**：UL-DD，视频 + CAN，KSS 三分类。
- **分心检测**：DCPT，视频 + 音频，九分类。

当前阶段只建设单模态基线、公共数据接口与双模态 MulT 接入条件；**不使用 MMSA，不引入文本或虚假第三模态，暂不实施轻量化**。仓库尚未包含真实数据处理结果、已训练模型或实验成绩。

## 固定实验口径

| 项目 | 疲劳任务 | 分心任务 |
| --- | --- | --- |
| 数据集 | UL-DD | DCPT |
| 模态 | 视频 + CAN | 视频 + 音频 |
| 标签 | `low=0`：KSS < 4；`medium=1`：4 ≤ KSS < 7；`high=2`：KSS ≥ 7 | 固定九类，编号 0–8 |
| 公共样本 | 30 秒窗口，30 秒步长 | 一个原始名义 10 秒片段 |
| 内部时间结构 | 视频保留 6 个连续 5 秒子窗口 | 音视频保留真实时间支持区间 |
| 主评测 | 240 秒父区间 Macro-F1，同时报告 30 秒指标 | 原始片段 Macro-F1 |

详细约束见 [实验核心规则](docs/EXPERIMENT_RULES.md)；数据和特征格式见 [公共接口](docs/INTERFACES.md)。完整实验规范由项目负责人另行维护，仓库摘要不能覆盖其决定。

## 五人分工

| 成员 | 主要责任 |
| --- | --- |
| 邓金祥 | 疲劳视频预处理、既有成果适配、疲劳视频单模态基线 |
| 胡煦轩 | 分心音频预处理、分心音频单模态基线 |
| 陈星宇 | 分心视频预处理、分心视频单模态基线 |
| 饶棋涛 | 双模态 MulT 修改、简单融合对照与模型接口测试 |
| 李坤洋 | 疲劳 CAN、公共接口与评测、完整实验、仓库管理；项目总负责人 |

模块边界、交付物和 AI 协作方式见 [团队与 AI 指南](docs/TEAM_AND_AI.md)。

## 快速开始

要求 Python 3.10 或更高版本。在已激活的独立虚拟环境中执行；以下命令只安装轻量公共依赖并运行合成测试，不会下载数据或训练模型。

```bash
python -m pip install -e ".[dev]"
python scripts/check_repository.py
python -m pytest
```

验证 metadata 的单行命令（路径需替换为实际位置）：

```bash
python -m driver_state.validation.metadata --task fatigue --metadata path/to/metadata.csv --feature-root path/to/features --report path/to/validation_report.json
```

验证通过返回退出码 0，否则返回 1，并输出 JSON 报告。此工具只验证已实现的结构与特征检查，不能代替真实时间同步、数据来源及 test 使用审计。

## 仓库结构

```text
configs/           # 任务规则和本机路径示例
docs/              # 实验摘要、接口与协作规范
src/driver_state/  # 公共常量、schema 和验证逻辑
scripts/           # 仓库级检查入口
tests/             # 不依赖真实数据的合成测试
manifests/         # 可共享的样本/划分清单说明
artifact_index/    # 大文件版本与校验值索引，不存大文件本体
results/           # 可公开的小型、已核验汇总结果
```

原始视频、音频、CAN、完整特征、模型权重和完整运行包不得提交到 GitHub。路径、账号、令牌和网盘密码也不得写入仓库。详见 [贡献指南](CONTRIBUTING.md)。

## MulT 接入边界

首轮骨架不复制原始 MulT。后续由模型负责人以独立提交引入双模态适配，并保留上游来源和许可证；模型必须消费本仓库定义的特征、时间和 mask 接口。
