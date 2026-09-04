# 团队与 AI 开发入口

## 模块边界

| 成员 | 主要代码位置（待逐步实现） | 输入与交付 |
| --- | --- | --- |
| 邓金祥 | preprocessing/fatigue_video；models/unimodal/fatigue_video | 现有视频成果 → 5 秒子窗口索引、30 秒标准特征、视频基线 |
| 胡煦轩 | preprocessing/distraction_audio；models/unimodal/distraction_audio | 原始音频 → DCPT 片段特征、音频基线 |
| 陈星宇 | preprocessing/distraction_video；models/unimodal/distraction_video | 原始视频 → DCPT 片段特征、视频基线 |
| 饶棋涛 | models/mult；models/simple_fusion | 标准双模态 batch → logits、模型测试和消融配置 |
| 李坤洋 | preprocessing/fatigue_can；models/unimodal/fatigue_can；data、engine、evaluation、validation | CAN 特征与基线、公共加载/训练/评测、完整实验与仓库管理 |

以上路径相对于 `src/driver_state/`。首轮目录只是责任槽位，不代表相应处理器、训练器或模型已经完成。

每个单模态模块交付配置、metadata、特征卡、审计报告、单模态基线、标准特征索引、验证器薄封装和真实小样本检查。大型文件不入库，版本与校验值进入 artifact_index。

音视频负责人共同核验 DCPT 时间原点和配对；李坤洋与邓金祥共同核验 UL-DD 时间与标签映射。接口相同是必要条件，不是时间同步的充分证据。

## 交给 AI 的最小背景

```text
先读取 README.md、docs/EXPERIMENT_RULES.md、docs/INTERFACES.md 和 CONTRIBUTING.md。
另外完整读取团队提供的实验总规范 v0.2；若没有拿到，先说明缺少完整版，不能自行推断缺失决定。

本次负责人：[姓名]
本次任务：[一个明确模块或问题]
现有代码/数据位置：[本次对话提供，不提交本机路径]
允许修改范围：[明确目录]
输入/输出：[引用公共接口和版本]
验收：[应运行的测试与预期行为]

只实现本次任务，不擅自改标签、窗口、split、评测或公共接口。
不使用 MMSA，不制造第三模态，不开始轻量化。
单模态任务不实现融合或 MulT；已有疲劳视频成果先检查再适配。
不要猜特征维度、时间映射或权重来源；无法确认时集中报告。
不上传原始数据、特征、权重、密钥；不删除用户已有内容。
不得编造运行记录、指标或成功状态。完成后报告文件改动、已运行检查、失败项和剩余限制。
```

成员提交前必须亲自查看修改范围和检查结果。AI 不能代替团队批准公共接口变更，不能以“为了跑通”跳过失败样本或测试。
