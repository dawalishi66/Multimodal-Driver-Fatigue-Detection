# 配置

两份任务 JSON 固化已确认规则，但 `status=preflight_required` 表示它们不是可直接启动正式训练的完整配置。没有虚构特征维度、编码器或冻结 cohort。

`simple_fusion_v1.example.json` 和 `dual_modal_mult_v1.example.json` 是模型配置模板，不是正式实验配置。每个文件的 fatigue 与 distraction profile 必须择一解析，不能在一个模型实例中混合两个任务；正式运行前必须由 verified feature manifest 提供实际输入维度。模板中的候选参数不代表真实特征已经核验、正式实验配置已经冻结或参数已经达到最优。

将 paths.example.json 复制为 paths.local.json 后填写本机路径；后者被 Git 忽略。源代码使用配置解析路径，不硬编码成员电脑位置。

`can_preprocessing_v1.json` 固化 UL-DD CAN 的 10 Hz 特征、质量阈值和时钟异常策略。

`fusion/fatigue_video_can/g2_preflight_v1.json` 绑定已验收的 UL-DD 视频＋CAN
complete-8 train/validation共同集合，供真实特征G2冒烟使用。它不包含test入口，也不是
正式训练配置。
