# 配置

两份任务 JSON 固化已确认规则，但 `status=preflight_required` 表示它们不是可直接启动正式训练的完整配置。没有虚构特征维度、编码器或冻结 cohort。

将 paths.example.json 复制为 paths.local.json 后填写本机路径；后者被 Git 忽略。源代码使用配置解析路径，不硬编码成员电脑位置。
