# G3 视频来源绑定与五秒网格复验（2026-10-10）

邓金祥在 PR #9 head `6b84813` 提出的两项 P2 校验缺口已在代码提交
`f4df004d581652f973de26be389ab7f4fbd48eae` 中修复；本记录是李坤洋的修复与
自测证据，**不能代替邓金祥对新提交的非作者复审**。

1. Dataset 与配对构建器均读取同一视频 handoff ZIP 内的 30 秒窗口表、5 秒来源表和
   特征索引。配对行的视频 sample/member/SHA、subject、session、split、时间、
   KSS 及 240 秒父区间与六个有序 5 秒来源逐项交叉校验。来源索引缺失或不符即失败。
2. 视频特征从 session 相对时间转换后，严格要求六段连续 `[0,5)、…、[25,30)`
   支持区间及中心时刻。该视频专属规则不施加到 CAN。
3. 新增跨驾驶员、跨 session、跨 split、乱序、来源时间／父区间错误及视频支持区间
   间隙、短段、重叠等负例。全仓库 CPU 测试 **299 passed**，仓库内容策略检查 PASS。
4. 对原视频 handoff ZIP 的 1,909 个 30 秒候选核验来源索引：
   **1,909/1,909 PASS**；冻结配对集合 1,600 个样本的真实 NPZ 均通过连续五秒
   网格校验。ZIP SHA-256 仍为
   `00a6a2fdb5596ecfe6ed6fc688f42ad7680f509346b3e10b1ebef3de6a59b4e5`。
5. 冻结 complete-8 清单未修改，SHA-256 仍为
   `201066ddb6bccb39f63db6a115662f0d6b5c33c55d47673374d143b04f646e7b`。
   全量 Dataset 复验 train 1,272 窗口／159 父区间、val 328／41，来源绑定、视频
   网格、逐文件哈希及 batch 冒烟全部 PASS。
6. 在该干净提交上重跑 SimpleFusion 和 DualModalMulT 的真实特征小批量预检，均 PASS；
   没有正式训练或计算新成绩，`test_manifest_accessed=false`。

新的完整验收报告与预检报告均保留在外部 `Paired_Video_CAN_v1` 目录，
相对位置及 SHA-256 见 `artifact_index/fatigue_video_can_g3_video_binding_20261010.json`。
原始数据、旧清单、既有模型权重和运行包均未修改。正式 test 继续锁定。

**待办：** PR #9 最新 head 的 CI 通过后，请邓金祥针对修复提交重新审核视频来源、
五秒网格与冻结清单。其 Request changes 未撤销前不得合并 `main`。
