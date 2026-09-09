# GitHub 仓库协作与 AI 开发总规范

> 版本：v1.1
>
> 适用仓库：`dawalishi66/Multimodal-Driver-Fatigue-Detection`
>
> 适用人员：邓金祥、胡煦轩、陈星宇、饶棋涛、李坤洋
>
> 维护人：李坤洋
> 本文用于团队成员学习 GitHub、日常协作、代码审核和给 AI 提供开发背景。

## 1. 文档怎么使用

第一次参与仓库开发时，按顺序阅读：

1. 本文第 2—8 节，完成账号、仓库和本地 Git 配置。
2. 开始任务前阅读第 9—12 节，按 Issue、分支、提交和 PR 流程操作。
3. 负责人和审核人使用第 13—15 节进行审核、合并和版本管理。
4. 让 AI 协助写代码时，复制第 17 节的提示词并填写方括号内容。

项目规则的优先级如下：

1. 李坤洋维护的《驾驶员状态检测项目实验总规范》当前正式版本；
2. `docs/EXPERIMENT_RULES.md` 和 `docs/INTERFACES.md`；
3. 本文及 `CONTRIBUTING.md`；
4. 模块配置、Issue 和 PR 中的任务说明；
5. 成员或 AI 的临时建议。

发现冲突时停止修改并报告，不能自行选择更容易得到高分的规则。

## 2. 团队采用的协作方式

本项目固定采用以下方式：

- 仓库保持公开，但不上传原始数据、完整特征、权重和隐私内容。
- 五人加入同一个仓库成为 Collaborators，不采用五个 Fork。
- 每项工作先建立 GitHub Issue，再从最新 `main` 创建短期任务分支。
- 普通成员禁止直接向 `main` 提交代码，修改必须通过 Pull Request（PR）。
- 普通 PR 至少由一名非作者审核；李坤洋检查 CI 和实验边界后统一合并。
- 李坤洋作为仓库管理员可以直接推送或独立合并，用于仓库设置、紧急修复或明确需要快速落地的负责人操作；日常功能开发仍优先使用 PR，并记录绕过原因和实际检查结果。
- 公共接口和融合模型需要额外的受影响模块负责人参与审核。
- 日常 Git 操作使用命令行，Issue、PR、审核和合并使用 GitHub 网页。
- `main` 只保存已经审核、测试通过且说明完整的代码。

## 3. 五人责任和审核关系

| 成员 | 主要责任 | 主要代码区域 | 普通审核人 |
| --- | --- | --- | --- |
| 邓金祥 | 疲劳视频预处理、适配和单模态基线 | `preprocessing/fatigue_video`、疲劳视频基线 | 李坤洋 |
| 胡煦轩 | 分心音频预处理和单模态基线 | `preprocessing/distraction_audio`、分心音频基线 | 陈星宇 |
| 陈星宇 | 分心视频预处理和单模态基线 | `preprocessing/distraction_video`、分心视频基线 | 胡煦轩 |
| 饶棋涛 | Simple Fusion、MulT、模型接口和消融 | `models/simple_fusion`、`models/mult` | 李坤洋和至少一名上游模态负责人 |
| 李坤洋 | CAN、公共接口、训练评测、完整实验和仓库管理 | CAN模块及 `data`、`engine`、`evaluation`、`validation` | 邓金祥或实际使用该接口的成员 |

表中的代码区域相对于 `src/driver_state/`。成员可以为完成本模块任务修改测试、配置和文档，但不得顺手重构其他成员模块。

以下内容属于公共合同，修改前必须建立带 `interface-change` 标签的 Issue：

```text
src/driver_state/constants.py
src/driver_state/schemas.py
src/driver_state/data/
src/driver_state/engine/
src/driver_state/evaluation/
src/driver_state/validation/
docs/EXPERIMENT_RULES.md
docs/INTERFACES.md
公共标签、窗口、split、metadata和特征配置
```

公共合同 PR 必须说明旧接口、新接口、受影响模块、迁移方式和版本变化。李坤洋与至少一名受影响模块负责人确认后才能合并。

## 4. GitHub 中保存什么

### 4.1 可以提交

- 源代码、命令行工具和模块适配器；
- 机器无关的配置和依赖说明；
- metadata、NPZ、batch 和模型接口定义；
- 不依赖真实数据的小型合成测试；
- 数据审计和实验结果的脱敏汇总；
- 特征、权重和运行包的 `artifact_index`；
- 运行方法、失败项、已知限制和许可证来源。

### 4.2 禁止提交

- 原始视频、音频、CAN 和数据集压缩包；
- 完整 NPZ/Numpy 特征缓存；
- `.pt`、`.pth`、checkpoint、ONNX 等模型文件；
- 完整预测明细和完整实验运行包；
- 人脸、声音、姓名、联系方式等隐私内容；
- 本机绝对路径、网盘密码、临时下载链接、Token 和密钥；
- 未经许可复制的第三方代码或权重。

大型内容保存在本地数据目录、学校服务器或团队私有网盘。GitHub 只保存相对位置、版本、大小、样本数和 SHA-256。不要通过 Git LFS 绕过本项目的数据管理规则。

## 5. Git 和 GitHub 基础概念

| 名称 | 含义 |
| --- | --- |
| Repository/仓库 | 项目代码及其历史记录 |
| Local/本地 | 当前成员电脑上的仓库副本 |
| Remote/远端 | GitHub 上的仓库，默认名称为 `origin` |
| `main` | 稳定主分支，只接收审核后的 PR |
| Branch/分支 | 为一个具体任务创建的独立开发线 |
| Commit/提交 | 一次可追溯的代码快照 |
| Pull | 从 GitHub 获取更新 |
| Push | 将本地提交上传到 GitHub |
| Issue | 有负责人和验收条件的任务或问题 |
| Pull Request/PR | 请求审核并将分支合并进 `main` |
| Review | 对 PR 进行评论、批准或要求修改 |
| CI/Actions | GitHub 自动执行仓库检查和 CPU 测试 |
| Tag/Release | 对重要稳定版本建立不可变标识和说明 |

## 6. 负责人首次设置 GitHub

### 6.1 邀请成员

1. 收集其他四人的 GitHub 用户名。
2. 打开仓库网页。
3. 进入 `Settings → Collaborators → Add people`。
4. 逐一邀请，确认成员已经接受。

五人不得共用一个 GitHub 账号。每人应开启两步验证，并在 GitHub `Settings → Emails` 中取得自己的 `noreply` 邮箱，避免公开私人邮箱。

### 6.2 保护 main

进入 `Settings → Rules → Rulesets`。如果没有该入口，使用 `Settings → Branches → Add branch protection rule`。匹配分支填写 `main`，启用：

- Require a pull request before merging；
- Required approvals：1；
- Dismiss stale approvals when new commits are pushed；
- Require status checks to pass；
- 选择工作流 `lightweight-checks` 的 `test` 检查；
- Require branches to be up to date before merging；
- Require conversation resolution before merging；
- 禁止 force push 和删除 `main`；
- 规则不强制管理员执行。当前唯一管理员李坤洋可以直接推送或独立合并；其他成员仍必须满足保护规则。

如果暂时无法选择 `test`，先创建一个 PR 并等待 Actions 运行一次，再回到规则中选择。

### 6.3 合并设置

进入 `Settings → General → Pull Requests`：

- 保留 `Allow merge commits`；
- 暂时关闭 Squash merging 和 Rebase merging；
- 开启 Automatically delete head branches；
- 暂不开启自动合并。

### 6.4 标签和里程碑

建议建立模块标签：

```text
fatigue-video
fatigue-can
distraction-audio
distraction-video
fusion
common-interface
```

建议建立管理标签：

```text
blocked
needs-review
interface-change
data-contract
results
```

建议建立里程碑：

```text
Phase 1：单模态基线
Phase 2：双模态融合
Phase 3：轻量化与部署
```

## 7. 每名成员首次配置本地仓库

以下命令适用于 Windows Miniconda Prompt 或 cmd。每台电脑只需执行一次克隆：

```bat
cd /d "<你选择的项目父目录>"
git clone https://github.com/dawalishi66/Multimodal-Driver-Fatigue-Detection.git
cd Multimodal-Driver-Fatigue-Detection
```

只为当前仓库设置身份：

```bat
git config user.name "自己的GitHub用户名"
git config user.email "从GitHub Settings → Emails复制的noreply邮箱"
```

确认配置和远端：

```bat
git config user.name
git config user.email
git remote -v
git status
```

不得把 GitHub 密码、访问 Token 或网盘密码写入代码、配置和聊天截图。

## 8. Issue：先定义任务再写代码

一项 Issue 对应一个可以独立验收的交付，不把某个人几个月的全部工作写成一个 Issue。

Issue 推荐模板：

```markdown
## 目标
[本次要解决的一个明确问题]

## 负责人和审核人
- 负责人：
- 审核人：

## 输入
- 数据集/特征版本：
- 输入路径或公共接口：
- 依赖的Issue/PR：

## 允许修改范围
- 允许：
- 禁止：

## 输出
- 代码/配置/文档：
- 外部artifact及索引：

## 验收
- [ ] 仓库检查通过
- [ ] CPU测试通过
- [ ] 真实小样本检查完成或说明未执行原因
- [ ] 未使用test调参
- [ ] 未上传数据、权重、隐私和绝对路径

## 实验规范影响
- 是否修改标签、窗口、split、metadata、特征或模型接口：否/是（说明）
```

## 9. 分支规则

### 9.1 命名

分支名包含类型、Issue 编号和任务：

```text
feat/12-distraction-audio-panns
feat/15-fatigue-video-adapter
feat/18-dual-modal-mult
fix/21-audio-mask-validation
docs/25-feature-interface
results/31-distraction-audio-baseline-v1
```

禁止使用姓名、`final`、`new`、`test2`、`最新版`作为分支名。

### 9.2 创建原则

- 每个新任务都从最新 `main` 创建新分支。
- 一条分支只完成一个 Issue，不混入无关格式化和重命名。
- 不建立长期个人分支；PR 合并后删除任务分支。
- B 依赖 A 时，优先先合并 A，再从更新后的 `main` 开始 B。
- 已经形成依赖分支时，先合并父分支，再将最新 `main` 合入子分支。
- 禁止向共享分支 force push，团队默认不使用 rebase 改写历史。

## 10. 日常开发命令

### 10.1 开始任务

假设 Issue 是 `#12`：

```bat
cd /d "<你的仓库目录>"
git switch main
git pull --ff-only origin main
git switch -c feat/12-distraction-audio-panns
git status
```

如果 `git pull --ff-only` 失败，不要改用 force 或 reset，先检查本地是否误在 `main` 上提交了内容并联系李坤洋。

### 10.2 开发过程中查看变化

```bat
git status
git diff
```

`git status` 用于确认当前分支和改动文件；`git diff` 用于查看尚未加入提交区的具体变化。

### 10.3 提交前检查

```bat
python scripts/check_repository.py
python -m pytest
```

涉及真实数据的小样本检查在本地执行，并把实际命令、数据版本和结果写入 PR；不能要求 GitHub Actions 下载数据或训练模型。

只添加本次任务文件：

```bat
git add src/driver_state/preprocessing/distraction_audio
git add tests
git add configs
git diff --cached
```

确认无误后提交：

```bat
git commit -m "feat: add DCPT audio feature extractor"
git push -u origin feat/12-distraction-audio-panns
```

团队默认不使用 `git add .`，避免把数据、临时文件和无关修改一起提交。

### 10.4 提交信息

格式统一为：

```text
type: concise summary
```

常用类型：

```text
feat: 新功能、数据管线或模型
fix: 修复错误
test: 增加或修正测试
docs: 只修改文档
refactor: 不改变行为的结构调整
results: 登记已核验实验结果
chore: 仓库和依赖维护
```

一次提交表达一个完整意图。不要写 `update`、`改了一下`、`final`。

## 11. 创建、审核和更新 PR

### 11.1 作者创建 PR

1. 打开 GitHub 仓库，进入 `Pull requests → New pull request`。
2. `base` 选择 `main`，`compare` 选择自己的任务分支。
3. 标题使用提交类型，例如 `feat: add DCPT audio feature extractor`。
4. 按仓库 PR 模板填写负责人、修改范围、输入输出、验证和限制。
5. 在正文写 `Closes #12` 关联 Issue。
6. 选择对应审核人并等待 CI。

PR 必须如实写明：

- 实际运行过的命令和输出；
- 未运行的检查及原因；
- 数据、特征、split、代码和权重版本；
- 是否读取 test；
- 是否修改公共接口；
- 失败项和仍未确认的问题。

### 11.2 审核人操作

1. 打开 PR 的 `Files changed`。
2. 逐文件检查，并将已经看完的文件标记为 Viewed。
3. 检查实现是否超出 Issue 范围。
4. 检查标签、split、时间、mask、特征维度和类别顺序。
5. 检查测试是否真的覆盖新行为。
6. 点击 `Review changes`：
   - `Comment`：提出非阻塞问题；
   - `Approve`：同意合并；
   - `Request changes`：存在必须修复的问题。

AI 给出的审查意见只能作为参考，审核人本人必须看代码和检查结果。

### 11.3 作者修改 PR

审核后继续在原分支修改，不新建第二个 PR：

```bat
git add 具体文件
git diff --cached
git commit -m "fix: address review feedback"
git push
```

PR 会自动更新，CI 会重新运行。新提交会使旧批准失效时，需要审核人重新确认。

### 11.4 负责人合并

普通 PR 由李坤洋确认：

- 至少一名非作者已经批准；
- CI 全部通过；
- 所有阻塞对话已经解决；
- 分支基于最新 `main`；
- 没有数据、权重、隐私、绝对路径或密钥；
- 没有未经批准修改公共实验规则；
- 结果和实际执行范围表述准确。

随后点击 `Merge pull request → Confirm merge`。普通成员不得通过命令行绕过分支保护。

李坤洋可以使用管理员权限独立合并，必要时也可以直接推送到 `main`。使用该例外时，应在 PR、Issue 或提交说明中记录原因，并如实记录 CI、仓库检查和测试是否执行；管理员权限不表示可以绕过实验规范、数据安全或结果真实性要求。

### 11.5 合并后清理

成员本地执行：

```bat
git switch main
git pull --ff-only origin main
git branch -d feat/12-distraction-audio-panns
```

只有 Git 确认分支已经合并后才使用 `-d`。不要使用 `-D`强制删除未合并工作。

## 12. 同步 main、冲突和误操作

### 12.1 其他 PR 合并后更新任务分支

```bat
git switch main
git pull --ff-only origin main
git switch feat/自己的任务分支
git merge main
python scripts/check_repository.py
python -m pytest
git push
```

### 12.2 处理冲突

先运行：

```bat
git status
```

冲突文件中会出现三段 Git 标记：当前分支内容的开始标记、双方内容的分隔标记和传入分支内容的结束标记。

理解双方修改后保留正确内容，删除冲突标记，然后运行：

```bat
git add 冲突文件
git commit
python -m pytest
git push
```

如果冲突涉及公共接口、标签、split 或时间映射，不允许凭感觉选择一边，必须由李坤洋和对应负责人共同确认。

### 12.3 在 main 上误改代码

- 以下规则适用于普通成员。李坤洋经明确判断可以直接维护 `main`，但日常功能开发仍建议从任务分支开始。
- 尚未提交：不要 push，立即停止；可以先用 `git status` 和 `git diff` 保存现场并联系负责人。
- 已经提交但未 push：不要 reset、不要继续 push，联系李坤洋处理。
- 已经 push：立即报告，不要用 force push 隐藏历史。

### 12.4 误提交数据或密钥

- 未 push：停止操作并联系负责人清理提交。
- 已 push：立即报告；密钥要立刻撤销或轮换。仅在新提交中删除文件不能从历史中消除泄漏。
- 禁止成员自行重写共享历史。

团队成员和 AI 均不得自行执行 `git reset --hard`、`git clean -fd`、`git checkout --`或共享分支 force push。

## 13. 三类 PR 和正式实验流程

### 13.1 代码 PR

只提交实现、配置、测试和说明，不混入正式结果。真实小样本运行证据写入 PR，但真实样本和运行包不上传。

### 13.2 数据合同 PR

冻结标签、窗口、被试 split、样本集合、时间映射、特征版本和质量规则。必须使用 `data-contract` 标签，并由李坤洋和受影响负责人审核。

### 13.3 结果 PR

正式训练必须使用已经合并的干净 `main`。结果 PR 只登记脱敏汇总和 artifact 索引，不同时修改模型训练逻辑。

正式实验顺序：

```bat
git switch main
git pull --ff-only origin main
git status
git rev-parse HEAD
```

要求 `git status` 显示工作区干净。将 `git rev-parse HEAD` 的输出写入 `run_manifest.json`，然后运行训练。完成并独立核验后建立结果分支：

```bat
git switch -c results/31-distraction-audio-baseline-v1
git add results
git add artifact_index
git diff --cached
git commit -m "results: record distraction audio baseline v1"
git push -u origin results/31-distraction-audio-baseline-v1
```

## 14. 外部数据、特征、权重和实验包

正式外部存储建议结构：

```text
driver_state_project_data/
├── raw/                   # 只读原始数据
├── processed/             # 按数据集/模态/特征版本保存
├── manifests/             # 完整逐样本清单
├── experiment_runs/       # 权重、预测和完整运行包
└── checksums/             # SHA-256清单
```

每个共享特征包必须同时包含：

```text
features/
feature_index.jsonl
metadata文件
subject split文件
extraction_config.json
validation_report.json
README.md
SHA256SUMS.txt
```

GitHub 中的 artifact 索引至少记录：

```json
{
  "artifact_name": "dcpt_audio_panns_v1",
  "storage": "team_private_storage",
  "relative_path": "processed/DCPT/audio/panns_v1",
  "sample_count": 1080,
  "feature_version": "panns_cnn14_16k_v1",
  "sha256": "实际校验值"
}
```

仓库中不保存网盘密码、个人绝对路径或会过期的签名 URL。下载方式通过团队私有渠道发送。

## 15. 版本、发布和负责人检查

代码里程碑建议：

```text
v0.1.0  工程骨架
v0.2.0  单模态基线完成
v0.3.0  双模态简单融合和MulT完成
v0.4.0  轻量化和部署实验完成
v1.0.0  项目最终冻结交付
```

Git 标签不替代数据和特征版本。每次实验同时记录 Git commit、manifest、split、feature、quality 和 config 版本。

李坤洋每周检查：

1. 打开的 Issue、PR 和 blocked 项；
2. CI 失败和长期无人审核的 PR；
3. 公共接口是否出现未批准变更；
4. 是否混入数据、权重、隐私和绝对路径；
5. 数据合同和正式实验是否对应精确版本；
6. 已合并分支是否清理；
7. Milestone 进度是否真实；
8. 公开 README 和仓库描述是否把计划误写为已完成成果。

## 16. 提交与审核速查表

### 成员提交前

```text
[ ] 当前不是main（李坤洋明确使用管理员直推时除外）
[ ] 分支来自最新main
[ ] 修改只对应一个Issue
[ ] git diff已经亲自检查
[ ] 仓库检查和pytest通过
[ ] 真实小样本验证范围已记录
[ ] 没有使用test调参
[ ] 没有数据、权重、隐私、密钥和绝对路径
[ ] 失败项和未验证项已如实说明
```

### 审核人批准前

```text
[ ] 逐文件看过修改
[ ] 输入输出和版本明确
[ ] 接口与实验规范一致
[ ] 新行为有测试
[ ] 报告与实际证据一致
[ ] 没有把模拟测试写成真实实验
[ ] 没有把单模态扩展集合写成公平融合对比集合
```

### 负责人合并前

```text
[ ] 非作者审核通过（李坤洋使用管理员独立合并时记录例外原因）
[ ] CI通过
[ ] 对话已解决
[ ] 分支已更新到最新main
[ ] 公共规则修改已获批准
[ ] 数据和大文件策略通过
[ ] PR标题、Issue和Milestone正确
```

## 17. 交给 AI 的完整提示词

复制以下内容给 AI，并填写所有方括号。不要只发送“帮我写代码”。

```text
你正在协助开发 Multimodal-Driver-Fatigue-Detection 仓库。

一、开始前必须读取
1. README.md
2. CONTRIBUTING.md
3. docs/GITHUB_WORKFLOW.md
4. docs/EXPERIMENT_RULES.md
5. docs/INTERFACES.md
6. docs/TEAM_AND_AI.md
7. 团队提供的《驾驶员状态检测项目实验总规范》当前正式版本

如果完整实验规范没有提供，必须说明缺失，不得猜测其中尚未写入仓库的决定。

二、固定项目背景
- 疲劳任务：UL-DD，视频+CAN，KSS三分类：KSS<4为0，4<=KSS<7为1，KSS>=7为2。
- 疲劳公共样本：30秒窗口、30秒步长；视频内部保留6个连续5秒子窗口；KSS父区间为240秒。
- 分心任务：DCPT，上半身视频+音频，固定九类0—8；一个原始名义10秒片段是一个样本，不增加滑动窗口。
- 当前不使用MMSA，不创建文本或虚假第三模态。
- 当前先完成单模态和基础双模态，不擅自开始轻量化。
- 单模态负责人不在自己的任务中实现MulT；MulT由指定负责人接入标准特征。
- 原始数据、完整特征、模型权重和完整运行包不进入GitHub。

三、本次任务
- 负责人：[姓名]
- 仓库本地路径：[路径]
- 当前Issue：[编号和链接]
- 当前分支：[分支名]
- 目标：[一个明确目标]
- 输入数据/特征版本：[版本]
- 允许修改：[目录或文件]
- 禁止修改：[目录或文件]
- 输出：[代码、配置、文档、索引]
- 验收命令和预期行为：[具体内容]
- 是否允许读取真实数据：[是/否及范围]
- 是否允许读取test：[默认否]
- 是否允许提交、push或创建PR：[分别明确，默认均否]

四、工作规则
1. 先只读检查git status、当前分支、相关代码、配置、测试和现有修改。
2. 保留用户已有修改，不覆盖无关内容。
3. 本次只完成Issue范围，不顺手重构其他模块。
4. 不擅自修改标签、窗口、split、metadata、时间映射、质量门槛和公共特征接口。
5. 如确需修改公共接口，先报告影响并等待负责人决定；不能为了让测试通过而降低质量要求。
6. 不根据test成绩调整模型、阈值、特征或数据筛选。
7. 失败文件必须保留错误记录，不能静默跳过或伪造成功。
8. 不上传原始数据、特征、权重、隐私、绝对路径、密钥和网盘信息。
9. 不复制来源和许可证不明的第三方代码或权重。
10. 不执行git reset --hard、git clean -fd、git checkout --、force push或删除他人分支。
11. 没有明确授权时，不commit、不push、不创建PR、不合并。
12. 先用合成或10—50个真实样本冒烟，再考虑批量处理；不得把冒烟结果称为正式实验。

五、必须验证
- 运行与修改范围相称的单元测试。
- 运行 python scripts/check_repository.py。
- 条件允许时运行 python -m pytest。
- 检查metadata、特征形状、dtype、NaN/Inf、mask、时间、ID、标签和split。
- 如果没有运行某项检查，明确说明原因，不得写成PASS。

六、最终报告格式
1. 完成内容；
2. 修改文件；
3. 输入输出和版本；
4. 实际运行的命令与结果；
5. 未运行或失败的检查；
6. 是否访问test；
7. 是否产生外部数据、特征、权重或运行包及其位置；
8. 是否影响公共接口；
9. 已知限制和下一步；
10. 建议的commit信息和PR说明，但没有授权时不要实际提交或push。
```

## 18. 官方参考

- [邀请个人仓库协作者](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)
- [受保护分支](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches)
- [审核 Pull Request](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/reviewing-proposed-changes-in-a-pull-request)
- [使用 Issues 管理工作](https://docs.github.com/en/issues/tracking-your-work-with-issues/learning-about-issues/planning-and-tracking-work-for-your-team-or-project)
