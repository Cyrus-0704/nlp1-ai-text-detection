# NLP-1 大模型生成文本检测

三人协作的初赛项目。仓库按发起人要求公开，任何人都能查看和下载；公开不代表任何人都有推送权限。职责与共享接口见 [TEAM_WORKPLAN.md](TEAM_WORKPLAN.md)。每人只更新自己的任务清单和工作代码，不必共同维护一份进度表。

| 人员 | 任务文档 | 主要交付 |
| --- | --- | --- |
| A：数据与评估 | [A 的任务清单](docs/tasks/A_DATA_EVALUATION.md) | 清洗数据、固定划分、版本与统一评估 |
| B：主模型（本仓库发起人） | [B 的任务清单](docs/tasks/B_MAIN_MODEL.md) | XLM-R 主模型、预测、模型包 |
| C：基线与报告 | [C 的任务清单](docs/tasks/C_BASELINE_REPORT.md) | 字符基线、融合与报告 |

## 获取项目与数据

```powershell
git lfs install
git clone https://github.com/Cyrus-0704/nlp1-ai-text-detection.git
cd nlp1-ai-text-detection
git lfs pull
```

原始 `data.jsonl` 约980 MB，使用 Git LFS。公开仓库可直接克隆。若已经从官方获取同一数据，或暂时只要代码，可先跳过大文件下载：

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone https://github.com/Cyrus-0704/nlp1-ai-text-detection.git
Remove-Item Env:\GIT_LFS_SKIP_SMUDGE
cd nlp1-ai-text-detection
git lfs install --local --skip-smudge
git lfs pull --include="data.jsonl"
```

最后一行只在需要原始数据时运行；skip-smudge让日常拉取先更新指针，不自动下载所有大文件。后续按需获取 A 的版本目录，例如 `git lfs pull --include="data/processed/v1/**"`。数据版本和校验方式见 [data/README.md](data/README.md)。

## 本机环境独立管理

每人在自己电脑创建环境，路径和名称自行选择。安装后的 Python/Conda、site-packages、缓存和本机配置不上传。仓库里的 `requirements/` 只是可共享的依赖版本说明，不是任何人的环境副本。

- A/C 可先安装基础依赖：`python -m pip install -r requirements/base.txt`。
- B 的已验证 CUDA 路线：先按官方命令安装适配本机驱动的 PyTorch，再安装 `requirements/train.txt`。当前验证版本为 PyTorch 2.13.0+cu126；GPU 可用性需每台电脑单独检查。
- 代码使用当前已激活环境的 `python`；不要把其他成员的盘符、用户名或解释器绝对路径写进共享代码。
- 预训练权重不入库，通过 [main_model/README.md](main_model/README.md) 的固定版本下载命令各自获取。

## 日常协作

已获写权限的队员使用自己的工作分支，例如 `work/a-data`、`work/b-model`、`work/c-baseline`。首次创建：

```powershell
git switch main
git pull --ff-only
git switch -c work/a-data
```

以 A 为例，完成一段工作后只暂存本人的交付路径，并先检查文件清单：

```powershell
git status --short
git add docs/tasks/A_DATA_EVALUATION.md data/processed/v1 data/manifest.raw.json
git diff --cached --stat
git commit -m "A: deliver versioned internal data splits"
git push -u origin work/a-data
```

按实际已存在的文件调整 `git add` 参数；不要直接套用不存在的目录。通过 Pull Request 合并到 main，接收方再拉取 main 和需要的 LFS 文件。未合并分支的结果不会自动出现在 main。不要 force-push 覆盖别人的提交。

尚无写权限的队员可先Fork到自己的GitHub账号，在自己的副本建分支、推送，再向本仓库main提PR。若需直接推送团队分支，提供GitHub用户名，由仓库所有者添加协作者；仅把仓库公开不会自动赋予写权限。

## 什么该上传

| 上传并协作 | 留在各自电脑 |
| --- | --- |
| 源代码、个人任务清单、共享接口说明 | 已安装环境、解释器、Conda目录和依赖缓存 |
| 确认的原始数据、A冻结的数据版本与manifest | 各自重新生成的临时数据、缓存和调试输出 |
| 接收方需要的预测、配置、指标摘要和最终模型 | 下载的官方预训练模型、个人实验目录和全部中间checkpoint |
| 报告源稿、对外声明、必要的依赖版本说明 | 本机绝对路径、IDE设置、凭据、访问令牌和本机安装日志 |

每次交接把“对方实际需要”的结果整理到 `handoffs/A/`、`handoffs/B/`、`handoffs/C/`；只有需要共同使用的模型才放到 `handoffs/shared_models/<版本>/`，通过 LFS 上传。`runs/`、`models/`、`environment/` 默认保持本地。原始数据不重复拷贝成多个版本。

LFS 会使用仓库所有者的存储和下载额度；大文件只在必要版本交付时上传，不把每个训练checkpoint都推上云端。官方说明：[普通Git文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage)。

官方验证集仅用于方案冻结后的最终推理，不能探查内容、统计分布或参与调参。提交组委会的材料须另行匿名化，不把 `.git` 和团队协作文档混进比赛模型包。
