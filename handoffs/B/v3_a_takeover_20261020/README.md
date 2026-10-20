# A/seed43 任务接管交付（2026-10-20）

本次按用户明确调整后的分工，将 A 的三项实验交给 C、B 的现有主机接续执行。交付阶段只准备文件，不启动 preflight、pilot、正式训练或后台排程；未来由执行者手动使用 `--execute`，且前置实验全部实际完成后才运行。

## 接管顺序与实验归属

| 实际执行者 | 必须先完成的本人实验 | 随后接管的实验 | 接管时固定参数 |
| --- | --- | --- | --- |
| C | C/seed44 的 T0 → T1 → T2 全部完成 | A/seed43 的 T0 | `--owner A --arm T0` |
| B | B/seed42 的 T0、T1、T2 全部完成，最后一项为 B/T2 | A/seed43 的 T1 → T2 | `--owner A --arm T1`，随后 `--owner A --arm T2` |

“完成”须核对原工作区各项正式实验的 `execution.json`，状态为 `COMPLETE_LOCAL_DEV_NOT_INDEPENDENT_EFFECT`，并有相应训练、导出、独立重载和 dev 评估产物。仅 GPU 空闲、已有 7500 步 checkpoint、某个子检查通过或时间预算耗尽，均不满足前置条件。

两条接管队列分别遵守上述前置条件，不新增跨主机的等待关系。A 的各 arm 都从固定原始预训练模型独立初始化；A/T1、A/T2 不读取 A/T0 或 B/C 已训练的权重。

逻辑 owner 始终为 **A**，seed 始终为 **43**。实际执行者和主机另行记录，不将 C 执行的 A/T0 改称 C/seed44，也不将 B 执行的 A/T1、A/T2 改称 B/seed42。C、B 各自更新自己的任务清单，记录本人接管操作与产物，不代改 A 的进度。

## 交付文件与冻结身份

- 共用入口：本目录的 `run_takeover.py`。
- 冻结代码包：`v3-code-20261020.zip`，大小 **239474 字节**。
- 代码包 SHA-256：`67d98d34411215c3a91b1c781e344c913e80eaf8ffd7e2cb2f4a9b0e968ac4b7`。
- 包内 `main_model/configs/v3/package_manifest.json` 的 SHA-256：`56168d5f005d2cfa310660d5736dca6380afc38bc1c36342cd6b9ef952c8ea36`。
- GitHub prerelease 下载：[本次接管文件与冻结代码包](https://github.com/Cyrus-0704/nlp1-ai-text-detection/releases/tag/v3-a-seed43-takeover-20261020)。下载文件统一保存到执行者 E 盘的 `E:\Codex\transfer\v3-a-takeover`；该页面只表示文件交付，不表示任务已启动。

复用同一冻结代码包，不修改原工作区的训练源码、现有代码包或运行记录。原始模型和冻结 internal-v2 数据的版本、内容、SHA 均保持不变，直接引用执行者现有的本机路径，无须重新传输模型或数据。

原始预训练来源仍为 `microsoft/mdeberta-v3-base`，revision 为 `a0484667b22365f84929a935b5e50a51f71f159d`。原包的全部模型、数据及文件身份检查继续生效，不通过重命名、替换数据或关闭校验绕过。

## 入口参数与手动执行

| 参数 | 含义 |
| --- | --- |
| `--executor B\|C` | 实际执行者；决定检查哪一组本人实验，以及接管哪些 A/seed43 arm。 |
| `--source-workspace` | 执行者原 B 或 C 工作区；用于核对前置完成状态，并读取 `.local/v3/B.json` 或 `.local/v3/C.json` 中已有的模型、数据路径。 |
| `--workspace` | 全新的隔离工作区；不能覆盖或复用已有实验目录。 |
| `--code-archive` | 上述固定大小和 SHA 的现有 ZIP。 |
| `--execute` | 未来明确手动启动；省略时只显示计划。 |

在本目录运行以下命令前，由执行者将三个路径变量设置为自己的实际路径。默认命令仅查看计划：

```powershell
python -B run_takeover.py --executor C --source-workspace $sourceWorkspace --workspace $newWorkspace --code-archive $codeArchive
```

B 查看自己的接管计划时，将 `--executor C` 改为 `--executor B`。此阶段不自动等待、轮询或排程。

确认前置条件满足后，未来手动执行时才在同一命令末尾加 `--execute`：

```powershell
python -B run_takeover.py --executor C --source-workspace $sourceWorkspace --workspace $newWorkspace --code-archive $codeArchive --execute
```

执行入口应在全新工作区核验并解包固定 ZIP，复用原工作区配置中的模型、数据路径，再以逻辑 `owner A` 生成接管主机配置和本机 preflight。每个指定 arm 依次完成 `pilot → prepare → run → package`；前一阶段失败即停止，不自动放宽资源门槛或跳过失败。B 的 A/T1 完成后才进入 A/T2。

须使用执行者本人原任务验证过的 Python 3.11 解释器；脚本会核对主机与解释器身份。B 在原工作区根目录的预留命令如下，本轮仅保存，不执行 `--execute`：

```powershell
python -B handoffs/B/v3_a_takeover_20261020/run_takeover.py --executor B --source-workspace . --workspace .local/workspaces/v3-a-takeover-B --code-archive runs/b_v3_preparation/packages-runtimefix1/v3-code-20261020.zip
```

C 将 `$sourceWorkspace` 设为其 C/seed44 三项正式实验所在的原工作区，`$codeArchive` 指向下载的 ZIP，`$newWorkspace` 选择与原模型同盘、使模型相对路径仅含 ASCII 的全新目录。若原任务属于“最终检查点完成后恢复，但缺最终训练 result.json”的特殊情况，脚本会停止供人工复核，不据此改写原任务或绕过完成门槛。

隔离工作区不能作为修改冻结矩阵、seed 或训练配方的理由。固定 512 长度、batch1/累积16、7500 有效更新、原始预训练起点及各 arm 的原配方全部不变。Windows 下需使用同盘且适合现有 tokenizer 相对路径规则的工作区，并从该工作区根目录执行后续命令。

## 预算、历史失败与停止边界

接管主机必须为 A/seed43 的每个指定 arm 重新完成本机 pilot；不得复用 B/seed42、C/seed44 或原 A 主机的 PASS。此次明确迁移后的新主机 pilot 各有独立 **1 小时**预算，正式训练每 arm **10 小时**，独立评估每 arm **0.5 小时**。A 不自动继承 B/T0 的 11 小时或 B/T1 的 13 小时例外。原包 RAM、磁盘、CUDA 独占检查及时间外推门槛继续生效，超门槛即停止。

原 A 的全部失败现场、原始日志和账本永久保留，包括已发生的 **25.09 秒**历史支出；不重写为零，不导入为新主机 PASS，也不作为可恢复权重使用。新的主机配置和 pilot 属于本次明确授权的迁移验收；历史失败成本与新主机支出分别列示，不能表述为整体成本归零。

原工作区和新工作区的 GPU 锁路径不同，不能仅凭新工作区能取得锁判断本人旧实验已完成。前置完成检查及启动前 GPU 进程检查均须保留。现有 WDDM 检查对 native 非 Python CUDA 工作负载存在识别限制，执行者还需确认没有其他该类任务；不自动终止用户程序。

若启动后失败，保留该工作区和累计账本，停止后回报；不得反复新建目录重置已消耗预算。不得将本次文件交付或计划检查称为训练可运行、pilot 通过或正式实验完成。

## 产物交接与解释范围

冻结入口默认将接管结果保存在新工作区的 `runs/v3/V3-20261020-A-s43-T*`，轻量交接产物位于 `handoffs/A/v3/`。这些路径表达的是逻辑 A/seed43 的实验归属；执行者应在自己的清单中另记实际 operator、host、代码包 SHA、对应原工作区和新工作区。不要为体现执行者身份而修改冻结实验 ID、协议或产物目录。

本次用户新分工明确覆盖原冻结包及其 README 中“每人／同机完成自己的三项实验”的执行安排；冻结训练数学、数据、模型来源和验收标准不变。报告须说明 A/seed43 的 T0 在 C 主机、T1/T2 在 B 主机执行，跨 arm 对照存在硬件／驱动等主机因素混杂。固定 seed 不能保证不同主机逐位等同，也不能把这些结果当作新增独立 seed 或额外重复实验。保留全部成功与失败结果，不择优替换。

本次没有执行新任务或创建自动排程。是否能在接管主机完成各项实验，仍以该主机后续实际 preflight、pilot、时间预算、独立重载及正式 dev 产物为准。
