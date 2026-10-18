# C：轻量基线、融合与技术报告

维护人：C。最后核对：2026-10-18。仅C更新本文件的清单和记录；未勾选表示本仓库尚未确认/验收，不推断C未开展工作。

## 当前目标和前置条件

目标：建立CPU轻量基线，为主模型提供可信对照，判断融合收益，并统稿技术报告。

- 安装Git LFS、拉取公开仓库；提交改动需协作写权限或Fork后提PR。在自己电脑准备Python环境及 `requirements/base.txt`，不需要等待B的GPU或复制B的环境目录。
- 已读 `TEAM_WORKPLAN.md` 的标签、预测、划分与验证集规则。
- **C01代码框架、实验表和提纲可立即准备；C02需A真实数据与统一评估。** 使用相同train/dev，不自行重新划分。
- **C03需B/C同一调参集的预测齐全**，以及A的数据版本和评估口径；C04需A/B贡献自己的章节；最终PDF需方案与结果冻结。
- 当前已接收B08报告材料、第一版冻结草案和B本地统一融合验收；A/B的逐点数值验收差异尚未闭环，内部test与官方验证集均未使用。

## 任务清单

- [x] **C01 代码框架与报告提纲**：字符TF-IDF＋逻辑回归；统一读写接口；建立共享实验记录和报告框架。代码建议放baseline/，报告放reports/；已有路径优先复用。已完成 `baseline/run.py`、运行说明、实验表模板和匿名报告提纲；internal-v2 smoke train/dev 端到端运行通过，200条预测无缺失/多余编号，保存后重载预测CSV哈希一致。smoke结果仅作流程验证，不作为正式性能结论。
- [x] **C02 基线与错误分析**：从字符n-gram 2—5、类别均衡权重开始；词表/IDF/分类器只在训练集拟合；输出dev预测，用A脚本评估全体/短文本及S；记录参数、数据版本、耗时和内部错误类型。不用内部测试反复选参数。internal-v2 train/dev 哈希通过manifest校验；119842条训练、15086条dev。基线Macro-F1=0.946766、短文本Macro-F1=0.918468、临时S=0.938276；按domain/generator/长度的隐私保护错误统计已生成。短文本暂用空白切词<100，等待A确认官方多语言计词实现。
- [x] **C03 融合与校准（复核完成，第一版候选待冻结）**：按预先固定的C/B=0.7/0.3复核B 512/seed43/2000预测和融合文件，未进行新权重/阈值搜索。B七份512预测/融合CSV均通过manifest SHA、15086行、输入ID顺序、唯一性和概率范围校验。融合文件与公式最大概率误差`1.11e-16`；A脚本复算全体宏F1=0.968463、短文本宏F1=0.953381、临时S=0.963939、总错误440；log-loss=0.148132，Brier=0.034057。C/B单独正确569/567条。B已在Python 3.11.17环境加载C模型，完整dev预测对参考最大差4.9996e-11、classes_=[0,1]；统一离线包完整dev的B/C/融合最大差为0/4.9996e-11/3.4997e-11，0.5决策一致。仍为同dev开发结果，短文本口径临时、融合未冻结。A独立复现已报告0.5决策全一致，但B分量274条、融合4条在原逐点容差下超限，数值验收差异待A/B闭环。
- [x] **C04 实验归档和报告统稿（第一版可审阅稿已交）**：已整合A数据/评测、B08主模型与训练资源、C基线/融合/错误分析和A/B跨环境复现状态，更新 `reports/technical_report_draft.md`、`reports/experiment_log.csv`，新增 `reports/release_checklist.md` 与可审阅PDF/DOCX。报告保留训练源码追溯限制、seed波动、训练覆盖不足、同dev选择偏差和OOD证据不足；生成器切片仅作描述性分析；未追加权重/阈值/校准搜索。最终定稿仍需第一版冻结和一次内部test结果，故C04按“第一版完成、最终待补”记录。
- [ ] **C05 最终交付**：若融合保留，向量器/基线模型/校准器和准确顺序已通过H04交B，无需重交；剩余为第一版冻结后纳入一次内部test验收结果、核对报告参数/指标与最终模型一致、完成最终匿名technical_report.pdf及提交材料检查。当前不运行test或官方验证集。

## 输入与交付标准

输入：A六字段train/dev和统一评估命令；B同一dev的 `sample_id,p_machine`。

交付：

- 基线代码、模型与向量器的保存/加载逻辑；必要模型用LFS。
- `handoffs/C/<实验编号>/`：dev预测、参数、数据版本/哈希、代码commit、A脚本指标、简短错误分析。
- 若融合：完整权重/校准器和先后处理规则、单模型/融合对比、可复制推理命令；B无需自行猜接口。
- `reports/`：实验记录表、报告源稿、基线/融合章节、来源/许可证与匿名检查清单、最终PDF。正文不出现团队/学校/个人身份。

实验表至少含编号、负责人、数据版本、模型、主要参数、seed、全体宏F1、短文本宏F1、综合S、耗时和结论。官方计词未明确时注明临时口径；合成调试指标不写成竞赛性能。

## 上传边界

上传：基线/融合代码、确有交接需要的模型与预测、配置和指标摘要、共享实验表、报告源稿/PDF/构建脚本、来源与匿名检查清单、本任务清单。

不上传：个人Python/Conda环境、包缓存、IDE设置、机器绝对路径配置、令牌、全部本机实验目录、临时TF-IDF矩阵和调参缓存。临时结果放 `.local/` 或 `runs/`；只把接收方需要的结果整理进handoffs后提交。

## 下一步与交接记录

下一步：将本版报告源稿、可审阅PDF/DOCX、实验表、来源/许可证与匿名检查清单经 `work/c-baseline` 提交并向main提PR，在本issue #6回执接收的B commit、报告位置、完成项和剩余阻塞。随后等待A/B数值验收闭环与第一版冻结；冻结后由A执行一次内部test，C把结果纳入同一报告并导出最终匿名PDF。不追加融合权重/阈值/校准搜索，不重复H04，不运行test或官方验证集。

| 日期 | 任务编号 | 状态/产物与commit | 验收结果 | 已交给谁/下一步 |
| --- | --- | --- | --- | --- |
| 2026-10-15 | C01 | 待C补充，协调窗口尚未核实交付 | 待确认 | 等A数据运行基线；融合等B预测 |
| 2026-10-16 | C01 | `work/c-baseline` commit `2e9c9aa`：新增 `baseline/run.py`、运行说明、实验表和报告提纲 | Python编译与CLI检查通过；internal-v2 smoke 1000/200训练、评估通过；保存/重载预测SHA-256一致。临时词数口径，非正式分数 | C01 已提交推送 |
| 2026-10-16 | C02 | `baseline/run.py`、`baseline/analyze.py`；`handoffs/C/c_v2_round1/`；internal-v2 train/dev哈希分别为`5a5dca9d…b7ff6` / `1d80e7b6…76600` | 完整dev 15086行对齐；全体Macro-F1=0.946766、短文本Macro-F1=0.918468、临时S=0.938276；拟合530.635秒。误报319、漏报428；切片不导出正文或sample_id | 交A统一复核；不触碰官方验证集、不用内部test挑参 |
| 2026-10-16 | C03初轮 | `baseline/compare.py`；B交接提交`4ca7c858…736795b`；完整比较和融合CSV在`handoffs/C/c_v2_round1/` | B三份预测及C预测均通过15086行/顺序/概率验收；dev初选C/B=0.7/0.3，CE2000_s43临时S=0.957972。条件group bootstrap区间不消除同dev选权重的偏差，未冻结方案 | 将结果交B讨论；等待A确认官方评测口径，按统一口径重算后再决策 |
| 2026-10-16 | C03/H04/报告更新 | 同步B PR #4 merge `89870c7c9ac716b9747e20c76800bbed8692114c`；新增`baseline/verify_fixed_fusion.py`及`handoffs/C/c_v2_round2_h04/` | B 6个512节点及固定融合CSV的manifest SHA、格式、编号顺序均通过；固定权重公式通过；Python3.14重载C模型，完整dev参考预测SHA完全一致。Python3.11兼容未验；训练时源码指纹未记录，已明确披露 | 请B完成跨环境加载，A确认官方口径；随后提交C PR |
| 2026-10-17 | C03跨环境复核 | B本地统一融合验收、`handoffs/B/b_fusion_local_acceptance/`；A独立复现回执 | B Python3.11加载C模型成功，完整dev最大差4.9996e-11、classes_=[0,1]；统一包0.5决策一致。A报告B分量274条、融合4条超原逐点容差，分类决策全一致；原核验程序未数值PASS，原因待闭环 | 不重复H04；在报告中写“分类决策一致、数值验收差异待A/B闭环” |
| 2026-10-18 | C04第一版收尾 | 从main接收B08材料、冻结草案（PR #11合并head `1bb48c8`）；更新`reports/technical_report_draft.md`、`reports/experiment_log.csv`；新增`reports/release_checklist.md`、`reports/build_technical_report.py`、`reports/check_report_anonymity.py`、`reports/release_anonymity_check.json`、10页可审阅`technical_report_v1_review.pdf/.docx`及`requirements/report.txt` | 报告数字与B `metrics.csv`/`resources.csv`、C H04和A回执逐项对照；来源/许可证与匿名机械检查通过（`PASS_REVIEW_PDF_PENDING_FINAL_FREEZE`）；未新增训练/融合搜索，未运行test/官方val | 经`work/c-baseline`提交并提PR；在issue #6回执；等待A/B数值闭环与冻结 |
