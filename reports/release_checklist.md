# 第一版报告来源、许可证与匿名检查清单

维护人：C。日期：2026-10-18。适用于 `reports/technical_report_draft.md` 及其导出的 `technical_report_v1_review.pdf`。本清单是报告流程的一部分，不是新的跨人交接说明；A/B 的个人清单不由 C 代改。第一版提交：`work/c-baseline` commit `ce64d4f`，PR #13。

## 1. 本版报告使用的接收版本

| 项目 | 接收值 | 证据位置 |
| --- | --- | --- |
| B08 报告材料 | `handoffs/B/B08_REPORT_MATERIALS_FOR_C_20261017.md` | 从 `origin/main` 同步 |
| 第一版冻结草案 | `handoffs/B/RELEASE_FREEZE_DRAFT_20261017.md`，状态 `DRAFT_PENDING_ACCEPTANCE_REVIEW_AND_TEAM_FREEZE` | 同上 |
| B 本地统一融合验收 | `handoffs/B/FUSION_LOCAL_ACCEPTANCE_20261016.md`、`handoffs/B/b_fusion_local_acceptance/acceptance.json` | 同上 |
| B 512 两 seed 指标与资源 | `handoffs/B/b_v2_length512_two_seed/metrics.csv`、`resources.csv` | 同上 |
| C H04 接口 | `handoffs/C/c_v2_round2_h04/README.md` | 本分支既有交付 |
| C 基线与固定融合复核 | `handoffs/C/c_v2_round1/`、`baseline/verify_fixed_fusion.py` | 本分支既有交付 |

本版报告不新增训练、不重跑融合、不搜索权重/阈值/校准，不访问内部 test 或官方验证集。

## 2. 数字来源对照

| 报告中的数字/主张 | 一手来源 | 核对结果 |
| --- | --- | --- |
| Train 119,842 / Dev 15,086 / Test 14,967；SHA-256 | `data/processed/v2/manifest.json` | 与报告 2.1 一致 |
| 基线 0.946766 / 0.918468 / 0.938276；319/428 错误；530.635 秒 | `handoffs/C/c_v2_round1/baseline_run.json`、`baseline_dev_predictions.csv`、A evaluator 输出 | 与报告 3.1、4.1、4.3 一致 |
| 256/512 两 seed 的单模型指标、错误、log loss、Brier | `handoffs/B/b_v2_length512_two_seed/metrics.csv` | 与报告 4.1、4.4 一致 |
| 512/seed42、seed43 训练时间、峰值显存、终点推理时间、处理行次 | `handoffs/B/b_v2_length512_two_seed/resources.csv`、`L512_s42_u2000_summary.json`、`L512_s43_u2000_summary.json` | 与报告 3.2 一致 |
| 融合 0.961422 / 0.949922 / 0.957972；0.963726 / 0.940363 / 0.956717；0.968463 / 0.953381 / 0.963939 | `handoffs/B/b_v2_length512_two_seed/metrics.csv` 第 12–14 行 | 与报告 4.2 一致 |
| 融合相对旧 256/seed43 的 S 增量 0.005967、错误 534→440、条件区间 `[0.230, 0.969]` 个百分点 | `handoffs/B/b_v2_length512_two_seed/paired_comparisons.json`、B08 材料 | 与报告 4.2 一致；保留同 dev 选择偏差说明 |
| 融合短文本 0.953381 低于主模型单模型 0.956943 | `metrics.csv` 第 10、14 行 | 与报告 4.2 一致 |
| 基线单独正确 569 / 主模型单独正确 567；都正确 13,770 / 都错误 180 | `handoffs/C/c_v2_round2_h04/README.md`、固定融合复核 | 与报告 4.3 一致 |
| 主模型跨 Python 3.11 加载基线最大差 4.9996e-11；统一包 B 0 / C 4.9996e-11 / 融合 3.4997e-11 | `handoffs/B/FUSION_LOCAL_ACCEPTANCE_20261016.md`、`b_fusion_local_acceptance/acceptance.json` | 与报告 4.5 一致 |
| 完整 dev 推理：C 34.467 秒、B 244.087 秒、B 峰值 1,232.797 MiB | `handoffs/B/FUSION_LOCAL_ACCEPTANCE_20261016.md` | 与报告 3.2、4.5 一致 |
| A 独立复现：B 超限 274/15,086、融合 4/15,086；C 0；0.5 决策全一致；原脚本未数值 PASS | issue #5 的 A 回执（2026-10-17 comment 6233764167），由 B 冻结草案引用 | 报告写为“分类决策一致，数值验收差异待闭环”；未写 CPU/CUDA 原因已证实 |
| 模型来源 revision、C joblib SHA、B 权重 SHA、bundle SHA、参考预测 SHA | `handoffs/B/RELEASE_FREEZE_DRAFT_20261017.md`、`handoffs/C/c_v2_round2_h04/README.md` | 与报告 5.1 一致 |
| 数据为赛题方提供、原始哈希不变 | `data/manifest.raw.json`、`data/processed/v2/manifest.json` | 与报告 2.1、附录 A 一致；具体条款仍待确认 |

## 3. 许可证与引用核对

| 组件 | 许可证/条款 | 核对方式 | 状态 |
| --- | --- | --- | --- |
| XLM-R base | MIT | Hugging Face 模型卡 `license: mit`（2026-10-18 读取） | 已核对 |
| scikit-learn 1.9.1 | BSD-3-Clause | 官方项目许可证 | 已核对 |
| NumPy 2.4.6 | BSD-3-Clause | 官方项目许可证 | 已核对 |
| pandas 3.0.6 | BSD-3-Clause | 官方项目许可证 | 已核对 |
| PyTorch 2.13.0+cu126 | BSD-3-Clause（BSD 风格） | 官方项目许可证 | 已核对 |
| Transformers 4.57.6 | Apache-2.0 | 官方项目许可证 | 已核对 |
| Accelerate 1.15.0 | Apache-2.0 | 官方项目许可证 | 已核对 |
| joblib 1.6.0 | BSD-3-Clause | 官方项目许可证 | 已核对 |
| SciPy 1.17.1 | BSD-3-Clause | 官方项目许可证 | 已核对 |
| 原始 `data.jsonl` | 赛题方提供，受赛题规则约束 | `data/manifest.raw.json` 只记录来源为 competition-provided training data | 来源/哈希已核对；具体使用与再分发条款待数据负责人确认 |
| 本项目代码 | 未单独声明开源许可证 | 仓库未提供 LICENSE | 最终提交前复核；不声称可对外再许可 |

参考文献已在报告末尾列出；XLM-R 使用模型卡与 ACL 2020 论文，scikit-learn、NumPy、PyTorch、Transformers 使用各自原始论文或官方文档。未核验的文献或许可不写入。

## 4. 匿名检查

检查对象：`reports/technical_report_draft.md`、由它生成的 `technical_report_v1_review.docx` 与 `.pdf`、报告文件名、提交文件清单。

- [x] 正文不出现成员姓名、GitHub 用户名、学校、队伍名称、仓库 URL、commit 或本机绝对路径。
- [x] 正文用“数据与评测流程 / 主模型 / 轻量基线”描述分工，不使用成员代号。
- [x] DOCX 核心属性清空作者、最后修改者、备注、分类、关键词；标题只保留报告标题。
- [x] PDF 为匿名审阅稿，文件名不含个人、学校或队伍标识。
- [x] 报告内只引用公开论文/模型卡 URL；不引用团队仓库地址。
- [x] 提交清单只含报告源稿、构建脚本、PDF/DOCX、实验表和本清单；不含 `.git`、`.local`、环境、缓存、个人 runs、模型权重或中间 checkpoint。
- [ ] 最终匿名 PDF：待第一版冻结与一次内部 test 验收后，用同一构建脚本重新导出并复查。
- [ ] 数据条款：待数据负责人按赛题规则确认后，补入报告附录 A 和最终声明。

### 已执行的机械检查

1. 对报告 Markdown、DOCX 解包后的 XML、PDF 原始字节搜索以下敏感串：`Cyrus`、`Cyrus-0704`、`cat-dog-good`、`qll`、`zyj`、`Lenovo`、`F:\`、`C:\`、`github.com/Cyrus-0704`。
2. 检查 DOCX `docProps/core.xml`：`dc:creator`、`cp:lastModifiedBy` 为空；`dc:title` 为报告标题。
3. 检查 PDF 头 `%PDF-`、页数与元数据；对首页、数据表、结果表、跨环境表和附录等页面做了渲染检查，中文、表格和页眉页脚显示正常。
4. 结果已写入 `reports/release_anonymity_check.json`，状态 `PASS_REVIEW_PDF_PENDING_FINAL_FREEZE`；检查脚本为 `reports/check_report_anonymity.py`。若检查发现残留，回到源文件修正后重新导出，不手工改 PDF。

### 当前审阅文件

| 文件 | 说明 |
| --- | --- |
| `reports/technical_report_draft.md` | 报告可编辑源稿（本次主改文件） |
| `reports/technical_report_v1_review.docx` | 由源稿构建的可编辑审阅稿 |
| `reports/technical_report_v1_review.pdf` | 10 页可审阅 PDF；内部审阅稿，不是最终提交版 |
| `reports/release_anonymity_check.json` | 匿名机械检查结果与文件 SHA-256 |
| `reports/build_technical_report.py` | Markdown → DOCX 构建脚本 |
| `reports/check_report_anonymity.py` | 匿名检查脚本 |

## 5. 当前结论与下一步

第一版报告已达到“可审阅稿”状态：数据、单模型、融合、资源、跨环境加载、独立复现状态、来源/许可证和匿名检查均已整合。仍不能称为最终冻结或 H05 完成，原因是：

1. A/B 在原逐点容差下的数值差异尚未闭环（B 274 条、融合 4 条超限，分类决策一致）；
2. 全队尚未记录第一版冻结日期/版本；
3. 内部 test 尚未执行，官方验证集未访问；
4. 数据使用/再分发条款仍待确认；
5. 最终匿名 PDF 尚未生成。

下一步由 C 在冻结和 test 结果到位后更新同一报告并导出最终 PDF；不新增训练、不重选融合、不运行 test/官方 val。
