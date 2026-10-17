# B统一融合本地验收与A/C下一步

2026-10-17更新：用户已发送模型ZIP；A已通过PR #10报告同包CPU独立复现，但机器检查文件与B分量2.56e-5最大差仍待复核，详见[冻结草案](RELEASE_FREEZE_DRAFT_20261017.md)。本版优先工程交付，Dropout继续暂停。下文“稍后交A/待发送”为此前记录，原包及历史验收清单不改写。

用户已前台完成现有候选的独立包推理。本轮B仅CPU复核运行产物、更新代码/交接，不启动训练、GPU推理或后台监测。原runs及已验收推理包原样保留。

## 已通过什么

- 200条固定smoke_dev和15086条固定dev的统一入口均PASS，FP32/batch8、512token、B seed43/2000、C/B=.7/.3。编号/原顺序/唯一性/行数、有限[0,1]、machine方向和.5分类决策核验通过。
- 完整dev：B相对原独立重载参考最大差0；C最大差4.9996e-11；融合最大差3.4997e-11。smoke的B/融合最大差4.1723e-7/1.2515e-7，均在原定容差内且分类决策一致。不同batch组合的smoke差异不冒充逐位相同。
- 融合全体宏F1=.968463、临时短文本宏F1=.953381、临时S=.963939；human召回=.956324、machine召回=.979147，误报240/漏报200、共440错。概率质量与固定融合开发记录相容，精确数值见交接metrics.json；dev没有进入梯度更新。
- dev的C CPU推理34.467秒、B GPU推理244.087秒，B峰值allocated1232.797MiB（仅此次推理，不与训练峰值混用）。本地验证证明当前包复现已有dev候选，不代表新性能、A独立环境H05或官方成绩。

## 模型清单已交，权重稍后单独交A

用户明确选择本次先合代码、验收证据和模型清单，约1.15GB推理包保留本机`runs/b_fusion_acceptance_final/inference_bundle/`，稍后单独交A。该包只有一个B权重/分词器、C模型、离线代码、依赖和三份参考概率，不含优化器、其他seed、环境或中间checkpoint。本轮LFS上传已停止；仓库不含这个推理包，不能仅凭git pull开始A模型复现。

本机发送用ZIP现已准备在`.local/b_fusion_delivery/b_v2_fusion_s43_2000_for_A.zip`（915002576字节，约915MB），旁边有同名.sha256和给A的README，ZIP内也包含说明与纯标准库文件校验脚本。ZIP SHA-256为`f5b34bb2d87e884ce876152875f0692c6e8f8eb03de26de9c50af696886dd687`；20个归档条目的SHA均已核验，原包字节不变。用户自行把ZIP及.sha256发给A；A解压至项目`handoffs/shared_models/`后，模型目录为`handoffs/shared_models/b_v2_fusion_s43_2000/`。当前仅“打包待发送”，不代表已传输、A已收到或H05完成；发送包保留本地，不上传Git。

原始bundle.json SHA-256：`5775334edda46069d45f7eaade2222bea2c7393d19c6faae44a3ed4a256ab50a`。`b_fusion_local_acceptance/model_inventory.json`是清单的LF副本，其自身SHA在交接manifest中；不要将副本SHA误认成原bundle SHA。交付状态见model_delivery.json。包内DEVELOPMENT_CANDIDATE/PENDING_A是历史状态，B本地PASS写于包外证据。

A取得完整目录后，先核对原bundle SHA和17个文件SHA，再在本人独立环境运行。PATH_TO_BUNDLE替换为实际接收目录；下列命令只使用A冻结的内部smoke_dev/dev，不访问test或官方val：

```powershell
python PATH_TO_BUNDLE/fusion_inference.py --bundle PATH_TO_BUNDLE --input data/processed/v2/smoke_dev.jsonl --output runs/a_fusion_smoke
python PATH_TO_BUNDLE/verify_fusion_output.py --bundle PATH_TO_BUNDLE --input data/processed/v2/smoke_dev.jsonl --predictions runs/a_fusion_smoke --output runs/a_fusion_smoke_check.json

python PATH_TO_BUNDLE/fusion_inference.py --bundle PATH_TO_BUNDLE --input data/processed/v2/dev.jsonl --output runs/a_fusion_dev
python PATH_TO_BUNDLE/verify_fusion_output.py --bundle PATH_TO_BUNDLE --input data/processed/v2/dev.jsonl --predictions runs/a_fusion_dev --output runs/a_fusion_dev_check.json
```

先按机器安装合适torch构建和包内requirements.txt；默认CUDA，CPU可加`--device cpu`且不依赖nvidia-smi，但完整XLM-R CPU推理较慢。A须回传bundle SHA、环境、两个核验报告、警告和实际GitHub证据；仅复评已给CSV不等于模型独立复现。具体准备/恢复说明见FUSION_DROPOUT_RUNBOOK_20261016.md。

## 按用户建议收窄评测协议

依据是用户本轮提供的题面澄清，不额外声称已取得官方答疑。综合分公式与<100词边界已明确；继续只把多语言计词、概率转类别、CSV准确模板/编号对应/类别方向和真实截止时间列为未决。内部固定word/.5/100及p_machine；现有结果按同一版本/口径比较，最终按正式模板适配概率。

## 下一步分工，已完成内容不重复安排

**A：**旧512两seed及固定融合的CSV复评已由PR #7记录，无须再以同一任务重复验收。现在优先与B接收同SHA完整推理包，取得后做真正独立环境smoke/full-dev复现；同时本人整理一页已明确/内部约定/未决协议，完成数据与评测章节。对默认`--min-len 20`额外清洗先说明依据和影响：先核对源码、既有清洗审计及计数；若需新增统计，应另拟仅train/dev或train-only方案，不读test、不重新生成/替换本轮划分。不得把“题面只要求过滤空文本”偷换成“现流程已官方认可”或直接切换新数据。发现确定泄漏/标签问题则单独报告，不能用当前高分掩盖。

**C：**H04及固定融合复核已交付，B跨Python版本加载及完整dev也通过，无需再训基线或重交同一模型。现在使用B实际本地验收数字/运行入口更新本人报告和C04/C05进展；提供正式匿名材料检查清单，核对代码/文件名/报告中的个人、学校、队伍信息与内部本机路径。报告写清训练覆盖不足、seed波动、同dev选择偏差和OOD证据缺失；将“test评估后决定最终方案”改为“冻结后一次验收，不回流反复选参”。不追加融合权重搜索。

**B：**本地统一入口已通过，但B07保持未完成，等待A独立复现；B06完整训练覆盖/未见生成器实验仍未完成。已有Dropout0.2双seed固定预算配置仅属准备，不执行、不宣布优于0.1，不改现有2000步候选。A真实回执到位后，按既有门槛记录准确bundle SHA与证据，再由用户自行前台运行0.2；仍从原始预训练初始化、不复用0.1状态。

内部test继续封存。先把方案、数据、内部评测口径和选择记录冻结，再由A一次正式test验收；不能凭v2 dev高分宣布跨时间或未见生成器泛化。官方val只在冻结后最终推理。

## 文件与回执

`b_fusion_local_acceptance/`只整理必要本地验收摘要、smoke/dev融合预测、格式/参考比对、A评估、阶段和SHA，不上传完整runs、模型权重或既有三套参考CSV；只有必要的模型文件清单。本机原始analysis/attempt-002.zip仍保留。通知在原issue #5/#6延续，发布/合并不等于A已读或H05完成。
