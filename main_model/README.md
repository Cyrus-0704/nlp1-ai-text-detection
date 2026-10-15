# B：主模型训练和预测

任务状态见[个人清单](../docs/tasks/B_MAIN_MODEL.md)，共享接口见[团队约定](../TEAM_WORKPLAN.md)。各人自己准备环境；本文使用当前环境的python和仓库相对路径，本机绝对路径/安装日志不上传。

当前v2先导结果见 [完整dev对照与图](../handoffs/B/b_v2_full_pilot/README.md)：CE500宏F1=0.882048、临时S=0.880403，本轮优于加权500步。仅单seed、约6.7%训练覆盖，尚非最终模型。小样50步链路及两项完整dev保存重载均已通过，不重复现有验证代替下一阶段实验。

## 依赖与预训练模型

B在Windows/NVIDIA上的已验证版本是Python3.11、PyTorch2.13.0+cu126与Transformers4.57.6；其他电脑先确认驱动和PyTorch构建适配。

```powershell
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/train.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python main_model/download_model.py --output-dir models/xlm-roberta-base --revision e73636d4f797dec63c3081bb6ed5c7b0bb3f2089
```

模型来自FacebookAI/xlm-roberta-base，固定revision与文件哈希写入本机source_manifest.json。models/不上传，队员需要时自行下载；原始预训练权重不能与合成调试checkpoint混淆。

## A交付后的真实小样验证

收到A的manifest、六字段真实内部train/dev及统一评估说明后，从仓库根目录运行。以下路径按A实际版本调整，输出目录必须是新的。

```powershell
python main_model/train.py --train-file data/processed/v2/smoke_train.jsonl --dev-file data/processed/v2/smoke_dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_v2_smoke_50_repeat --data-kind a-internal --data-manifest data/processed/v2/manifest.json --evaluation-script data_tools/evaluate.py --seed 42 --max-steps 50 --max-length 256 --batch-size 1 --gradient-accumulation 16 --learning-rate 2e-5 --deterministic
python main_model/predict.py --model-dir runs/b_v2_smoke_50_repeat/checkpoint --input data/processed/v2/smoke_dev.jsonl --output runs/b_v2_smoke_50_repeat/dev_predictions_independent.csv --precision fp32 --batch-size 1
python data_tools/evaluate.py --gold data/processed/v2/smoke_dev.jsonl --pred runs/b_v2_smoke_50_repeat/dev_predictions.csv --threshold 0.5 --short-by word --short-threshold 100 --save-report runs/b_v2_smoke_50_repeat/a_evaluation_repeat.txt
python main_model/package_checkpoint.py --checkpoint runs/b_v2_smoke_50_repeat/checkpoint
python main_model/check_contracts.py
```

train/dev支持UTF-8 JSONL/CSV，需sample_id,group_id,text,label,generator,domain，两类均有，human=0/machine=1。预测仅需sample_id,text；CSV编号按字符串读取并保留前导零，输出sample_id,p_machine保持顺序。

入口拒绝train/dev中的编号、分组、空白规范化原文重叠。只用A内部划分训练调参，不传官方验证集。模型分类头随机初始化是正常情况。

`--data-manifest` 校验输入文件大小与 SHA-256 并保存版本来源；`--evaluation-script` 在严格核验独立进程预测后调用 A 的脚本，显式按空白词数少于100的临时口径和0.5阈值评估。保存前后对比、训练速度、峰值显存、代码commit和实际序列长度记录于 `run_summary.json`；A报告为 `a_evaluation_report.txt`。推理入口只输出概率，不计算标签分布或指标。

当前小样配置：256 token、batch1、梯度累积16、lr2e-5、weight_decay0.01、seed42、FP16、梯度检查点、num_workers0、恒定学习率。50步指实际优化器更新，GradScaler跳步单独记录。推理默认FP32；保存前后对比使用同一精度和batch。

## 已核实的本机验证与限制

B本机已经完成正式预训练模型下载及1000/200合成样本50次更新验证、独立进程重载预测；这些数据和运行目录均留本地。合成样本不代表真实来源标签，不得报告竞赛性能。不要重复该轮验证来代替A真实小样验收。

训练支持周期检查点、完整断点续训和类别加权。临时指标阈值0.5，短文本按空白分词少于100，不等于官方多语言计词。

## 保存、恢复与类别加权

`--save-steps 100 --keep-checkpoints 2` 控制优化器更新边界的周期保存与保留数量。`checkpoints/step-XXXXXX/` 包含模型、分词器、AdamW、FP16 scaler、Python/NumPy/Torch/CUDA随机状态、累计日志和样本迭代位置；先写临时目录，完成哈希清单后才改名为有效检查点。`checkpoint/` 则是最终推理包，不含恢复用优化器状态。

恢复必须指定一个新的输出目录，数据、预训练来源、随机种子、长度、batch、累积、学习率及加权设置应保持一致。输入哈希和恢复状态哈希不一致会被拒绝。当前使用恒定学习率，无可变调度器状态；`--max-steps` 是恢复后的总目标步数，可以增大。当前仅支持单GPU、num_workers=0；同机确定性核验不代表跨软件版本/硬件逐位一致。

```powershell
python main_model/train.py --train-file data/processed/v2/train.jsonl --dev-file data/processed/v2/dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_v2_ce2000_resume --data-kind a-internal --data-manifest data/processed/v2/manifest.json --evaluation-script data_tools/evaluate.py --seed 42 --max-steps 2000 --max-length 256 --batch-size 1 --gradient-accumulation 16 --learning-rate 2e-5 --save-steps 250 --keep-checkpoints 2 --class-weighting none --deterministic --inference-batch-size 8 --resume-from runs/b_v2_full_ce500/checkpoints/step-000500

python main_model/check_training_state.py
```

`--class-weighting balanced` 仅按训练标签计算 `w_c=N/(2*N_c)`，损失为 `mean(每条CE * w[label])`；不能在batch=1时直接使用按batch权重归一化的加权均值，否则权重会被抵消。`check_training_state.py` 验证了该梯度比例和跨epoch样本游标恢复。恢复用 `.pt` 只读取本地可信运行生成的文件，不作为推理交付要求。

## 内部dev诊断图

可选依赖 `requirements/analysis.txt`；不会进入模型推理包。每个run目录必须有PASS的运行摘要与完全匹配同一dev编号/哈希的预测；默认0.5阈值，不搜索阈值。图表和分组bootstrap仅用于开发诊断。

```powershell
python main_model/analyze_performance.py --gold-internal-dev data/processed/v2/dev.jsonl --run CE500=runs/b_v2_full_ce500 --run Balanced500=runs/b_v2_full_balanced500 --output-dir runs/b_v2_full_analysis_repeat
```

该入口输出宏F1/AUC/概率分布/训练曲线、混淆矩阵、domain/generator分项和以dev的group_id为单位的bootstrap区间。区间仅反映当前固定模型下dev抽样不确定性，不覆盖不同seed和训练样本变化。单类别或小样本切片只能定位问题，不能据此宣称某生成器的可靠泛化能力。历史v1的CE50轨迹只留档；本轮只比较同一v2完整dev的两项500步结果。

## 按需交接

runs/和models/保持本地。把A/C需要的真实dev预测、参数、数据版本/哈希、commit与指标摘要整理到handoffs/B/<实验编号>/。需要全队加载的最终模型放handoffs/shared_models/<版本>/，使用LFS；不上传环境、缓存、全部日志或每个中间checkpoint。

正式冻结推理与官方CSV导出由B提供入口、A运行验收，融合方案若保留须把C的完整规则集成到入口。详细责任和前置条件见B个人清单。

## 修复数据验收

A的internal-v2已通过B独立复现，报告见 `handoffs/B/full_data_acceptance_v2/`。PR #3已合并main `7231172`并同步B；新实验一律使用v2输入及新输出目录。

```powershell
python main_model/verify_data_delivery.py --manifest data/processed/v2/manifest.json --raw-file data.jsonl --split-dir .local/internal-v2-repeat --output-dir runs/b_acceptance_v2_repeat --reproduce
```

验收只解析train/dev；内部test只核验不透明字节。合并前只读检查可用 `--preprocess-script` 指定A交付的原样脚本快照，并用 `--delivery-commit` 记录其准确提交；manifest和小样必须来自该提交，不能改哈希绕过门槛。

## internal-v2完整数据先导实验

PR #3已合并main `7231172`，B已同步；数据独立复现通过。先导预算固定为每项500次优化器更新，单seed42，分别从原始预训练模型开始。该预算只覆盖约8000条训练抽取，不是完整epoch，不称为最终模型。完整dev仅评估冻结步数的预测，不用内部test选参数。

```powershell
python main_model/train.py --train-file data/processed/v2/train.jsonl --dev-file data/processed/v2/dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_v2_full_ce500 --data-kind a-internal --data-manifest data/processed/v2/manifest.json --evaluation-script data_tools/evaluate.py --seed 42 --max-steps 500 --max-length 256 --batch-size 1 --gradient-accumulation 16 --learning-rate 2e-5 --class-weighting none --save-steps 250 --keep-checkpoints 2 --deterministic --inference-batch-size 8
python main_model/train.py --train-file data/processed/v2/train.jsonl --dev-file data/processed/v2/dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_v2_full_balanced500 --data-kind a-internal --data-manifest data/processed/v2/manifest.json --evaluation-script data_tools/evaluate.py --seed 42 --max-steps 500 --max-length 256 --batch-size 1 --gradient-accumulation 16 --learning-rate 2e-5 --class-weighting balanced --save-steps 250 --keep-checkpoints 2 --deterministic --inference-batch-size 8
```

完整划分按A脚本本地生成并校验，不上传副本。新增 `--inference-batch-size` 默认1；完整dev使用8，保存前与新进程加载后使用同一FP32精度、同一batch和顺序严格比较；不改变训练batch与有效batch。运行摘要记录两段推理耗时。图表标签同时适用于小样dev和完整dev。

完整dev对照已完成；以下用于按需复画评估图及不含正文的诊断数据，输出目录须新的：

```powershell
python main_model/analyze_performance.py --gold-internal-dev data/processed/v2/dev.jsonl --run CE500=runs/b_v2_full_ce500 --run Balanced500=runs/b_v2_full_balanced500 --output-dir runs/b_v2_full_pilot_analysis
python main_model/profile_lengths.py --data-manifest data/processed/v2/manifest.json --model-dir models/xlm-roberta-base --output-dir runs/b_v2_length_profile_repeat
```

性能/ROC/概率/训练曲线为performance，混淆矩阵为confusion，召回/PR/固定10箱可靠性/宏F1组bootstrap区间为error_analysis，均提供PNG/PDF。可靠性曲线只作诊断，不拟合校准器或搜索阈值。长度入口只读manifest声明的train/dev并核验哈希，统计完整token长度（含特殊token），不读test、不导出文本或改写划分。
