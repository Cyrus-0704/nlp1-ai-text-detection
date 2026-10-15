# B：主模型训练和预测

任务状态见[个人清单](../docs/tasks/B_MAIN_MODEL.md)，共享接口见[团队约定](../TEAM_WORKPLAN.md)。各人自己准备环境；本文使用当前环境的python和仓库相对路径，本机绝对路径/安装日志不上传。

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
python main_model/train.py --train-file data/processed/v1/smoke_train.jsonl --dev-file data/processed/v1/smoke_dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_real_smoke_50 --data-kind a-internal --data-manifest data/processed/v1/manifest.json --evaluation-script data_tools/evaluate.py --seed 42 --max-steps 50 --max-length 256 --batch-size 1 --gradient-accumulation 16 --learning-rate 2e-5
python main_model/predict.py --model-dir runs/b_real_smoke_50/checkpoint --input data/processed/v1/smoke_dev.jsonl --output runs/b_real_smoke_50/dev_predictions_independent.csv --precision fp32 --batch-size 1
python data_tools/evaluate.py --gold data/processed/v1/smoke_dev.jsonl --pred runs/b_real_smoke_50/dev_predictions.csv --threshold 0.5 --short-by word --short-threshold 100 --save-report runs/b_real_smoke_50/a_evaluation_repeat.txt
python main_model/package_checkpoint.py --checkpoint runs/b_real_smoke_50/checkpoint
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
python main_model/train.py --train-file data/processed/v1/smoke_train.jsonl --dev-file data/processed/v1/smoke_dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_resumed_new --data-kind a-internal --data-manifest data/processed/v1/manifest.json --evaluation-script data_tools/evaluate.py --max-steps 400 --save-steps 100 --keep-checkpoints 2 --class-weighting none --deterministic --resume-from runs/b_ce200/checkpoints/step-000200

python main_model/verify_resume.py --train-file data/processed/v1/smoke_train.jsonl --dev-file data/processed/v1/smoke_dev.jsonl --data-manifest data/processed/v1/manifest.json --model-dir models/xlm-roberta-base --output-dir runs/b_resume_verification_repeat
python main_model/check_training_state.py
```

`--class-weighting balanced` 仅按训练标签计算 `w_c=N/(2*N_c)`，损失为 `mean(每条CE * w[label])`；不能在batch=1时直接使用按batch权重归一化的加权均值，否则权重会被抵消。`check_training_state.py` 验证了该梯度比例和跨epoch样本游标恢复。恢复用 `.pt` 只读取本地可信运行生成的文件，不作为推理交付要求。

## 内部dev诊断图

可选依赖 `requirements/analysis.txt`；不会进入模型推理包。每个run目录必须有PASS的运行摘要与完全匹配同一dev编号/哈希的预测；默认0.5阈值，不搜索阈值。图表和分组bootstrap仅用于开发诊断。

```powershell
python main_model/analyze_performance.py --gold-internal-dev data/processed/v1/smoke_dev.jsonl --run CE50=runs/b_ce50_control --run CE200=runs/b_ce200 --run Balanced200=runs/b_balanced200 --output-dir runs/b_performance_analysis
```

该入口输出宏F1/AUC/概率分布/训练曲线、混淆矩阵、domain/generator分项和以dev的group_id为单位的bootstrap区间。区间仅反映当前固定模型下dev抽样不确定性，不覆盖不同seed和训练样本变化。单类别或小样本切片只能定位问题，不能据此宣称某生成器的可靠泛化能力。CE50对照取自CE200同一训练轨迹的50步检查点，与此前旧50步流程结果区分。

## 按需交接

runs/和models/保持本地。把A/C需要的真实dev预测、参数、数据版本/哈希、commit与指标摘要整理到handoffs/B/<实验编号>/。需要全队加载的最终模型放handoffs/shared_models/<版本>/，使用LFS；不上传环境、缓存、全部日志或每个中间checkpoint。

正式冻结推理与官方CSV导出由B提供入口、A运行验收，融合方案若保留须把C的完整规则集成到入口。详细责任和前置条件见B个人清单。

## 修复数据的合并前验收

A的internal-v2已通过B独立复现，报告见 `handoffs/B/full_data_acceptance_v2/`。PR #3合入main并同步后才能作为主干交付使用；v1命令仅保留历史示例，新实验改用v2输入及新输出目录。

```powershell
python main_model/verify_data_delivery.py --manifest data/processed/v2/manifest.json --raw-file data.jsonl --split-dir .local/internal-v2-repeat --output-dir runs/b_acceptance_v2_repeat --reproduce
```

验收只解析train/dev；内部test只核验不透明字节。合并前只读检查可用 `--preprocess-script` 指定A交付的原样脚本快照，并用 `--delivery-commit` 记录其准确提交；manifest和小样必须来自该提交，不能改哈希绕过门槛。
