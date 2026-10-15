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
python main_model/train.py --train-file data/processed/v1/smoke_train.jsonl --dev-file data/processed/v1/smoke_dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_real_smoke_50 --data-kind a-internal --seed 42 --max-steps 50 --max-length 256 --batch-size 1 --gradient-accumulation 16 --learning-rate 2e-5
python main_model/predict.py --model-dir runs/b_real_smoke_50/checkpoint --input data/processed/v1/smoke_dev.jsonl --output runs/b_real_smoke_50/dev_predictions_independent.csv --precision fp32 --batch-size 1
python main_model/check_contracts.py
```

train/dev支持UTF-8 JSONL/CSV，需sample_id,group_id,text,label,generator,domain，两类均有，human=0/machine=1。预测仅需sample_id,text；CSV编号按字符串读取并保留前导零，输出sample_id,p_machine保持顺序。

入口拒绝train/dev中的编号、分组、空白规范化原文重叠。只用A内部划分训练调参，不传官方验证集。模型分类头随机初始化是正常情况。

当前小样配置：256 token、batch1、梯度累积16、lr2e-5、weight_decay0.01、seed42、FP16、梯度检查点、num_workers0、恒定学习率。50步指实际优化器更新，GradScaler跳步单独记录。推理默认FP32；保存前后对比使用同一精度和batch。

## 已核实的本机验证与限制

B本机已经完成正式预训练模型下载及1000/200合成样本50次更新验证、独立进程重载预测；这些数据和运行目录均留本地。合成样本不代表真实来源标签，不得报告竞赛性能。不要重复该轮验证来代替A真实小样验收。

当前训练代码用于固定步数新实验；正式全量训练前仍需完善周期检查点、完整断点续训和类别加权配置。临时指标阈值0.5，短文本按空白分词少于100，仅检查链路，不等于官方多语言计词。

## 按需交接

runs/和models/保持本地。把A/C需要的真实dev预测、参数、数据版本/哈希、commit与指标摘要整理到handoffs/B/<实验编号>/。需要全队加载的最终模型放handoffs/shared_models/<版本>/，使用LFS；不上传环境、缓存、全部日志或每个中间checkpoint。

正式冻结推理与官方CSV导出由B提供入口、A运行验收，融合方案若保留须把C的完整规则集成到入口。详细责任和前置条件见B个人清单。
