# B04：真实小样 50 步交接

用途：供 A 复核预测和评估、C 对照同一 smoke_dev 的基线；不是全量训练或冻结比赛模型。产物已推送 B 分支，交接提交 `25ad6e9`，在[草稿PR #2](https://github.com/Cyrus-0704/nlp1-ai-text-detection/pull/2)待复核，尚未合并 main。原始预训练权重、训练模型、缓存及完整运行目录留本地。

## 版本与输入

- A 数据：`internal-v1`，commit `e760e63a1f797eb4562d56be059c099c97597cd8`，已随 PR #1 合并 main `28e6f52` 并同步 B 分支。
- 训练代码：`05ca4900b5b9612e51016e9554fd59cda7621e65`；运行启动时 main_model 代码无未提交修改。
- `data/processed/v1/smoke_train.jsonl`：1000 条，human=366、machine=634；`smoke_dev.jsonl`：200 条，human=73、machine=127。哈希见 `run_summary.json`，与 A manifest 一致。
- 六字段、标签、编号/组别/规范化正文隔离已验证。正式 train/dev/test 未重建；完整集合来源与隔离仍依据 A 审计。未读取内部 test 或官方验证集。
- 从 `FacebookAI/xlm-roberta-base` 原始预训练模型开始，revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`；未从合成调试模型继续训练。来源及权重 SHA-256 见 `pretrained_source_manifest.json`。

## 实际结果

50 次实际更新，无 FP16 跳步；处理 800 条训练样本（不是完整一轮），256 token 上限、batch=1、累积16、lr=2e-5、seed42、FP16和梯度检查点、num_workers=0。实际序列达到256 token。训练51.622秒、0.969更新/秒、15.497条/秒，平均损失0.683802；峰值显存 allocated=5724.45 MiB、reserved=5906 MiB。GPU 为 RTX 4060 Laptop。

保存前、独立进程加载、打包目录独立入口三份预测 CSV 完全一致，FP32/batch=1 概率最大差0。200行编号/顺序/方向、有限性和[0,1]范围检查通过。`p_machine=softmax(logits)[:,1]`，human=0、machine=1。

使用 A 的 `data_tools/evaluate.py`，阈值0.5，显式 `--short-by word --short-threshold 100`：

| 指标 | 结果 |
| --- | ---: |
| 全体宏 F1 | 0.388379 |
| 短文本宏 F1（30条） | 0.400000 |
| 临时加权分 0.7×全体+0.3×短文本 | 0.391865 |
| human 召回率 | 0 |
| machine 召回率 | 1 |

**200条全部被判为machine。** 本轮只通过真实数据训练、保存、加载和评估链路验收，检测质量不足。短文本按 `len(text.split())<100`，属于临时空白计词规则，不能当作官方多语言规则或官方分数。仅评估 smoke_dev，不与完整dev指标混用。

## 文件与复核命令

`manifest.json` 列出交接文件的格式来源、字节数及 SHA-256；此目录不包含训练正文、模型权重或个人路径。主要产物为 `dev_predictions.csv`、`training_arguments.json`、`run_summary.json`、`metrics.json`、`a_evaluation_report.txt` 和 `packaged_inference_check.json`。

在仓库根目录、各人自己的 Python 环境运行：

```powershell
# A/C复核本次200条预测，显式沿用相同临时短文本规则
python data_tools/evaluate.py --gold data/processed/v1/smoke_dev.jsonl --pred handoffs/B/b_real_smoke_50/dev_predictions.csv --threshold 0.5 --short-by word --short-threshold 100

# 重训时使用新的输出目录，原始模型需已按固定revision准备
python main_model/train.py --train-file data/processed/v1/smoke_train.jsonl --dev-file data/processed/v1/smoke_dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_real_smoke_50_repeat --data-kind a-internal --data-manifest data/processed/v1/manifest.json --evaluation-script data_tools/evaluate.py --seed 42 --max-steps 50 --max-length 256 --batch-size 1 --gradient-accumulation 16 --learning-rate 2e-5

# B本地现有模型的独立推理；输出文件必须尚不存在
python main_model/predict.py --model-dir runs/b_real_smoke_50/checkpoint --input data/processed/v1/smoke_dev.jsonl --output runs/b_real_smoke_50/dev_predictions_manual.csv --precision fp32 --batch-size 1
```

本地模型包在 `runs/b_real_smoke_50/checkpoint/`，已包含模型、分词器、标签映射、训练参数、来源、数据manifest、推理代码、共享依赖说明和文件哈希清单。它不是最终共享模型，未上传权重；不要求其他成员照搬本机环境。

## 下一步

B04流程检查完成。先补B05周期保存/完整断点恢复，再按A固定脚本复现并核对完整划分，为全量训练确定步数及评估周期。只用内部dev比较更充分训练与类别加权等少量方案，不基于本轮小样直接冻结阈值或宣布模型有效。官方多语言计词/阈值仍待A核实。
