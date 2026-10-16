# internal-v2 小样50步：流程通过，性能不足

A修复PR #3已合并main，提交 `7231172ba8e99dfe9dead7b40bf0c47e24ba3f21`；B同步后训练commit `1183ed55e1b357cf1c1d6e743a79663918bb3e24`，启动时main_model代码干净。数据与manifest准确指纹见运行摘要。

- 从固定原始XLM-R开始，1000条train、200条dev，seed42、长度256、batch1、累积16、lr2e-5、无加权、FP16、梯度检查点、num_workers0、确定性模式。
- RTX4060 Laptop真实参与计算；50次有效更新、52次尝试、2次FP16跳步，832个GPU batch；损失全部有限，平均0.649291。
- 活动训练62.326秒、含周期保存67.695秒；0.802更新/秒、13.35样本/秒；训练峰值allocated/reserved为5772.20/5950 MiB。实际序列达到256token。
- 保存前、新进程加载后和独立打包入口的200行FP32/batch1预测CSV字节完全一致，最大概率差0；编号/顺序/行数、machine=1方向、有限[0,1]通过。
- A脚本全体宏F1=0.471588；短文本宏F1=0.475000，临时综合分=0.472612。短文本仅21条，显式空白计词<100、阈值0.5，非官方计词与分数。
- 混淆矩阵（行真实/列预测，human先）为[[6,67],[0,127]]；194/200预测machine，human召回仅0.0822。50步只验证链路，不作为最终性能。

只读取内部train/dev。未读取官方验证集；内部test仅在此前数据验收中作不透明哈希校验，未参与训练调参。v1旧分数不与v2直接比较，亦未从旧小样权重继续训练。

## 命令

```powershell
python main_model/train.py --train-file data/processed/v2/smoke_train.jsonl --dev-file data/processed/v2/smoke_dev.jsonl --model-dir models/xlm-roberta-base --output-dir runs/b_v2_smoke_50_repeat --data-kind a-internal --data-manifest data/processed/v2/manifest.json --evaluation-script data_tools/evaluate.py --seed 42 --max-steps 50 --max-length 256 --batch-size 1 --gradient-accumulation 16 --learning-rate 2e-5 --class-weighting none --save-steps 50 --keep-checkpoints 1 --deterministic
python main_model/predict.py --model-dir runs/b_v2_smoke_50/checkpoint --input data/processed/v2/smoke_dev.jsonl --output runs/b_v2_smoke_50/predict_repeat.csv --precision fp32 --batch-size 1
```

本机模型/分词器/独立推理源码/依赖说明及打包哈希在 `runs/b_v2_smoke_50/checkpoint/`，恢复状态在 `runs/b_v2_smoke_50/checkpoints/step-000050/`，均不上传；本目录仅交接预测与必要配置/摘要，机器路径已转为仓库相对路径。

下一步：完整train/dev固定预算的CE500/Balanced500对照，各从原始预训练模型开始；单seed与不完整epoch仍属于先导实验，未冻结最终模型。
