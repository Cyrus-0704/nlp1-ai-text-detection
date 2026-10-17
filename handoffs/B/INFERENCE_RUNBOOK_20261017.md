# 第一版候选：离线预测与输入输出核验

本说明面向使用已训练模型的接收人，无训练步骤。候选与尚未关闭的验收问题见[冻结草案](RELEASE_FREEZE_DRAFT_20261017.md)。当前官方val仅允许冻结后的最终推理/必要格式检查，内部test由A在冻结后执行；下面的示例使用人工占位输入，不使用这些封存集合。

## 文件与环境

使用用户另行发送的模型ZIP，先比较同名.sha256，并按包内接收说明解压、核对17个包文件。模型包不在本PR内，git pull不会取得权重；不要重新下载原始模型代替已训练权重。原bundle.json SHA必须是`5775334edda46069d45f7eaade2222bea2c7393d19c6faae44a3ed4a256ab50a`。

在所选工作目录准备以下结构；目录名称可变，命令路径相应调整。辅助检查脚本可以单文件复制，只有Python标准库依赖。保持原模型目录字节不变。

```text
model_bundle/                 # 已解压的b_v2_fusion_s43_2000目录
  bundle.json
  B/                         # 已训练权重、配置与分词器
  C.joblib
  fusion_inference.py / common.py / gpu_guard.py
  verify_fusion_output.py / requirements.txt / references/
check_inference_io.py         # 来自main_model/check_inference_io.py，放包外
input.jsonl
```

已验证的Python版本为3.11系列。接收机器先安装合适的PyTorch CPU或CUDA构建，再按包内requirements.txt安装其余固定依赖；已安装且验收过的环境直接复用。CPU选--device cpu，CUDA选--device cuda；全量XLM-R CPU推理可能较慢，GPU速度与显存以接收机器为准。环境安装时可能需要网络，模型推理本身使用本地文件。

```powershell
python -m pip install -r model_bundle/requirements.txt
python -m pip check
```

## 输入与预检

UTF-8（可带BOM）JSONL或CSV，每条必须有sample_id和text；编号必须非空、唯一且为字符串，保留前导零；正文必须为非空字符串。JSONL每行一个对象，CSV含表头并对逗号/换行正确加引号。可保留其他字段，但推理只使用正文。检查器不修复、去重、排序或规范化数据；不根据label计算指标。

人工示例，仅用于接口演示，不能当作模型性能测试：

```jsonl
{"sample_id":"0002","text":"这是一条用于检查接口的示例文本。"}
{"sample_id":"0001","text":"Another example used only to demonstrate the input format."}
```

先在包外运行预检；报告的父目录须已存在，每次报告选择新文件名。

```powershell
python check_inference_io.py --input input.jsonl --output input_check.json
```

## 离线预测与结果检查

确认模型包、输入和环境后执行一个设备命令，输出目录必须不存在。已验收smoke/dev不用因本说明重新运行。

```powershell
# CUDA机器
python model_bundle/fusion_inference.py --bundle model_bundle --input input.jsonl --output predictions_run1 --device cuda
# 无CUDA机器使用下一条，两个命令只选一个
python model_bundle/fusion_inference.py --bundle model_bundle --input input.jsonl --output predictions_run1 --device cpu

python check_inference_io.py --input input.jsonl --predictions predictions_run1/predictions.csv --output output_check.json
```

最终内部接口文件为predictions_run1/predictions.csv：表头严格为sample_id,p_machine，每个输入编号一行、顺序与输入相同；p_machine为machine=1概率，必须有限且位于[0,1]。不四舍五入为0/1，不通过电子表格软件重新保存导致前导零丢失。内部阈值0.5只用于评估，输出CSV始终保留概率。

目录同时保存B.csv、C.csv、prediction_manifest.json和stages.json。manifest记录输入、模型包、预测文件指纹和运行设置。若程序非零退出或存在failure.json，保留失败证据，排查后使用新输出目录；不要把部分CSV当成完整结果。

## 检查能证明什么

| 工具/记录 | 核验范围 | 不代表什么 |
| --- | --- | --- |
| check_inference_io.py预检 | 字段类型、非空正文、唯一字符串编号、CSV结构、JSON重复键 | 数据来源合法性或划分隔离 |
| check_inference_io.py输出核验 | 精确列名/列序、原编号/顺序/行数、有限[0,1]、输入/预测SHA | 模型质量、概率方向、融合公式或使用了指定模型 |
| 原包verify_fusion_output.py | 固定smoke/dev与包内参考概率的逐点容差/分类一致性 | 任意新输入的格式检查；无参考编号时会拒绝 |
| fusion_inference.py及manifest | 校验包文件、模型类别映射、固定融合并记录运行指纹 | 官方模板已经适配或A独立验收已经关闭 |

格式检查器不加载模型、读取参考概率或探查类别/生成器分布；通过状态只能写成PASS_INPUT_FORMAT或PASS_INPUT_OUTPUT_FORMAT。概率方向与模型来源须结合受信模型包和实际执行记录确认，不能从CSV数值反推。

## 常见失败与处理

- 编号被改成数字、重复/缺失/乱序：回到原输入和原预测定位转换问题，不能删行或按编号交集掩盖。
- 空正文、字段缺失、重复表头/JSON键、CSV行列错位：修复输入格式并保留数据版本记录，不静默丢弃。
- 概率NaN/无穷/越界、漏行：查看本次阶段及失败记录；不截断概率或补默认值。
- 包SHA不符：停止，确认传输完整性和确切版本；不要改manifest让错误文件通过。
- 已存在输出/检查报告：选新路径，避免覆盖已验收结果。
- CPU/GPU参考概率超出既定容差：保留报告、环境和脚本指纹交B复核。不能仅因分类相同自行改容差。

正式比赛CSV需A确认准确模板、编号映射与概率方向后适配；当前内部两列格式不是对官方要求的替代。最终发行需另附真实评估记录、报告与冻结确认，不能仅凭本说明宣称交付验收完成。
