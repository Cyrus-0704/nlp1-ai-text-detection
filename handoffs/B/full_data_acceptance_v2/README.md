# internal-v2：B独立验收通过（2026-10-16）

A交付提交 `574eed6079d2f1b31012f45b74c2f4e72c4d6ed5`，[PR #3](https://github.com/Cyrus-0704/nlp1-ai-text-detection/pull/3)。验收时PR仍为OPEN，main为 `ab34e8e`，因此本报告不代表main已经接收v2。只读快照保存在本机忽略目录，没有合并A个人分支或修改其脚本/清单/划分。

## 实际检查

- 原始源文件大小与SHA-256匹配；A原样修复脚本、seed42、min_len20完整复现。
- train119842条（human43922/machine75920）、dev15086条（human5495/machine9591）；六字段、唯一编号和标签约定通过。
- train/dev/test的字节数、SHA-256均完全匹配A的v2 manifest，旧版的复现差异已消除。
- train/dev的sample_id、group_id、空白规范化正文以及忽略大小写规范化正文交集均为0，旧版发现的正文隔离阻塞已消除。
- 新smoke_train1000/1000、smoke_dev200/200与对应正式集合逐条精确匹配；各自文件大小、哈希、样本/类别数符合清单。
- `status=PASS_TRAIN_DEV_ACCEPTANCE`，`training_gate_passed=true`，`failed_checks=[]`。

内部test仅作为不透明字节流核验大小和SHA-256，没有解析样本或评估性能。生成脚本的示例日志未阅读、未上传；官方验证集未访问。A脚本内部的三集合隔离断言成功完成；B独立正文检查范围为train/dev。

## 复现

PR #3合入main并同步后，在仓库根目录使用新输出路径：

```powershell
python main_model/verify_data_delivery.py --manifest data/processed/v2/manifest.json --raw-file data.jsonl --split-dir .local/internal-v2-repeat --output-dir runs/b_acceptance_v2_repeat --delivery-commit 574eed6079d2f1b31012f45b74c2f4e72c4d6ed5 --reproduce
```

本次为合并前审查，使用 `--preprocess-script` 指向从该A提交导出的原样脚本，manifest亦来自同一提交；小样从该提交对应LFS对象取得。实际检查器指纹与运行时未提交状态见execution_provenance.json。

## 当前情况与下一步

本阶段只做数据验收，没有启动新GPU训练、没有新的模型性能分数。旧v1小样分数只作历史记录，不作为v2成绩。

下一步：PR #3合入main、B同步后，从固定原始XLM-R（revision `e73636d4f797dec63c3081bb6ed5c7b0bb3f2089`）开始v2小样50步完整流程，再以固定预算开展完整train/dev的无加权/加权对照。阈值0.5；短文本显式指定空白计词少于100词并标注临时口径。内部test继续封存，不用来选参数。

本目录只交接验收摘要和指纹证据，不含正文、完整划分、私有日志或模型权重。
