# 数据版本与共享

- 原始文件在仓库根目录 `data.jsonl`，通过 LFS 存储；不要修改或重复上传多份。
- `manifest.raw.json` 记录原始文件字节数与SHA-256，不含文本内容。下载后可用 `Get-FileHash .\data.jsonl -Algorithm SHA256` 核对。
- A 负责交付 `data/processed/v1/` 等冻结版本。每版含 train/dev/test、约1000/200真实小样、manifest与清洗审计；其他成员不自行重划。
- 新数据版本使用新目录，不覆盖旧版；manifest写相对路径、来源、哈希、数量、分组与划分规则、seed和生成脚本。
- 个人试验数据与调试缓存留在本地忽略目录，例如 `.local/`、`runs/`；只有交接需要的数据进入正式版本目录。
- `*.jsonl`、`*.csv`、Parquet等已配置LFS。按需 `git lfs pull --include="data/processed/v1/**"`。
- 官方验证集不得用于内容探查、分布统计或训练调参；当前仓库未包含官方验证集。
