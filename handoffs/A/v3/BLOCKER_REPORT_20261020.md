# V3-20261020 A 机阻塞报告：T0 pilot 被冻结代码 GPU 独占检查阻断（Windows WDDM）

日期：2026-10-20。owner=A、seed=43、workspace ROOT=`E:\数据挖掘\.local\workspaces\nlp1-v3-A\v3-code`（E 盘 ASCII 同盘布局）。本报告只登记事实与证据，不改判任何验收状态，等待 B 决定后续。

## 已通过的前置步骤（全部退出 0）

| 步骤 | 结果 | 关键证据 |
| --- | --- | --- |
| `tools/v3_bootstrap.py verify-code --root .` | PASS_CODE_PACKAGE_HASHES_ONLY（35 文件） | manifest_sha256 `05c3b35c0f13e46fe5e49509ae3fbc1e4c0bcc62b91753547fad3d79f8aaad6d` |
| 旧机配置备份 | 失效 `.local/v3/A.json`（绑定 `C:\Users\LENOVO\Desktop\数据挖掘\...` 旧路径）移入 `.local/v3/legacy_c_20261020/A.json`；`A.preflight.json` 不存在，无需备份 | 本机保留 |
| `v3_delivery.py configure --owner A --model-dir E:\数据挖掘\models\mdeberta-v3-base-a0484667 --data-dir E:\数据挖掘\data\processed\v2` | PASS，重写 `.local/v3/A.json`（machine 哈希不变：`b39551694a784dbfd0b7653730c9329f80c97cd41425f022d70029b20ec7500b`，主机名+解释器未变） | 4 文件模型 + 3 文件冻结数据字节与 SHA 全部匹配（train 282022850 B、dev 34861480 B、smoke_dev 200 行） |
| `v3_delivery.py preflight --owner A --cuda` | **PASS_LOCAL_CUDA_PREFLIGHT** | RTX 4060 Laptop GPU 8GB、驱动 591.86、device_count=1；Python 3.11.17（`C:\Users\LENOVO\anaconda3\envs\zyy-gpu\python.exe`）；11 项冻结包版本逐一精确匹配（torch 2.13.0+cu126）；E 盘空闲 ≈103 GiB ≥ 90 GiB |

## 失败步骤

`python main_model/v3_run.py pilot --owner A --arm T0 --execute`（后台，控制台日志本机留存 `.local/v3_pilot_T0_console.log`）在 **任何训练/子进程启动之前** 失败于 `require_idle_cuda()`：

```
RuntimeError: Another CUDA workload exists; preserve it and stop: ['2748, C:\Windows\System32\dwm.exe', ...共 29-31 条...]
```

## 根因（结构性，非临时占用）

- 冻结代码 `v3_run.py` 的 `require_idle_cuda()` 将 `nvidia-smi --query-compute-apps=pid,process_name` 的全部输出视为冲突（白名单仅自身进程树）。
- 本机 Windows + WDDM 模式（RTX 4060 Laptop，驱动 591.86）下，该查询枚举**所有持有 GPU 上下文的进程**，包括桌面合成器 `dwm.exe`、`explorer.exe`、ShellHost、SearchHost 等系统进程；`dwm.exe` 在活动桌面会话中恒存在且不可关闭。
- 因此该检查在本机**结构上无法通过**，与是否关闭 Steam/QQ/Edge/Wallpaper Engine 等用户程序无关（实测关闭前查询列表 ≥29 条，含系统进程）。
- 已实测复跑同一查询确认列表稳定存在 `dwm.exe`（PID 2748）。

## 现场保留与后果

- `runs/v3/V3-20261020-A-s43-T0/pilot/` 保留 `protocol.json`（含完整签名）、`ledger.json`（仅初始化，无任何子 stage）、`failure.json`；**零训练更新、零 checkpoint、零子进程产物**。
- 冻结 `pilot()` 使用 `root.mkdir(exist_ok=False)`：T0 pilot 目录已存在，**同一 experiment_id 无法按现冻结代码重跑**；T1/T2 未启动（同一检查点必然同样失败，且会烧毁其目录）。
- 失败证据已按协议打包：`handoffs/A/v3/V3-20261020-A-s43-T0/`（含 `delivery_manifest.json`）与 `V3-20261020-A-s43-T0-evidence.zip`（6616 B，SHA-256 见同名 `.sha256`），execution 状态 `PILOT_FAILED_FORMAL_NOT_STARTED`。

## 请 B 决定的事项

1. **代码修复**：`require_idle_cuda()` 的 Windows WDDM 兼容方案（例如按 v2 `gpu_guard.py` 思路过滤项目相关 python 进程 + GPU 显存/利用率阈值判断，或改用仅枚举 CUDA 上下文的可靠接口）；修复需重新出包走 verify-code 身份链。
2. **T0 目录处理**：现有 `runs/v3/V3-20261020-A-s43-T0/pilot/`（pre-start 失败、无任何训练）如何重领——原目录清理、attempt 子目录还是新约定，由 B 明确，A 不自行开新目录掩盖失败。
3. **环境侧替代方案是否接受**（不改代码的前提下验证）：Legion 混合模式让 iGPU 接管显示 + 退出 NVIDIA App/Overlay、停 NvContainer 后，`--query-compute-apps` 是否可清空；若 B 认可该前置条件，A 可先做只读验证再请示重跑。
4. **T1/T2**：在 1-3 落定前不启动。

## 边界声明

本次未执行 7500 步正式长训练、未触碰内部 test / 官方 val / 外部封存集；configure/preflight 均为只读校验+本地配置写入；除 `.local/v3/` 本机配置与失败现场外无其他写入。

---

# 后续（同日）：runtimefix1 部署通过，T0/T1 pilot 被 RAM 底线停止（均零训练），T2 槽位保留干净

## runtimefix1 领取与重验（全部退出 0）

- 收到 B 的 `packages-runtimefix1/v3-code-20261020.zip`（232813 B，SHA-256 `4821e8a2e1c54341422fe397141da9cd74df02ef8ccb8082d1f72c1dc9d95977`，与 `delivery.json` 冻结记录一致；随附 `.sha256` 与 `delivery.json` 现存仓库根目录，属 B 的分发产物，A 未纳入本人分支提交）。
- 按 RUNTIME_FIX 要求**旧目录 `v3-code` 原样保留（含首轮 T0 失败证据），修正版在新隔离目录 `v3-code-runtimefix1` 领取**；模型/数据经本机配置复用 E 盘同 SHA 原件。
- `verify-code` PASS（36 文件，manifest `1162db643c1e985adb698e177bd83666e553ca92777b7d6033abfb5751fceab4` = delivery.json 记录）；`configure` PASS（同 E 盘路径）；`preflight --cuda` = **PASS_LOCAL_CUDA_PREFLIGHT**（新代码身份链完整）。
- 新 `require_idle_cuda()` WDDM 分支**实测通过**：桌面/浏览器等非 Python GPU 进程不再误判，pilot 得以进入训练子阶段。

## T0（runtimefix1 工作区）：RAM 底线停止（父监控，零训练）

- 启动前 `psutil.virtual_memory().available` = 5.88 GiB，低于进入训练所需水位；父级 `ProgressMonitor.check`（30s 周期）在第一个子 stage 启动前判定 `disk<20GiB 或 ram<4GiB` 越线（磁盘 103.2 GiB 充足，实为 RAM）。`runs/v3/V3-20261020-A-s43-T0/pilot/` 留有 protocol/ledger/failure.json，**零更新、零 checkpoint**。

## T1（runtimefix1 工作区）：RAM 底线停止（子监控，零更新）

- 用户释放内存至启动前 available=5.88 GiB 后启动；通过 GPU 检查与父监控，`resource50` 子进程启动。子进程资源监控（0.2s 采样，冻结门槛 `minimum_ram_gib=4`，见 matrix.json `_limits`）记录：
  - 首个采样：子进程 RSS 1.17 GB、available 4.67 GiB；
  - 28 个采样后（RSS 涨至 1.85 GB，**模型尚未加载完**）：available 跌至 3.59 GiB < 4 GiB → `Resource floor breached; preserve checkpoint and propose an explicit change` → 训练未开始即停。
- 全链证据：`runs/v3/V3-20261020-A-s43-T1/pilot/`（protocol/ledger/failure + `resource50/attempt-001/resources.jsonl` 28 行采样、console.log）；控制台本机留存 `.local/v3_pilot_T1_console.log`。
- 已按协议打包：本工作区 `handoffs/A/v3/V3-20261020-A-s43-T1/` + `-evidence.zip`（10631 B，SHA `5c77e27e8514ff531d7bee9dcac07ecc887402f9fb0826d4533457318c409324`），状态 `PILOT_FAILED_FORMAL_NOT_STARTED`。

## 量化结论（供预算/方案决策）

- 本机总物理内存 15.7 GB；pilot 子进程（torch + mDeBERTa 权重加载 + 父进程本体 + 系统波动）从启动到越线实测消耗 ≈2.29 GiB 且未到峰（估峰值 ≈3.1 GiB）。要在全程稳住 4 GiB 底线，**启动前 available 需 ≥7.5 GiB**；实测多次清理后最高只到 7.22 GiB（差 0.28 GiB）。
- **T0/T1 pilot 目录均已烧毁（两次失败均零训练更新）**；冻结 `pilot()` 为一次性（`exist_ok=False`，无 attempt 参数）。**T2 目录保持干净未动**——用户选定留给 B 决定后使用。
- 结合 B 在 RUNTIME_FIX 中登记的 T0 实测（0.1133 有效更新/s → 7500 步 ≈18.4h > 10h 上限），A 机（同为 4060 Laptop）大概率触发同一预算门槛；**预算是否调整属用户决定权，A 不改任何冻结参数**。

## 请 B 决定的事项（合并更新）

1. **RAM 底线方案**：`minimum_ram_gib=4` 对 16 GB 内存笔记本是否放宽/改为按 commit 或按进程 RSS 判定，或提供低内存机器的替代门槛；如出包需同步 bump 版本以生成新 experiment 目录。
2. **已烧毁的 T0/T1 pilot 目录重领方式**（两次均零训练）：attempt 机制、目录清理约定或版本 bump，由 B 明确；A 不自行清理或重开目录。
3. **T2 槽位**：A 机保持干净，等待 1-2 落定后一次性使用；启动前 A 将确保 available ≥7.5 GiB。
4. **10h 预算**：B 机已实测 18.4h 估值；是否由用户调整预算后设计新 attempt，与 1-3 一并定。

## 边界声明（追加）

runtimefix1 回合同样零推理/零训练（T1 子进程在加载权重阶段即停，未执行任何优化器更新）、未触碰内部 test / 官方 val / 外部封存集；未修改任何冻结文件；所有失败现场原样保留。
