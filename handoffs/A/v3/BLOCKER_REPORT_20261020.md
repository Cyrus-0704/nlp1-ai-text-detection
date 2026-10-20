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
