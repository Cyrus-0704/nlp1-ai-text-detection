> 用途：粘贴到 GitHub Issue。若 V3 主 Issue 已存在（如 #19/#20），建议整段作为评论回贴主 Issue；若单独开新 Issue，用下方标题+正文。

## 标题

```
[A 回执] V3-20261020 pilot 两轮受阻：WDDM GPU 检查（runtimefix1 已修）+ RAM 底线（16GB 内存机）；T2 槽位保留干净待决
```

## 正文

**A 侧（owner=A、seed=43）两轮 V3 pilot 全记录。两轮均零训练更新、零推理、未触碰内部 test / 官方 val / 外部封存集；未修改任何冻结文件。证据均在 `work/a-data` 分支 `handoffs/A/v3/`（commit `8a7ab40`、`6d28d84`）。**

### 第一轮（旧代码包，工作区 `v3-code`）

- `verify-code` / `configure`（E 盘路径）/ `preflight --cuda` 全部 PASS（RTX 4060 Laptop 8GB、驱动 591.86、torch 2.13.0+cu126、11 项冻结包精确匹配）。
- T0 pilot 在训练启动前被 `require_idle_cuda()` 阻断：Windows WDDM 下 `nvidia-smi --query-compute-apps` 枚举所有持 GPU 上下文进程（含不可关闭的 `dwm.exe`），**结构性无法通过**。
- 失败证据：`handoffs/A/v3/V3-20261020-A-s43-T0/` + `-evidence.zip`（6616 B，状态 `PILOT_FAILED_FORMAL_NOT_STARTED`）。

### 第二轮（runtimefix1，按 RUNTIME_FIX 在新隔离目录 `v3-code-runtimefix1` 领取，旧目录原样保留）

- `verify-code`（36 文件，manifest `1162db643c1e985adb698e177bd83666e553ca92777b7d6033abfb5751fceab4` = delivery.json 记录）/ `configure` / `preflight --cuda` 全部重新 PASS。
- **WDDM 修复实测有效**：新 `require_idle_cuda()` 只把其他 Python GPU 进程视为冲突，pilot 得以进入训练子阶段。
- T0：父监控 RAM 越线（启动前 available 5.88 GiB 不足），零更新即停。
- T1：子监控（`minimum_ram_gib=4`，0.2s 采样）在权重加载中越线——available 从 4.67 跌至 3.59 GiB（子进程 RSS 1.85 GB 且模型未加载完），零更新即停。28 行 `resources.jsonl` 采样留证。
- T1 失败证据：`handoffs/A/v3/V3-20261020-A-s43-T1/` + `-evidence.zip`（10631 B，SHA `5c77e27e8514ff531d7bee9dcac07ecc887402f9fb0826d4533457318c409324`）。

### 量化结论

- 本机总内存 15.7 GB；pilot 子进程从启动到越线实测吃 ≈2.29 GiB 且未到峰（估峰值 ≈3.1 GiB，含父进程与系统波动）。要全程稳住 4 GiB 底线，**启动前 available 需 ≥7.5 GiB**；多轮清理后实测最高 7.22 GiB（差 0.28 GiB）。
- T0/T1 pilot 目录均已烧毁（两次失败均零训练；冻结 `pilot()` 一次性语义 `exist_ok=False`，无 attempt 参数）。**T2 目录保持干净未动**（用户选定留给本轮决策后一次性使用）。
- 结合 RUNTIME_FIX 中 B 机 T0 实测（0.1133 有效更新/s → 7500 步估值 ≈18.4h > 10h 上限），A 机大概率触发同一预算门槛。

### 请 B/队长决策

1. **RAM 门槛**：`minimum_ram_gib=4` 对 16 GB 内存笔记本是否放宽（或改按 commit/进程 RSS 判定）；如出包请同步 bump 版本号以生成新 experiment 目录。
2. **T0/T1 重领方式**（两次均零训练）：attempt 机制、目录清理约定或版本 bump，A 不自行清理或重开目录。
3. **T2 使用时机**：目录干净，等 1-2 落定后一次性使用；启动前 A 将确保 available ≥7.5 GiB。
4. **10h 预算**：是否由用户调整预算后设计新 attempt。

完整细节见 `handoffs/A/v3/BLOCKER_REPORT_20261020.md`（含两轮证据链与 SHA）。
