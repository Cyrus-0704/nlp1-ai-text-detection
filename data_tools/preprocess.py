# -*- coding: utf-8 -*-
"""
preprocess.py —— "大模型生成文本检测与生成器归因" 竞赛 数据处理与划分脚本

功能总览
--------
1. 字段鲁棒兼容 (异构数据):
   - 人类文本: human_text > text > abstract
   - 机器文本: machine_abstract > machine_answer > machine_text
   - 列表字段 (如 *_reviews) 逐项展开; 生成器名取自字段前缀, 人类统一为 'human'
   - 机器文本仅在 "确实以对应提示词模板开头" 时剥离该模板, 绝不按固定长度截断
2. 清洗: 过滤空/过短文本; 单侧为空时保留有效侧; 人类样本按来源 ID 与正文双重去重;
   缺失 ID 时用 "正文指纹" 兜底, 再退化为行索引; 机器样本按 (生成器, 正文) 去重
3. 分组 (Group-based): 同一原文 / 来源 ID 归入同一 group_id; 且每条记录所有将
   被导出的规范化正文(含评审等列表展开项)全部注册正文指纹并入组, 任何两条
   记录共享任一最终正文即并入同组, 保证重复正文绝不跨集合泄露
4. 划分: 按 group 做 80/10/10 的 train/dev/test 划分, 固定随机种子 (默认 42)
5. 输出: train.jsonl / dev.jsonl / test.jsonl + 清洗与统计日志
6. 原始保护: 全程只读 data.jsonl, 绝不写入或覆盖

输出字段 (统一格式)
------------------
sample_id : 全局唯一样本 ID
group_id  : 分组 ID (同组必须落在同一集合)
text      : 文本正文
label     : human=0, machine=1
generator : 生成器名 (原始 model 名 / 评审列表前缀 / human)
domain    : 领域 (= 归一化后的 source, 缺失记为 unknown)

运行说明
--------
# 完整运行 (默认读取 data/data.jsonl, 输出到当前目录)
python preprocess.py

# 冒烟测试 (只处理前 2000 条, 快速校验流程)
python preprocess.py --limit 2000 --out-dir _smoke

# 常用可选参数
python preprocess.py --data data/data.jsonl --out-dir . --seed 42 --min-len 20

依赖: 仅使用 Python 标准库 (json / random / hashlib / re / argparse)。
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

# Windows 控制台中文输出兼容
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ----------------------------------------------------------------------------
# 可配置常量
# ----------------------------------------------------------------------------
HUMAN_FIELDS = ("human_text", "text", "abstract")            # 人类文本优先级
MACHINE_FIELDS = ("machine_abstract", "machine_answer", "machine_text")  # 机器文本优先级
ID_FIELDS = ("source_id", "source_ID", "question_id", "answer_id", "q_id", "a_id", "id")
PROMPT_FIELDS = ("prompt", "promt")                          # 注意数据里存在拼写错误 promt

# 注意: 不把 pandas 泄漏列 "Unnamed: 0" 当作 ID, 它只是行索引, 跨形态会撞号

REVIEW_RE = re.compile(r"^([A-Za-z0-9]+)_reviews$")          # xxx_reviews -> 生成器名 xxx
PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")  # {title} 之类占位符
WS_RE = re.compile(r"\s+")

SPLITS = ("train", "dev", "test")
RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}
# 允许在控制台/日志中展示正文示例的集合。internal test 属于内部测试集,
# 按角色 C 的隐私要求, 其正文绝不允许出现在控制台或日志里。
LOG_EXAMPLE_SPLITS = ("train", "dev")


# ----------------------------------------------------------------------------
# 通用小工具
# ----------------------------------------------------------------------------
def iter_texts(v):
    """把字段值规范成文本序列: 字符串 -> 单元素; 列表 -> 逐项展开"""
    if isinstance(v, str):
        yield v
    elif isinstance(v, list):
        for item in v:
            if isinstance(item, str):
                yield item


def norm_text(s):
    """用于去重/分组的正文指纹: 小写 + 合并空白"""
    return WS_RE.sub(" ", s).strip().lower()


def fp(s):
    """正文指纹的 md5, 避免在内存里保存全文"""
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def get_domain(d):
    """domain = 归一化 source, 缺失或空字符串统一记为 unknown"""
    v = d.get("source")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return "unknown"


def get_generator(d):
    """机器样本的默认生成器名 = 原始 model 名 (保留原样, 不做家族归一化)"""
    v = d.get("model")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return "unknown"


def pick_id(d):
    """按优先级挑选来源标识, 返回 (字段名, 字符串值); 无则返回 (None, None)"""
    for f in ID_FIELDS:
        v = d.get(f)
        if isinstance(v, str):
            if v.strip():
                return f, v.strip()
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            return f, str(v)
    return None, None


def fill_placeholders(tpl, d):
    """把提示词模板中的 {xxx} 用记录里同名字段填充, 无则填空串"""
    def _rep(m):
        v = d.get(m.group(1))
        if isinstance(v, str):
            return v
        if v is None:
            return ""
        return str(v)
    return PLACEHOLDER_RE.sub(_rep, tpl)


def build_prompt_candidates(d):
    """构造该记录可能的提示词模板(含占位符填充版本)"""
    raw = None
    for f in PROMPT_FIELDS:
        v = d.get(f)
        if isinstance(v, str) and v.strip():
            raw = v
            break
        if isinstance(v, list):
            raw = next((x for x in v if isinstance(x, str) and x.strip()), None)
            if raw:
                break
    if raw is None:
        v = d.get("prompts")               # PeerRead 形态的 prompts 列表
        if isinstance(v, list):
            raw = next((x for x in v if isinstance(x, str) and x.strip()), None)
    if raw is None:
        return []

    cands = set()
    for c in (raw, fill_placeholders(raw, d)):
        c = c.strip()
        if c:
            cands.add(c)
    return list(cands)


def strip_prompt(text, cands):
    """
    仅当 text 确实以某个提示词模板开头时才剥离该模板。
    返回 (处理后文本, 是否发生剥离)。绝不做固定长度截断。
    """
    if not cands:
        return text, False
    lead = len(text) - len(text.lstrip())
    body = text[lead:]                     # 跳过前导空白后再判断
    best = None
    for c in cands:
        if body.startswith(c) and (best is None or len(c) > len(best)):
            best = c
    if best is None:
        return text, False
    return body[len(best):].lstrip(), True


def collect_raw_samples(d):
    """
    从一条原始记录抽取所有 (文本, label, generator, 来源字段)。
    人类侧取首个可用字段 (列表则逐项展开), 并额外展开 human_reviews;
    机器侧取首个可用字段, 并额外展开非 human 的 *_reviews 列表。
    """
    out = []

    # ---- 人类侧: 首个可用字段优先 ----
    for f in HUMAN_FIELDS:
        items = [t for t in iter_texts(d.get(f))]
        if any(t.strip() for t in items):
            for t in items:
                out.append((t, 0, "human", f))
            break
    # ---- 人类评审列表 ----
    for k, v in d.items():
        m = REVIEW_RE.match(k)
        if m and m.group(1) == "human":
            for t in iter_texts(v):
                out.append((t, 0, "human", k))

    # ---- 机器侧: 首个可用字段优先 ----
    gen = get_generator(d)
    for f in MACHINE_FIELDS:
        items = [t for t in iter_texts(d.get(f))]
        if any(t.strip() for t in items):
            for t in items:
                out.append((t, 1, gen, f))
            break
    # ---- 机器评审列表 (生成器名取自字段前缀) ----
    for k, v in d.items():
        m = REVIEW_RE.match(k)
        if m and m.group(1) != "human":
            for t in iter_texts(v):
                out.append((t, 1, m.group(1), k))

    return out


# ----------------------------------------------------------------------------
# 并查集: 用于把同一原文 / 同一来源 ID / 重复正文的记录合并成同一组
# ----------------------------------------------------------------------------
class DSU:
    def __init__(self):
        self.p = []

    def _ensure(self, x):
        while len(self.p) <= x:
            self.p.append(len(self.p))

    def find(self, x):
        self._ensure(x)
        p = self.p
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            nxt = p[x]
            p[x] = r
            x = nxt
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def process(opts, log):
    data_path = os.path.abspath(opts.data)
    out_dir = os.path.abspath(opts.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---- 原始保护: 输出路径绝不允许指向原始数据文件 ----
    out_paths = {s: os.path.join(out_dir, s + ".jsonl") for s in SPLITS}
    for s, p in out_paths.items():
        if os.path.abspath(p) == data_path:
            raise SystemExit("[错误] 输出路径与原始数据文件相同, 已终止以保护原始数据: %s" % p)
    if not os.path.exists(data_path):
        raise SystemExit("[错误] 找不到数据文件: %s" % data_path)

    dsu = DSU()
    id_map = {}          # (domain, id) -> 首个记录索引
    hash_map = {}        # 正文指纹 -> 首个记录索引
    est_counts = []      # 每条记录的预估有效样本数 (用于按样本量的组划分)
    n_records = 0
    n_bad_lines = 0
    n_merged = 0

    log.append("=" * 72)
    log.append("数据清洗与统计日志")
    log.append("=" * 72)
    log.append("输入文件            : %s" % data_path)
    log.append("输出目录            : %s" % out_dir)
    log.append("随机种子(seed)      : %d" % opts.seed)
    log.append("最短文本长度(min_len): %d" % opts.min_len)
    log.append("记录数上限(limit)   : %s" % ("不限" if opts.limit <= 0 else opts.limit))
    log.append("")

    # ==================== Pass 1: 只读扫描, 构建分组与样本量估计 ====================
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if opts.limit > 0 and idx >= opts.limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_bad_lines += 1
                continue

            i = n_records
            n_records += 1
            dsu._ensure(i)

            # (a) 来源 ID 分组
            _, id_val = pick_id(d)
            if id_val is not None:
                key = (get_domain(d), id_val)
                if key in id_map:
                    if dsu.union(i, id_map[key]):
                        n_merged += 1
                else:
                    id_map[key] = i

            # (b) 正文指纹分组: 该记录"所有将被导出的规范化正文"(人类正文、评审
            #     列表逐项、机器文本; 与 Pass 2 完全相同的剥离/过滤口径)全部注册
            #     指纹并入组。任何两条记录只要共享任一最终正文, 必被并入同组,
            #     保证重复正文绝不跨 train/dev/test 分散。
            cands = build_prompt_candidates(d)
            cnt = 0
            for t, lab, _g, _f in collect_raw_samples(d):
                tt = strip_prompt(t, cands)[0] if lab == 1 else t
                tt = tt.strip()
                if len(tt) < opts.min_len:
                    continue
                cnt += 1
                hk = fp(norm_text(tt))
                if hk in hash_map:
                    if dsu.union(i, hash_map[hk]):
                        n_merged += 1
                else:
                    hash_map[hk] = i

            # (c) 预估有效样本数 (cnt 与上方循环同口径, 直接复用)
            est_counts.append(cnt)

    log.append("[规模]")
    log.append("  有效记录数        : %d" % n_records)
    log.append("  解析失败行数      : %d" % n_bad_lines)
    log.append("  因分组合并的记录对: %d" % n_merged)
    log.append("  唯一来源 ID 数    : %d" % len(id_map))
    log.append("  唯一正文指纹数    : %d" % len(hash_map))

    # ==================== 按组分得 train/dev/test ====================
    group_count = Counter()
    for i, c in enumerate(est_counts):
        if c > 0:
            group_count[dsu.find(i)] += c
    n_groups = len(group_count)

    items = sorted(group_count.items())          # 先按 root 排序, 保证可复现
    random.Random(opts.seed).shuffle(items)      # 固定种子打乱
    total_est = sum(c for _, c in items) or 1

    cur = {s: 0 for s in SPLITS}
    target = {s: RATIOS[s] * total_est for s in SPLITS}
    group_split = {}
    for root, c in items:
        # 贪心: 每次把当前"最缺样本"的集合补上, 使样本量尽量贴近 80/10/10
        s = max(SPLITS, key=lambda k: target[k] - cur[k])
        group_split[root] = s
        cur[s] += c

    log.append("")
    log.append("[分组与划分]")
    log.append("  分组(group)总数   : %d" % n_groups)
    log.append("  预估样本总量      : %d" % total_est)
    for s in SPLITS:
        log.append("  预估 %-5s 样本量  : %d (目标比例 %.0f%%)" % (s, cur[s], RATIOS[s] * 100))

    # ==================== Pass 2: 抽取 -> 清洗 -> 去重 -> 写盘 ====================
    field_use = Counter()          # 各字段实际贡献样本数
    strip_hits = 0                 # 发生提示词剥离的样本数
    drop_short = Counter()         # 被过滤的空/过短文本数(按 label)
    drop_dup_human = 0             # 人类: (来源标识 + 正文) 完全重复
    drop_dup_machine = 0

    seen_human = set()
    seen_machine = set()

    writers = {s: open(out_paths[s], "w", encoding="utf-8") for s in SPLITS}
    stats = {s: {"total": 0, "label": Counter(), "generator": Counter(), "domain": Counter()}
             for s in SPLITS}
    examples = {s: [] for s in LOG_EXAMPLE_SPLITS}   # 仅 train/dev 收集示例, test 不收集
    sample_seq = 0

    # 完整性自检: 记录各集合的正文指纹与 group_id 归属, 写出后做交叉核验
    split_fps = {s: set() for s in SPLITS}
    group_where = {}

    rec_idx = 0          # 与 Pass 1 对齐的记录索引(仅成功解析的记录计数)
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if opts.limit > 0 and idx >= opts.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                i = rec_idx
                rec_idx += 1

                split = group_split.get(dsu.find(i))
                if split is None:            # 该组无有效样本
                    continue
                gid = "g_%07d" % dsu.find(i)
                domain = get_domain(d)
                _, id_val = pick_id(d)

                cands = build_prompt_candidates(d)
                for t, lab, gen, fld in collect_raw_samples(d):
                    if lab == 1:
                        t2, hit = strip_prompt(t, cands)
                        strip_hits += int(hit)
                    else:
                        t2 = t
                    t2 = t2.strip()
                    if len(t2) < opts.min_len:       # 空文本 / 过短文本 -> 丢弃
                        drop_short[lab] += 1
                        continue

                    nt = norm_text(t2)
                    if lab == 0:
                        # 人类样本: 仅当"来源标识 + 正文"同时重复才算重复
                        # (同一来源下的多条不同正文, 如多篇评审, 应当保留)
                        t_sig = fp(nt)
                        sig = ("H", domain, id_val, t_sig) if id_val is not None \
                            else ("H", None, None, t_sig)
                        if sig in seen_human:
                            drop_dup_human += 1
                            continue
                        seen_human.add(sig)
                    else:
                        # 机器样本: (生成器 + 正文) 去重
                        m_sig = ("M", gen, fp(nt))
                        if m_sig in seen_machine:
                            drop_dup_machine += 1
                            continue
                        seen_machine.add(m_sig)

                    rec = {
                        "sample_id": "s_%08d" % sample_seq,
                        "group_id": gid,
                        "text": t2,
                        "label": lab,
                        "generator": gen,
                        "domain": domain,
                    }
                    sample_seq += 1
                    writers[split].write(json.dumps(rec, ensure_ascii=False) + "\n")

                    split_fps[split].add(fp(nt))
                    prev = group_where.setdefault(gid, split)
                    assert prev == split, "同一 group_id 被写入多个集合: %s" % gid

                    field_use[fld] += 1
                    st = stats[split]
                    st["total"] += 1
                    st["label"][lab] += 1
                    st["generator"][gen] += 1
                    st["domain"][domain] += 1
                    if split in examples and len(examples[split]) < 2:
                        examples[split].append((rec["sample_id"], lab, gen, domain, t2))
    finally:
        for w in writers.values():
            w.close()

    # ==================== 日志汇总 ====================
    total_out = sum(stats[s]["total"] for s in SPLITS)
    log.append("")
    log.append("[清洗统计]")
    log.append("  提示词模板剥离次数      : %d" % strip_hits)
    log.append("  过滤空/过短样本 (human)  : %d" % drop_short[0])
    log.append("  过滤空/过短样本 (machine): %d" % drop_short[1])
    log.append("  人类去重(来源标识+正文)  : %d" % drop_dup_human)
    log.append("  机器去重(生成器+正文)    : %d" % drop_dup_machine)
    log.append("")
    log.append("[字段使用统计] (各字段实际贡献的样本数)")
    for k, v in field_use.most_common():
        log.append("  %-18s : %d" % (k, v))

    log.append("")
    log.append("[最终输出统计]")
    log.append("  样本总数 : %d" % total_out)
    for s in SPLITS:
        st = stats[s]
        pct = (st["total"] / total_out * 100) if total_out else 0
        log.append("  --- %s.jsonl --- (%d 条, %.2f%%)" % (s, st["total"], pct))
        log.append("     label   : human=%d, machine=%d" % (st["label"][0], st["label"][1]))
        log.append("     generator: %s" % ", ".join("%s:%d" % (k, v) for k, v in st["generator"].most_common(12)))
        log.append("     domain   : %s" % ", ".join("%s:%d" % (k, v) for k, v in st["domain"].most_common(12)))

    log.append("")
    log.append("[划分完整性校验]")
    # 实测核验: 任两个集合的规范化正文指纹交集必须为 0 (不再仅凭分组推断)
    leak_pairs = []
    for a in range(len(SPLITS)):
        for b in range(a + 1, len(SPLITS)):
            sa, sb = SPLITS[a], SPLITS[b]
            inter = split_fps[sa] & split_fps[sb]
            if inter:
                leak_pairs.append((sa, sb, len(inter)))
    if leak_pairs:
        for sa, sb, n in leak_pairs:
            log.append("  [失败] %s ∩ %s 存在 %d 条重复规范化正文!" % (sa, sb, n))
        raise SystemExit("[错误] 划分完整性校验失败, 存在跨集合重复正文, 已终止")
    log.append("  分组是否跨集合泄露 : 否 (每个 group_id 只出现在一个集合中, 已逐条断言)")
    log.append("  正文是否跨集合重复 : 否 (train/dev/test 两两规范化正文指纹交集均为 0, 实测)")
    log.append("  原始数据是否被修改 : 否 (data.jsonl 全程只读)")

    log.append("")
    log.append("[各集合示例] (仅展示 %s; internal test 正文按隐私要求不输出)" % "/".join(LOG_EXAMPLE_SPLITS))
    for s in LOG_EXAMPLE_SPLITS:
        log.append("  --- %s ---" % s)
        for sid, lab, gen, dom, txt in examples[s]:
            prev = txt if len(txt) <= 120 else txt[:120] + "..."
            log.append("    %s label=%d generator=%s domain=%s" % (sid, lab, gen, dom))
            log.append("      %s" % prev)

    return out_paths, total_out


def main():
    ap = argparse.ArgumentParser(description="大模型生成文本检测与归因: 数据预处理与划分")
    ap.add_argument("--data", default="data/data.jsonl", help="原始数据路径 (只读)")
    ap.add_argument("--out-dir", default=".", help="输出目录 (存放 train/dev/test.jsonl 与日志)")
    ap.add_argument("--seed", type=int, default=42, help="随机种子, 默认 42")
    ap.add_argument("--min-len", type=int, default=20, help="最短文本长度阈值, 默认 20")
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 条记录(冒烟测试), 0 表示不限")
    opts = ap.parse_args()

    log = []
    out_paths, total_out = process(opts, log)
    text = "\n".join(log)
    print(text)
    log_path = os.path.join(os.path.abspath(opts.out_dir), "preprocess_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n[完成] 输出文件:")
    for s in SPLITS:
        print("  %s" % out_paths[s])
    print("  %s" % log_path)


if __name__ == "__main__":
    main()
