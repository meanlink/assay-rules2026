"""检验失败项的Apriori频繁项集与关联规则，不作因果断言。

探索/留出复验口径（防止同一批数据既找规则又证规则）：
- 按复检批次 batch_id 的稳定哈希固定七三分组，同一 batch_id 的全部记录
  永远落在同一组，分组与记录顺序、阈值调整无关，重算或打乱记录均不变。
- 探索组(约70%)只负责筛选频繁项集与规则；复验组(约30%)只核验探索组
  给出的规则，不再挖掘新规则。
- 每组分别计算覆盖(支持度)、置信度与提升度。复验组样本不足或无法支持
  规则时保留该规则并明确标注，不删除线索。
"""
from itertools import combinations
import hashlib
import numpy as np

ITEMS = ["heat_fail", "ph_fail", "viscosity_fail", "ash_fail", "color_fail"]
NAMES = {"heat_fail": "热稳定失败", "ph_fail": "pH偏离", "viscosity_fail": "黏度失败", "ash_fail": "灰分失败", "color_fail": "色差失败"}

# 固定七三分组：探索70% / 留出复验30%。阈值变化、记录打乱后仍按同一口径归组。
EXPLORATION_RATIO = 0.70
HOLDOUT_RATIO = 0.30
# 复验组前项出现低于该批次数时，指标可算但标注“样本不足，谨慎解读”。
HOLDOUT_MIN_ANTECEDENT_BATCHES = 5
MAX_ITEMSET_SIZE = 3


def metadata():
    return {
        "name": "检验失败组合追因", "short": "assay-rules",
        "tagline": "从复检批次找出高支持度的共败组合，为原因复核提供线索。",
        "client": "恒谱材料检测室 · 9名检验员 · 240批复检记录",
        "workflow": "实验室按单项失败率开会，检验员临时筛选同批次失败项来猜测共因。",
        "pains": ["罕见组合容易被误当规律", "高置信度可能只是后项本身常见", "复核假设没有记录支持度与提升度口径"],
        "upstream": "同一批次五项检验复核标签 assay_rechecks_202611.csv",
        "downstream": "质量工程师挑选共败组合追查制样与仪器记录，再安排独立复验",
        "entities": ["批次检验事务", "共败规则试验", "质量复核简报"],
        "dataset_name": "恒谱240批五项检验复核标签",
        "columns": [{"key": "batch_id", "label": "复检批次", "type": "string"}] + [{"key": item, "label": f"{NAMES[item]}(1/0)", "type": "number"} for item in ITEMS],
        "parameters": [{"key": "min_support", "label": "最小支持度(比例)", "type": "number", "default": 0.12, "min": 0.05, "max": 0.5}, {"key": "min_confidence", "label": "最小置信度(比例)", "type": "number", "default": 0.65, "min": 0.2, "max": 0.95}],
        "workbench_title": "检验共败规则搜索", "action_label": "挖掘频繁项集与规则", "unit": "复检批次",
        "accent": "oklch(52% 0.13 350)", "style": "laboratory",
        "seed_story": ["08:30 检验员导入 assay_rechecks_202611.csv 的240批结果", "08:40 确认1表示失败、0表示通过，每批仅保留最终复核结果", "09:00 设最小支持度0.12、最小置信度0.65", "09:01 按批次哈希固定七三分组：探索组挖规则、复验组只核验", "09:10 质量工程师先看复验提升度，再核对规则覆盖批次", "09:20 把支持度提高到0.3重算，批次归组保持不变", "09:30 下载带复验标注的关联规则交制样和仪器小组复核", "15:00 管理员配置每周复检简报的内部归档任务"],
        "method": "每批构成一条失败项事务；按batch_id稳定哈希固定七三分组，同批记录不跨组。探索组以Apriori逐层剪枝(最多3项)挖规则，复验组只核验这些规则。支持度=P(A∩B)，置信度=P(B|A)，提升度=P(B|A)/P(B)；两组分别计算，探索组仅展示提升度>1的规则。",
        "split_method": f"按复检批次batch_id的SHA256哈希取模固定分组：探索组约{int(EXPLORATION_RATIO*100)}%、留出复验组约{int(HOLDOUT_RATIO*100)}%；同一batch_id全部记录同组，分组不依赖记录顺序或分析阈值，重算与打乱记录后归组不变。历史任务保留其计算时的结果与原结论。",
        "limitations": ["共败关联不等于共同根因", "仅在已进入复检的样本中分析，存在选择偏差", "最多3项组合，不挖掘更长规则", "复验组只核验探索组规则，样本不足的规则会标注而非删除"],
    }


def seed_records():
    rng = np.random.default_rng(1012)
    rows = []
    for i in range(240):
        preparation = rng.random() < 0.32
        furnace = rng.random() < 0.20
        heat = rng.random() < (0.92 if preparation else 0.05)
        viscosity = rng.random() < (0.94 if preparation else 0.06)
        ph = rng.random() < (0.72 if preparation else 0.10)
        ash = rng.random() < (0.92 if furnace else 0.04)
        color = rng.random() < (0.88 if furnace else 0.08)
        rows.append({"batch_id": f"HP-R{i+1:03}", **{k: int(v) for k, v in zip(ITEMS, [heat, ph, viscosity, ash, color])}})
    return rows


def split_group(batch_id):
    """批次固定归组：SHA256前8字节取整后对100取模，<70为探索组，否则留出复验组。

    纯函数且只依赖 batch_id，因此同批记录必同组，且不受记录顺序、
    分析参数或时间影响——重算与打乱记录得到完全一致的归组。取8字节
    而非单字节，避免小样本下哈希取模偏斜，保证七三比例稳定。
    """
    digest = hashlib.sha256(str(batch_id).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "exploration" if bucket < int(EXPLORATION_RATIO * 100) else "holdout"


def _parse_transactions(records):
    """校验并把记录转成 (batch_id, 失败项frozenset) 事务，0全通过保留空事务。"""
    transactions = []
    for row in records:
        batch_id = row.get("batch_id")
        if batch_id is None or str(batch_id).strip() == "":
            raise ValueError("每条复检记录必须包含复检批次 batch_id，才能按批次固定分组。")
        try:
            values = [float(row[key]) for key in ITEMS]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("五项检验标记须完整且仅为0或1。") from exc
        if any(value not in (0, 1) for value in values):
            raise ValueError("检验标记仅接受0(通过)或1(失败)。")
        transactions.append((str(batch_id), frozenset(key for key, value in zip(ITEMS, values) if value == 1)))
    return transactions


def _mine_support(transactions, min_support):
    """Apriori逐层连接并以所有(k-1)子集频繁性剪枝，最多3项，返回频繁项集支持度。"""
    total = len(transactions)
    support = {}
    previous = set()
    for item in ITEMS:
        itemset = frozenset([item])
        fraction = sum(itemset <= t for t in transactions) / total
        if fraction >= min_support:
            support[itemset] = fraction
            previous.add(itemset)
    for size in range(2, MAX_ITEMSET_SIZE + 1):
        candidates = {a | b for a in previous for b in previous if len(a | b) == size}
        candidates = {c for c in candidates if all(frozenset(s) in previous for s in combinations(sorted(c), size - 1))}
        current = set()
        for itemset in sorted(candidates, key=lambda s: sorted(s)):
            fraction = sum(itemset <= t for t in transactions) / total
            if fraction >= min_support:
                support[itemset] = fraction
                current.add(itemset)
        previous = current
        if not previous:
            break
    return support, total


def _count(itemset, transactions):
    return sum(itemset <= t for t in transactions)


def _metric_row(antecedent, consequent, joint_count, antecedent_count, consequent_count, total):
    """按给定分组的事务计数计算覆盖/置信度/提升度，样本为0时返回None而非编造。"""
    support_pct = round(joint_count / total * 100, 2) if total else None
    confidence_pct = round(joint_count / antecedent_count * 100, 2) if antecedent_count else None
    if confidence_pct is None or consequent_count == 0:
        lift = None
    else:
        consequent_support = consequent_count / total
        lift = round((joint_count / antecedent_count) / consequent_support, 4) if consequent_support else None
    return support_pct, confidence_pct, lift


def _holdout_evaluation(antecedent, consequent, holdout_tx, holdout_total):
    """复验组只核验探索组给出的规则，不参与挖掘；返回计数、指标与标注。"""
    joint_set = antecedent | consequent
    joint_count = _count(joint_set, holdout_tx)
    antecedent_count = _count(antecedent, holdout_tx)
    consequent_count = _count(consequent, holdout_tx)
    support_pct, confidence_pct, lift = _metric_row(
        antecedent, consequent, joint_count, antecedent_count, consequent_count, holdout_total
    )
    if antecedent_count == 0:
        status, note = "holdout_unobserved", "复验组未出现该规则前项，无法核验，规则保留待后续批次复验"
    elif joint_count == 0:
        status, note = "holdout_unsupported", "复验组前项出现但未再共同失败，复验不支持该规则，规则保留并标注"
    elif lift is not None and lift <= 1 + 1e-12:
        status, note = "holdout_unsupported", f"复验组提升度{lift}不大于1，未呈现正向共败，规则保留并标注"
    elif antecedent_count < HOLDOUT_MIN_ANTECEDENT_BATCHES:
        status, note = "holdout_insufficient", f"复验组前项仅{antecedent_count}批(<{HOLDOUT_MIN_ANTECEDENT_BATCHES}批)，样本不足，指标仅供参考"
    else:
        status, note = "holdout_supported", "复验组样本充足且提升度大于1，规则通过留出复验"
    return {
        "count": joint_count, "antecedent_count": antecedent_count, "consequent_count": consequent_count,
        "support_pct": support_pct, "confidence_pct": confidence_pct, "lift": lift,
        "validation": status, "validation_note": note,
    }


def analyze(records, params):
    try:
        min_support = float(params.get("min_support", 0.12))
        min_confidence = float(params.get("min_confidence", 0.65))
    except (ValueError, TypeError) as exc:
        raise ValueError("支持度与置信度须为有效比例。") from exc
    if not 0.05 <= min_support <= 0.5 or not 0.2 <= min_confidence <= 0.95:
        raise ValueError("最小支持度须0.05~0.5，最小置信度须0.2~0.95。")
    if len(records) < 30:
        raise ValueError("至少需要30批复检事务，避免极少样本误导规则。")

    parsed = _parse_transactions(records)
    # 按批次固定七三分组；同一batch_id的全部事务强制落在同一组。
    exploration_tx, holdout_tx = [], []
    group_of_batch = {}
    for batch_id, items in parsed:
        group = group_of_batch.setdefault(batch_id, split_group(batch_id))
        (exploration_tx if group == "exploration" else holdout_tx).append(items)

    if len(exploration_tx) < 30 or len(holdout_tx) < 1:
        raise ValueError("固定分组后探索组或复验组样本不足，无法完成留出复验，请增加复检批次。")

    # 探索组：唯一负责筛选频繁项集与规则。
    support, exploration_total = _mine_support(exploration_tx, min_support)
    rules = []
    for itemset, joint in support.items():
        if len(itemset) < 2:
            continue
        for length in range(1, len(itemset)):
            for subset in combinations(sorted(itemset), length):
                antecedent = frozenset(subset)
                consequent = itemset - antecedent
                ex_joint = _count(itemset, exploration_tx)
                ex_ant = _count(antecedent, exploration_tx)
                ex_cons = _count(consequent, exploration_tx)
                ex_support_pct, ex_confidence_pct, ex_lift = _metric_row(
                    antecedent, consequent, ex_joint, ex_ant, ex_cons, exploration_total
                )
                if ex_confidence_pct is None or ex_confidence_pct / 100 < min_confidence:
                    continue
                if ex_lift is None or ex_lift <= 1 + 1e-12:
                    continue
                # 复验组：只核验探索组已选出的规则，不产生新规则。
                holdout = _holdout_evaluation(antecedent, consequent, holdout_tx, len(holdout_tx))
                rules.append({
                    "antecedent": " + ".join(NAMES[k] for k in sorted(antecedent)),
                    "consequent": " + ".join(NAMES[k] for k in sorted(consequent)),
                    "count": ex_joint,
                    "support_pct": ex_support_pct,
                    "confidence_pct": ex_confidence_pct,
                    "lift": ex_lift,
                    "holdout_count": holdout["count"],
                    "holdout_antecedent_count": holdout["antecedent_count"],
                    "holdout_support_pct": holdout["support_pct"],
                    "holdout_confidence_pct": holdout["confidence_pct"],
                    "holdout_lift": holdout["lift"],
                    "validation": holdout["validation"],
                    "validation_label": _VALIDATION_LABELS[holdout["validation"]],
                    "validation_note": holdout["validation_note"],
                })
    rules.sort(key=lambda row: (-row["lift"], -row["support_pct"], row["antecedent"], row["consequent"]))
    for rank, rule in enumerate(rules, 1):
        rule["rule_id"] = f"R{rank:02}"

    total = len(parsed)
    combination_count = sum(len(key) >= 2 for key in support)
    holdout_total = len(holdout_tx)
    split_info = {
        "strategy": "batch_hash_fixed_70_30",
        "rule": metadata()["split_method"],
        "exploration_ratio": EXPLORATION_RATIO,
        "holdout_ratio": HOLDOUT_RATIO,
        "total_batches": total,
        "exploration_batches": exploration_total,
        "holdout_batches": holdout_total,
        "exploration_pct": round(exploration_total / total * 100, 2),
        "holdout_pct": round(holdout_total / total * 100, 2),
        "exploration_role": "仅用于筛选频繁项集与规则",
        "holdout_role": "仅用于核验探索组给出的规则，不挖掘新规则",
        "holdout_min_antecedent_batches": HOLDOUT_MIN_ANTECEDENT_BATCHES,
    }
    supported = sum(1 for r in rules if r["validation"] == "holdout_supported")
    insufficient = sum(1 for r in rules if r["validation"] == "holdout_insufficient")
    unsupported = sum(1 for r in rules if r["validation"] in ("holdout_unsupported", "holdout_unobserved"))

    top = rules[:8]
    chart_series = [
        {"name": "探索组提升度（规则挖掘口径）", "values": [row["lift"] for row in top]},
        {"name": "复验组提升度（只核验，无值表示样本不足或不支持）", "values": [row["holdout_lift"] for row in top]},
    ]
    return {
        "summary": f"{total}批复检事务按批次固定七三分组（探索{exploration_total}批/复验{holdout_total}批）；探索组找到{combination_count}个多项频繁组合、{len(rules)}条提升度大于1的规则，其中{supported}条通过留出复验、{insufficient}条复验样本不足、{unsupported}条复验不支持或无法核验（均保留并标注）。",
        "metrics": [
            {"label": "复检批次", "value": total},
            {"label": "探索组批次", "value": exploration_total},
            {"label": "留出复验批次", "value": holdout_total},
            {"label": "探索后规则", "value": len(rules)},
            {"label": "复验通过", "value": supported},
            {"label": "复验样本不足", "value": insufficient},
            {"label": "复验不支持/未观测", "value": unsupported},
            {"label": "探索组最高提升度", "value": max([row["lift"] for row in rules], default=0)},
        ],
        "split": split_info,
        "table": {
            "columns": [
                {"key": "rule_id", "label": "规则编号"},
                {"key": "antecedent", "label": "前项A"},
                {"key": "consequent", "label": "后项B"},
                {"key": "count", "label": "探索-共同失败批次"},
                {"key": "support_pct", "label": "探索-支持度(%)"},
                {"key": "confidence_pct", "label": "探索-置信度(%)"},
                {"key": "lift", "label": "探索-提升度"},
                {"key": "holdout_count", "label": "复验-共同失败批次"},
                {"key": "holdout_antecedent_count", "label": "复验-前项批次"},
                {"key": "holdout_support_pct", "label": "复验-支持度(%)"},
                {"key": "holdout_confidence_pct", "label": "复验-置信度(%)"},
                {"key": "holdout_lift", "label": "复验-提升度"},
                {"key": "validation_label", "label": "复验结论", "badge": True},
                {"key": "validation_note", "label": "复验标注说明"},
            ],
            "rows": rules,
        },
        "chart": {"type": "bar", "labels": [row["rule_id"] for row in top], "series": (chart_series if top else [])},
        "findings": _build_findings(rules, supported, insufficient, unsupported),
        "method": metadata()["method"],
        "split_method": metadata()["split_method"],
        "warnings": _build_warnings(rules),
    }


_VALIDATION_LABELS = {
    "holdout_supported": "复验通过",
    "holdout_insufficient": "复验样本不足",
    "holdout_unsupported": "复验不支持",
    "holdout_unobserved": "复验未观测",
}


def _build_findings(rules, supported, insufficient, unsupported):
    findings = []
    if rules:
        first = rules[0]
        findings.append(
            f"首条规则{first['rule_id']}在探索组覆盖{first['count']}批、置信度{first['confidence_pct']}%、提升度{first['lift']}；"
            f"复验组覆盖{first['holdout_count']}批、置信度{first['holdout_confidence_pct']}%、提升度{first['holdout_lift']}，结论：{first['validation_label']}。"
        )
    else:
        findings.append("探索组当前阈值没有规则通过；可降低阈值，但应先确认复检样本量，且复验组不参与重新挖掘。")
    findings.append(f"留出复验汇总：{supported}条通过、{insufficient}条样本不足、{unsupported}条不支持或未观测；后两类规则保留并标注，供后续批次继续复验。")
    findings.append("探索组负责找线索、复验组只核验，同批记录不跨组，避免用同一批数据既找规则又证明规则。")
    findings.append("提升度大于1表示后项在前项出现时更常见；复验样本不足时指标波动大，仍需对照仪器与制样记录复核。")
    findings.append("反向规则单独计算，A→B与B→A的置信度通常不同。")
    return findings


def _build_warnings(rules):
    warnings = ["规则是统计关联，不能直接归因于材料或仪器故障。", "复验指标仅来自留出复验组，未通过或样本不足的规则已保留并明确标注，不应作为已验证结论。"]
    flagged = [r for r in rules if r["validation"] != "holdout_supported"]
    if flagged:
        warnings.append("以下规则复验未完全支持：" + "、".join(f"{r['rule_id']}({r['validation_label']})" for r in flagged) + "。")
    if not rules:
        warnings.append("探索组未找到满足阈值的规则。")
    return warnings
