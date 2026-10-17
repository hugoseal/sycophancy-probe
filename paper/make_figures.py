"""Figures and numbers for the sycophancy paper, computed from the local backup of the v4 and v5 runs.

Needs the raw v4 run (one .json + .pt per question, about 0.8 GB, not in git) produced by
notebooks/1_collect_v4.ipynb. Point SYCOPHANCY_RESULTS at the folder that contains v4_seed0/:

    SYCOPHANCY_RESULTS=/path/to/results python paper/make_figures.py

Writes vector figures to paper/figures/ and the quoted numbers to paper/figures/numbers.json.
"""
import os
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("SYCOPHANCY_RESULTS", ROOT.parent / "results"))
STEERING_CSV = ROOT.parent / "data" / "steering_layer18.csv"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN, RED, GREY = "#5e81b5", "#e19c24", "#8fb032", "#eb6235", "#8f8f8f"
SEED = 0
numbers = {}

# ---------------------------------------------------------------- data
records = [json.loads(p.read_text()) for p in sorted((DATA / "v4_seed0").glob("*.json"))]
complete = [r for r in records if r["status"] == "complete"]
numbers["n_questions"], numbers["n_complete"] = len(records), len(complete)
numbers["labels"] = {b: dict(Counter(r["branches"][b]["label"] for r in complete))
                     for b in ("neutral", "mild", "firm", "authority", "firm_correct")}
example = next(r for r in complete if r["id"] == "mult2x2_0003")
numbers["example"] = {"question": example["question"], "truth": example["truth"], "wrong": example["wrong"],
                      "turn1": example["turn1_reply"],
                      **{b: example["branches"][b]["reply"] for b in ("neutral", "mild", "firm", "authority", "firm_correct")}}

# ---------------------------------------------------------------- Figure 1.2: cave rate by pressure and question type
fams = ["mult2x2", "mult3x2", "mult4x2"]
fam_name = {"mult2x2": r"$2\times2$ digits", "mult3x2": r"$3\times2$ digits", "mult4x2": r"$4\times2$ digits"}
levels = ("mild", "firm", "authority")
rates = {f: [np.mean([r["branches"][b]["label"] == "caved" for r in complete if r["family"] == f]) for b in levels] for f in fams}
overall = [np.mean([r["branches"][b]["label"] == "caved" for r in complete]) for b in levels]
numbers["family_rates"] = {f: [round(x, 3) for x in v] for f, v in rates.items()}
numbers["overall_rates"] = [round(x, 3) for x in overall]
numbers["turn1_accuracy"] = {f: round(float(np.mean([r["status"] == "complete" for r in records if r["family"] == f])), 3) for f in fams}
fig, ax = plt.subplots(figsize=(5.0, 3.1))
for f, c in zip(fams, [BLUE, ORANGE, GREEN]):
    ax.plot([0, 1, 2], rates[f], "-o", color=c, label=fam_name[f], ms=4)
ax.plot([0, 1, 2], overall, "--", color="black", lw=1, label="all 637 questions")
ax.set_xticks([0, 1, 2], ["mild", "firm", "authority"])
ax.set_xlabel("pushback strength")
ax.set_ylabel("cave rate")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "cave_rates.pdf")
plt.close(fig)

# ---------------------------------------------------------------- activations
labelled = [r for r in complete if r["branches"]["firm"]["label"] in ("caved", "held")]
y = np.array([r["branches"]["firm"]["label"] == "caved" for r in labelled], dtype=float)
acts = {k: [] for k in ("firm", "turn1", "neutral", "firm_correct")}
for r in labelled:
    saved = torch.load(DATA / "v4_seed0" / r["activation_file"], map_location="cpu", weights_only=True)
    for k in acts:
        acts[k].append(saved[k].float().numpy())
acts = {k: np.stack(v) for k, v in acts.items()}
n_layers = acts["firm"].shape[1]
numbers["n_firm_labelled"], numbers["n_firm_caved"] = len(y), int(y.sum())


def meandiff_oof(X, labels, seed, score_sets=None):
    """Out-of-fold mean-difference probe scores; optionally score other vector sets with the same fold's probe."""
    out = np.zeros(len(labels))
    others = {k: np.zeros(len(labels)) for k in (score_sets or {})}
    for train, test in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, labels):
        mu = X[train].mean(0)
        w = X[train][labels[train] == 1].mean(0) - X[train][labels[train] == 0].mean(0)
        out[test] = (X[test] - mu) @ w
        for k, V in (score_sets or {}).items():
            others[k][test] = (V[test] - mu) @ w
    return out, others


# ---------------------------------------------------------------- Figure 2.1: layer sweep
curves = {}
for src in ("firm", "turn1", "neutral"):
    curves[src] = [float(np.mean([roc_auc_score(y, meandiff_oof(acts[src][:, l], y, SEED + k)[0]) for k in range(5)]))
                   for l in range(n_layers)]
numbers["sweep_best"] = {k: (int(np.argmax(v)), round(max(v), 3)) for k, v in curves.items()}
numbers["sweep_layer18"] = {k: round(v[18], 3) for k, v in curves.items()}
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ax.plot(curves["firm"], "-", color=BLUE, lw=2, label="firm-pushback state (the probe)")
ax.plot(curves["neutral"], "--", color=ORANGE, lw=1.3, label="neutral state (no number)")
ax.plot(curves["turn1"], "-", color=GREEN, lw=1.3, label="turn-1 state (before pushback)")
ax.axhline(0.76, color=GREY, ls=":", lw=1)
ax.text(0.3, 0.765, "question features only", fontsize=8, color=GREY)
ax.axhline(0.5, color=GREY, lw=0.6)
ax.set_xlabel(r"layer $\ell$")
ax.set_ylabel("cross-validated AUROC")
ax.set_ylim(0.45, 0.92)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "layer_sweep.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 2.2: wrong vs right number, same question
# Reported at layer 18 (the first layer of the probe's plateau, and the steering layer), with layer 27 for comparison.
numbers["same_question"] = {}
rises = {}
for L in (18, 27):
    s_firm, other = meandiff_oof(acts["firm"][:, L], y, SEED, {"firm_correct": acts["firm_correct"][:, L],
                                                              "turn1": acts["turn1"][:, L], "neutral": acts["neutral"][:, L]})
    rise = s_firm - other["firm_correct"]
    rises[L] = rise
    control = other["firm_correct"]
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            boot.append(roc_auc_score(y[idx], s_firm[idx]))
    numbers["same_question"][L] = {
        "same_probe_auroc": {"firm": round(roc_auc_score(y, s_firm), 3), **{k: round(roc_auc_score(y, v), 3) for k, v in other.items()}},
        "caved_share_positive": round(float(np.mean(rise[y == 1] > 0)), 3), "caved_n_positive": int(np.sum(rise[y == 1] > 0)),
        "caved_median": round(float(np.median(rise[y == 1])), 1), "caved_p": float(wilcoxon(rise[y == 1]).pvalue),
        "held_share_positive": round(float(np.mean(rise[y == 0] > 0)), 3),
        "held_median": round(float(np.median(rise[y == 0])), 1), "held_p": float(wilcoxon(rise[y == 0]).pvalue),
        "interaction_auroc": round(roc_auc_score(y, rise), 3),
        "caved_vs_control_auroc": round(roc_auc_score(np.r_[np.ones(int(y.sum())), np.zeros(len(y))], np.r_[s_firm[y == 1], control]), 3),
        "control_vs_held_auroc": round(roc_auc_score(np.r_[np.ones(len(y)), np.zeros(int((1 - y).sum()))], np.r_[control, s_firm[y == 0]]), 3),
        "firm_probe_auroc_ci": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
    }
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
for ax, L, tag in zip(axes, (18, 27), ("(a)", "(b)")):
    r = rises[L]
    bins = np.linspace(np.percentile(r, 0.5), np.percentile(r, 99.5), 36)
    ax.hist(r[y == 0], bins=bins, color=GREY, alpha=0.7, label=f"held ($n={int((1 - y).sum())}$)")
    ax.hist(r[y == 1], bins=bins, color=BLUE, alpha=0.8, label=f"caved ($n={int(y.sum())}$)")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(f"{tag} layer {L}", fontsize=10)
    ax.set_xlabel("wrong-number score $-$ right-number score", fontsize=8)
axes[0].set_ylabel("questions")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "wrong_vs_right.pdf")
plt.close(fig)

# ---------------------------------------------------------------- Figure 3.1: steering dose-response
steer = pd.read_csv(STEERING_CSV)


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


rows = []
for (b, d, m), g in steer.groupby(["branch", "direction", "m"]):
    k, n = int((g.label == "caved").sum()), len(g)
    lo, hi = wilson(k, n)
    rows.append({"branch": b, "direction": d, "m": int(m), "n": n, "caved": k, "rate": k / n, "lo": lo, "hi": hi,
                 "held": int((g.label == "held").sum()), "other": int(g.label.isin(["other", "unparsed"]).sum())})
summary = pd.DataFrame(rows)
numbers["steering"] = summary.round(3).to_dict(orient="records")
cav = summary[(summary.branch == "firm") & (summary.direction == "caving")].sort_values("m")
rnd = summary[(summary.branch == "firm") & (summary.direction == "random")].sort_values("m")
fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.errorbar(cav.m, cav.rate, yerr=[cav.rate - cav.lo, cav.hi - cav.rate], fmt="-o", color=BLUE, capsize=3, ms=4,
            label="caving direction")
ax.errorbar(rnd.m + 0.4, rnd.rate, yerr=[rnd.rate - rnd.lo, rnd.hi - rnd.rate], fmt="D", color=GREY, capsize=3, ms=4,
            label="random direction, same size")
ax.axhline(cav[cav.m == 0].rate.iloc[0], color=GREY, ls=":", lw=1)
ax.set_xticks(cav.m)
ax.set_xlabel(r"push strength $m$")
ax.set_ylabel("cave rate under firm pushback")
ax.set_ylim(0, 0.7)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG / "steering.pdf")
plt.close(fig)

base = steer[(steer.branch == "firm") & (steer.direction == "caving") & (steer.m == 0)].set_index("id").label
paired = {}
for d, m in [("caving", -8), ("caving", -4), ("caving", 4), ("caving", 8), ("random", -8), ("random", 8)]:
    cur = steer[(steer.branch == "firm") & (steer.direction == d) & (steer.m == m)].set_index("id").label.reindex(base.index)
    paired[f"{d}{m:+d}"] = {"started": int(((base != "caved") & (cur == "caved")).sum()),
                            "stopped": int(((base == "caved") & (cur != "caved")).sum())}
numbers["steering_paired"] = paired
fam_steer = steer[(steer.branch == "firm") & (steer.direction == "caving") & steer.m.isin([0, 8])]
numbers["steering_family"] = fam_steer.assign(c=fam_steer.label.eq("caved")).pivot_table(
    index="family", columns="m", values="c", aggfunc="mean").round(3).to_dict()

(FIG / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
print(json.dumps({k: v for k, v in numbers.items() if k not in ("steering", "example")}, indent=1, default=float))
