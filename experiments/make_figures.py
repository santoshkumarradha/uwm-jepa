"""Regenerate the evidence figures in the paper from bundled experiment outputs.

Reads JSON/CSV from ./data/ and writes PDF + PNG to ../paper/figures/.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "paper" / "figures"


COL = {
    "uwm": "#0072B2",
    "lstm": "#D55E00",
    "gru": "#009E73",
    "ortho_rnn": "#CC79A7",
    "mlp": "#777777",
    "plain": "#E69F00",
    "resid": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#111111",
    "grid": "#E6E2DA",
}

MODEL_LABEL = {
    "uwm": "UWM",
    "lstm": "LSTM",
    "gru": "GRU",
    "ortho_rnn": "Ortho-RNN",
    "mlp": "MLP",
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["TeX Gyre Termes", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.75,
            "axes.edgecolor": COL["black"],
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.55,
            "lines.markersize": 4.2,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 450,
        }
    )


def load_json(name: str):
    with open(DATA / name) as f:
        return json.load(f)


def load_csv(name: str):
    rows = []
    with open(DATA / name) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def despine(ax, grid_axis: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=COL["grid"], linewidth=0.55)
        ax.set_axisbelow(True)


def panel_label(ax, label: str) -> None:
    ax.text(-0.13, 1.04, label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def mean_std_from_agg(agg, model, metric, ks):
    means = [agg[model][f"{metric}_k{k}_mean"] for k in ks]
    stds = [agg[model][f"{metric}_k{k}_std"] for k in ks]
    return np.array(means), np.array(stds)


def line_with_band(ax, x, y, s, color, label, marker="o", linestyle="-"):
    ax.plot(x, y, color=color, marker=marker, linestyle=linestyle, label=label)
    ax.fill_between(x, y - s, y + s, color=color, alpha=0.12, linewidth=0)


def make_invariants() -> None:
    rows = load_csv("invariants.csv")
    ks = np.array([r["k"] for r in rows])
    spec = np.maximum(np.array([r["spec_drift"] for r in rows]), 1e-14)
    purity = np.abs(np.array([r["purity"] for r in rows]) - rows[0]["purity"]) + 1e-14
    entropy = np.abs(np.array([r["entropy"] for r in rows]) - rows[0]["entropy"]) + 1e-14

    fig, ax = plt.subplots(figsize=(3.35, 1.85))
    metrics = [
        (r"spectrum $\|\lambda_k-\lambda_0\|_2$", spec, COL["uwm"], "o"),
        (r"purity $|\mathrm{tr}(\rho_k^2)-\mathrm{tr}(\rho_0^2)|$", purity, COL["plain"], "s"),
        (r"entropy $|S(\rho_k)-S(\rho_0)|$", entropy, COL["gru"], "^"),
    ]
    y = np.arange(len(metrics))[::-1]
    ax.axvspan(1e-14, 3e-7, color="#F1EEE6", zorder=0)
    ax.axvline(3e-7, color="#9A8F7A", lw=0.8, ls=(0, (3, 2)))
    for yi, (label, vals, color, marker) in zip(y, metrics):
        vals = np.maximum(vals, 1e-14)
        ax.hlines(yi, vals.min(), vals.max(), color=color, lw=1.5, alpha=0.85)
        ax.scatter(vals, np.full_like(vals, yi), s=10, color=color, alpha=0.35, linewidths=0)
        ax.scatter(vals.max(), yi, s=28, marker=marker, color=color,
                   edgecolor="white", linewidth=0.45, zorder=4)
        ax.text(4e-4, yi, f"max {vals.max():.1e}", ha="right", va="center", fontsize=6.7, color=color)

    ax.set_xscale("log")
    ax.set_xlim(7e-15, 8e-4)
    ax.set_yticks(y, [m[0] for m in metrics])
    ax.set_xlabel("absolute drift over 30 blind-rollout steps")
    ax.set_title("Joint-state invariants remain at numerical precision")
    ax.text(0.67, 0.05, "precision band", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.5, color="#7A6F60")
    despine(ax, grid_axis="x")
    save(fig, "fig_invariants_verification")


def make_collapse_phase() -> None:
    data = load_json("multiseed_dashboard.json")
    records = data["records"]
    agg = data["agg"]
    order = ["uwm", "lstm", "ortho_rnn", "gru", "mlp"]

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.axhspan(-0.08, 0.05, color="#F4E6D0", zorder=0)
    ax.axvspan(0, 2.0, color="#F4E6D0", zorder=0)
    ax.axhline(0.05, color="#9A8F7A", lw=0.8, ls=(0, (3, 2)))
    ax.axvline(2.0, color="#9A8F7A", lw=0.8, ls=(0, (3, 2)))

    for model in order:
        rs = [r for r in records if r["model"] == model]
        ax.scatter([r["er"] for r in rs], [r["r2"] for r in rs],
                   s=22, color=COL[model], alpha=0.55, edgecolor="white", linewidth=0.35)
        ax.scatter(agg[model]["er_mean"], agg[model]["r2_mean"],
                   s=58, marker="D" if model in {"uwm", "lstm"} else "o",
                   color=COL[model], edgecolor=COL["black"], linewidth=0.55, zorder=4)

    offsets = {
        "uwm": (0.25, -0.03), "lstm": (0.25, 0.02), "ortho_rnn": (0.25, 0.03),
        "gru": (0.18, 0.045), "mlp": (0.18, -0.035),
    }
    for model in order:
        dx, dy = offsets[model]
        ax.text(agg[model]["er_mean"] + dx, agg[model]["r2_mean"] + dy, MODEL_LABEL[model],
                fontsize=7, color=COL[model], va="center")

    ax.text(0.15, -0.015, "collapse region", fontsize=6.8, color="#7A6F60", va="bottom")
    ax.set_xlim(0, 15.4)
    ax.set_ylim(-0.06, 0.98)
    ax.set_xlabel("effective rank of latent covariance")
    ax.set_ylabel(r"held-out probe $R^2$")
    ax.set_title("Collapse diagnostic: rank must carry task information")
    despine(ax, grid_axis="both")
    save(fig, "fig_collapse_phase")


def make_heldout_probe() -> None:
    data = load_json("multiseed_heldout_probe.json")
    order = ["uwm", "lstm", "ortho_rnn", "gru", "mlp"]
    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    x = np.arange(len(order))
    seed_offsets = np.linspace(-0.08, 0.08, 5)
    for i, model in enumerate(order):
        vals = np.array([r["r2"] for r in data["rows"] if r["model"] == model])
        ax.scatter(i + seed_offsets[: len(vals)], vals, s=14, color=COL[model], alpha=0.55, linewidths=0)
        mean = data["agg"][model]["mean"]
        std = data["agg"][model]["std"]
        ax.errorbar(i, mean, yerr=std, fmt="_", color=COL[model], ecolor=COL[model],
                    markersize=16, markeredgewidth=1.7, elinewidth=1.0, capsize=2.5, zorder=4)
        ax.text(i, min(0.965, mean + std + 0.045), f"{mean:.3f}",
                ha="center", fontsize=6.7, color=COL[model])
    ax.plot([0, 1], [0.99, 0.99], color=COL["black"], lw=0.75)
    ax.text(0.5, 1.005, "Welch p=0.70", ha="center", va="bottom", fontsize=6.8)
    ax.axhline(0, color="#9A8F7A", lw=0.75, ls=(0, (3, 2)))
    ax.set_xticks(x, [MODEL_LABEL[m] for m in order], rotation=18, ha="right")
    ax.set_ylabel(r"held-out probe $R^2$")
    ax.set_ylim(-0.18, 1.05)
    ax.set_title("Frozen context representation")
    despine(ax)
    save(fig, "fig_heldout_probe")


def make_blind_rollout() -> None:
    fair = load_json("fairhorizon_retrieval.json")
    residual = load_json("residual_predictor_retrieval.json")
    retention = load_json("retention_3panel.json")

    ks_retrieval = np.array([1, 3, 5, 10])
    ks_retention = np.array([1, 3, 5, 10, 20])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), gridspec_kw={"wspace": 0.32})

    ax = axes[0]
    uwm_m, uwm_s = mean_std_from_agg(fair["agg"], "uwm", "top1", ks_retrieval)
    plain_m, plain_s = mean_std_from_agg(residual["agg"], "plain", "top1", ks_retrieval)
    resid_m, resid_s = mean_std_from_agg(residual["agg"], "residual", "top1", ks_retrieval)
    line_with_band(ax, ks_retrieval, uwm_m, uwm_s, COL["uwm"], "UWM unitary", "o")
    line_with_band(ax, ks_retrieval, plain_m, plain_s, COL["plain"], "LSTM MLP", "s")
    line_with_band(ax, ks_retrieval, resid_m, resid_s, COL["resid"], "LSTM residual", "^")
    ax.axhline(1 / 500, color=COL["mlp"], linestyle=(0, (3, 2)), linewidth=0.8)
    ax.text(9.9, 0.035, "random", color=COL["mlp"], fontsize=6.5, ha="right", va="bottom")
    ax.set_xlabel(r"rollout horizon $k$")
    ax.set_ylabel("top-1 retrieval")
    ax.set_xticks(ks_retrieval)
    ax.set_ylim(-0.015, 0.98)
    ax.set_title("Future-latent retrieval")
    ax.legend(frameon=False, loc="upper right", handlelength=1.7)
    panel_label(ax, "a")
    despine(ax)

    ax = axes[1]
    for model, label, marker in [("uwm", "UWM-JEPA", "o"), ("lstm", "LSTM-JEPA", "s")]:
        y = np.array([retention["agg"][model]["delta"][str(k)][0] for k in ks_retention])
        s = np.array([retention["agg"][model]["delta"][str(k)][1] for k in ks_retention])
        line_with_band(ax, ks_retention, y, s, COL[model], label, marker)
    ax.axhline(0, color=COL["black"], linewidth=0.75)
    ax.axvline(5, color="#7A6F60", linewidth=0.8, linestyle=(0, (3, 2)), zorder=3)
    ax.text(5.25, -0.11, "trained horizon", color="#7A6F60", fontsize=6.5, ha="left", va="bottom")
    ax.axvspan(5.2, 21.0, color="#F7F4EE", alpha=0.8, linewidth=0, zorder=0)
    ax.axvspan(9.2, 21.0, color="#F1EEE6", alpha=0.95, linewidth=0, zorder=0)
    ax.text(19.7, 0.84, "weak teacher /\nextrapolation", color="#7A6F60", fontsize=6.5, ha="right", va="top")
    ax.set_xlabel(r"rollout horizon $k$")
    ax.set_ylabel(r"$\Delta R^2$ lost")
    ax.set_xticks(ks_retention)
    ax.set_ylim(-0.15, 0.92)
    ax.set_title("Teacher-to-blind loss")
    ax.legend(frameon=False, loc="upper left", handlelength=1.7)
    panel_label(ax, "b")
    despine(ax)
    save(fig, "fig_blind_rollout_evidence")


def make_action_binding() -> None:
    cf = load_json("multiseed_cf.json")
    action = load_json("actionctrl_multiseed.json")
    indicator = load_json("indicator.json")
    indicator_diag = load_json("indicator_lstm_cf_diagnostics.json")

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.58),
                             gridspec_kw={"width_ratios": [0.78, 1.03, 1.25], "wspace": 0.42})

    ax = axes[0]
    xs = np.array([0, 1])
    means = np.array([0.03, cf["aggregate"]["h1_over_h0"]["mean"]])
    stds = np.array([0.0, cf["aggregate"]["h1_over_h0"]["std"]])
    ax.errorbar(xs, means, yerr=stds, fmt="o", color=COL["black"], ecolor=COL["black"],
                elinewidth=0.95, capsize=3, markerfacecolor="white", markeredgewidth=1.1)
    ax.plot(xs, means, color=COL["mlp"], linewidth=0.9, linestyle=(0, (2, 2)))
    ax.axhline(1.0, color=COL["uwm"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(1.04, 1.02, "parity", color=COL["uwm"], fontsize=6.5, va="bottom")
    ax.set_xticks(xs, ["teacher\nforced", "counter-\nfactual"])
    ax.set_ylabel(r"$\|H_1\|/\|H_0\|$")
    ax.set_ylim(0, 1.27)
    ax.set_title("Action term")
    panel_label(ax, "a")
    despine(ax)

    ax = axes[1]
    ordered = [
        ("correct", "correct", COL["uwm"]),
        ("reversed", "reversed", COL["sky"]),
        ("base dyn.", "h1_zero", COL["mlp"]),
        ("wrong", "wrong", COL["plain"]),
        ("shuffled", "shuffled", COL["plain"]),
        ("negated", "negated", COL["lstm"]),
    ]
    labels = [x[0] for x in ordered]
    per_seed = {row["seed"]: row for row in action["rows"]}
    rel_rows = []
    for row in per_seed.values():
        base = row["correct"]
        rel_rows.append([(row[key] / base - 1.0) * 100.0 for _, key, _ in ordered])
    rel_rows = np.array(rel_rows)
    rel = rel_rows.mean(axis=0)
    rel_std = rel_rows.std(axis=0, ddof=1)
    colors = [x[2] for x in ordered]
    y = np.arange(len(ordered))
    ax.hlines(y, 0, rel, color=colors, linewidth=1.25)
    for j in range(rel_rows.shape[0]):
        ax.scatter(rel_rows[j], y + (j - 1) * 0.055, s=10, color=colors, alpha=0.38, linewidths=0, zorder=2)
    ax.errorbar(rel, y, xerr=rel_std, fmt="o", color=COL["black"], ecolor=COL["black"],
                markersize=3.5, elinewidth=0.85, capsize=2.0, zorder=4)
    ax.axvline(0, color=COL["black"], linewidth=0.7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("HS distance increase (%)")
    ax.set_title("Action perturbations")
    ax.set_xlim(-8, 80)
    ax.text(3.0, 2.35, r"$a=0$ and $H_1=0$", fontsize=6.3, color=COL["mlp"], ha="left", va="center")
    panel_label(ax, "b")
    despine(ax)

    ax = axes[2]
    conds = ["correct", "no_action", "shuffled", "wrong"]
    xpos = np.arange(len(conds))
    jitter = np.array([-0.042, 0.0, 0.042])
    for i, cond in enumerate(conds):
        for model, color, dx in [("uwm", COL["uwm"], -0.08), ("lstm", COL["mlp"], 0.08)]:
            vals = np.array([row[model][cond] for row in indicator["rows"]])
            ax.scatter(np.full_like(vals, xpos[i] + dx, dtype=float) + jitter, vals,
                       s=13, color=color, alpha=0.58, linewidths=0)
            mean = indicator["summary"][model][cond]["mean"]
            std = indicator["summary"][model][cond]["std"]
            ax.errorbar(xpos[i] + dx, mean, yerr=std, fmt="_", color=color, ecolor=color,
                        markersize=11, markeredgewidth=1.6, elinewidth=0.95, capsize=2.2, zorder=4)
    ax.axhline(0.5, color=COL["black"], linewidth=0.75, linestyle=(0, (3, 2)))
    ax.text(-0.25, 0.505, "chance", color=COL["black"], fontsize=6.5, ha="left", va="bottom")
    ax.annotate(
        "UWM action ordering",
        xy=(1.5, 0.704), xytext=(0.45, 0.792),
        arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": COL["uwm"], "shrinkA": 2, "shrinkB": 2},
        fontsize=6.4, color=COL["uwm"], ha="left",
    )
    maj = indicator_diag["majority_class"]
    count = indicator_diag["class_counts"][str(maj)]
    ax.text(3.35, 0.548, f"LSTM: all class {maj}", color=COL["mlp"], fontsize=6.4, ha="right", va="bottom")
    ax.text(3.35, 0.534, f"majority {count}/{indicator_diag['n_eval']}", color=COL["mlp"], fontsize=6.2, ha="right", va="bottom")
    ax.set_xticks(xpos, ["correct", "no action", "shuffled", "wrong"], rotation=22, ha="right")
    ax.set_ylabel("indicator accuracy")
    ax.set_ylim(0.47, 0.81)
    ax.set_title("Hidden-velocity indicator")
    ax.scatter([], [], color=COL["uwm"], s=18, label="UWM-CF")
    ax.scatter([], [], color=COL["mlp"], s=18, label="LSTM-CF")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.3)
    panel_label(ax, "c")
    despine(ax)
    save(fig, "fig_action_binding_evidence")


if __name__ == "__main__":
    set_style()
    make_invariants()
    make_collapse_phase()
    make_heldout_probe()
    make_blind_rollout()
    make_action_binding()
