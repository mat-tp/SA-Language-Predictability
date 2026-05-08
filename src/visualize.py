"""
Plotting functions for the SA language n-gram study.
Covers language comparison, training curves, Heaps' law,
model comparison, domain bias, and smoothing sensitivity.
"""

import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Consistent color palette across all plots
PALETTE = {
    "unigram": "#AAAAAA",
    "bigram_laplace": "#0D9488",
    "trigram_laplace": "#1A3A5C",
    "bigram_kn": "#F59E0B",
    "trigram_kn": "#DC2626",
    "heaps_data": "#1A3A5C",
    "heaps_fit": "#DC2626",
}

LANG_COLORS = {
    "english": "#0D9488",
    "afrikaans": "#1A3A5C",
    "sepedi": "#F59E0B",
    "zulu": "#DC2626",
}


def ensure_results_dir():
    """Creates results/ directory if it doesn't exist, returns its path."""
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def plot_language_comparison(all_results, metric="entropy",
                             token_norm="token", save_path=None):
    """
    Bar chart comparing bigram vs trigram performance across languages.
    Optional error bars from bootstrap confidence intervals.
    """
    if metric == "entropy":
        vals2 = [r.get(f"entropy2_{token_norm}", r["entropy2"]) for r in all_results]
        vals3 = [r.get(f"entropy3_{token_norm}", r["entropy3"]) for r in all_results]
        ylabel = f"Entropy (bits/{token_norm})"
        title = f"Cross-Entropy — per {token_norm}"
    elif metric == "perplexity":
        vals2 = [r.get(f"perplexity2_{token_norm}", r["perplexity2"]) for r in all_results]
        vals3 = [r.get(f"perplexity3_{token_norm}", r["perplexity3"]) for r in all_results]
        ylabel = "Perplexity"
        title = f"Perplexity — per {token_norm}"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    langs = [r["lang"].capitalize() for r in all_results]
    x = list(range(len(langs)))
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(langs) * 1.8), 5))

    # Try to extract bootstrap error bars
    yerr2 = yerr3 = None
    ci_key = f"ci_{token_norm}"
    ci2 = [r.get(ci_key + "_2") for r in all_results]
    ci3 = [r.get(ci_key + "_3") for r in all_results]
    
    if all(c is not None for c in ci2):
        # Ensure non-negative error values by taking max with 0
        yerr2 = [[max(0, v - c[1]) for v, c in zip(vals2, ci2)],
                 [max(0, c[2] - v) for v, c in zip(vals2, ci2)]]
    if all(c is not None for c in ci3):
        # Ensure non-negative error values by taking max with 0
        yerr3 = [[max(0, v - c[1]) for v, c in zip(vals3, ci3)],
                 [max(0, c[2] - v) for v, c in zip(vals3, ci3)]]

    b1 = ax.bar([i - w/2 for i in x], vals2, w,
                label="Bigram", color=PALETTE["bigram_laplace"],
                yerr=yerr2, capsize=4, error_kw={"linewidth": 1.2})
    b2 = ax.bar([i + w/2 for i in x], vals3, w,
                label="Trigram", color=PALETTE["trigram_laplace"],
                yerr=yerr3, capsize=4, error_kw={"linewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.bar_label(b1, fmt="%.2f", padding=4, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=4, fontsize=8)
    ax.set_ylim(0, max(vals2 + vals3) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    results_dir = ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(results_dir, f"comparison_{metric}_{token_norm}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_training_size_curves(experiment_results, title="Training Size Experiment",
                              save_path=None):
    """Plots entropy vs fraction of training data used, both per-token and per-character."""
    fracs = experiment_results["frac"]
    ent_token = experiment_results["entropy_token"]
    ent_char = experiment_results["entropy_char"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontweight="bold", fontsize=13)

    ax1.plot(fracs, ent_token, "o-", color=PALETTE["bigram_laplace"],
             linewidth=2, markersize=6)
    ax1.set_xlabel("Fraction of training data", fontsize=11)
    ax1.set_ylabel("Entropy (bits/token)", fontsize=11)
    ax1.set_title("Per-Token Entropy", fontsize=11)
    ax1.grid(True, alpha=0.3)

    ax2.plot(fracs, ent_char, "s-", color=PALETTE["trigram_laplace"],
             linewidth=2, markersize=6)
    ax2.set_xlabel("Fraction of training data", fontsize=11)
    ax2.set_ylabel("Entropy (bits/character)", fontsize=11)
    ax2.set_title("Per-Character Entropy", fontsize=11)
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    results_dir = ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(results_dir, "training_size.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_heaps_law(all_stats, save_path=None):
    """
    Plots vocabulary growth curves (log-log) with fitted Heaps' law lines.
    One subplot per language.
    """
    n_langs = len(all_stats)
    ncols = min(2, n_langs)
    nrows = math.ceil(n_langs / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    fig.suptitle("Vocabulary Growth (Heaps' Law)", fontsize=14, fontweight="bold")

    for idx, stats in enumerate(all_stats):
        ax = axes[idx // ncols][idx % ncols]
        ns = stats["heaps_ns"]
        vs = stats["heaps_vs"]
        beta = stats["heaps_beta"]
        K = stats["heaps_K"]
        r2 = stats["heaps_r2"]
        lang = stats["lang"].capitalize()
        col = LANG_COLORS.get(stats["lang"], "#333333")

        # Observed data
        ax.plot(ns, vs, ".", color=col, alpha=0.7, markersize=4, label="Observed")

        # Fitted curve
        if not (math.isnan(K) or math.isnan(beta)):
            ns_fit = [ns[0]] + list(range(ns[0], ns[-1],
                                          max(1, (ns[-1] - ns[0]) // 200)))
            vs_fit = [K * (n ** beta) for n in ns_fit]
            ax.plot(ns_fit, vs_fit, "-", color=PALETTE["heaps_fit"],
                    linewidth=2,
                    label=f"Fit: V={K:.1f}·N^{beta:.3f}\n(R²={r2:.3f})")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Tokens (N)", fontsize=10)
        ax.set_ylabel("Vocabulary size (V)", fontsize=10)
        ax.set_title(f"{lang}  β={beta:.3f}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for idx in range(n_langs, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.tight_layout()
    results_dir = ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(results_dir, "heaps_law.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(results_by_lang, metric="entropy_token", save_path=None):
    """
    Grouped bar chart comparing all model variants across languages.
    Shows unigram, bigram/trigram with Laplace and Kneser-Ney smoothing.
    """
    model_names = ["unigram", "bigram_laplace", "trigram_laplace",
                   "bigram_kn", "trigram_kn"]
    model_labels = ["Unigram", "Bigram\n(Laplace)", "Trigram\n(Laplace)",
                    "Bigram\n(KN)", "Trigram\n(KN)"]

    langs = sorted(results_by_lang.keys())
    n_models = len(model_names)
    n_langs = len(langs)
    w = 0.12
    offsets = [(i - n_models / 2 + 0.5) * w for i in range(n_models)]

    ylabel = metric.replace("_", " ").replace("entropy", "Entropy (bits)") \
                   .replace("perplexity", "Perplexity")
    title = f"{ylabel} — All Models"

    fig, ax = plt.subplots(figsize=(max(10, n_langs * 2.5), 5.5))

    for mi, (mname, mlabel) in enumerate(zip(model_names, model_labels)):
        vals = []
        cis_lo, cis_hi = [], []
        for lang in langs:
            lang_res = results_by_lang.get(lang, {})
            entry = lang_res.get(mname, {})
            vals.append(entry.get(metric, float("nan")))
            ci_key = "ci_" + ("token" if "token" in metric else "char")
            ci = entry.get(ci_key)
            if ci:
                cis_lo.append(max(0, vals[-1] - ci[1]))
                cis_hi.append(max(0, ci[2] - vals[-1]))
            else:
                cis_lo.append(0)
                cis_hi.append(0)

        x_pos = [i + offsets[mi] for i in range(n_langs)]
        any_ci = any(lo > 0 or hi > 0 for lo, hi in zip(cis_lo, cis_hi))
        yerr = [cis_lo, cis_hi] if any_ci else None
        ax.bar(x_pos, vals, w, label=mlabel,
               color=PALETTE.get(mname, "#888888"),
               yerr=yerr, capsize=3, error_kw={"linewidth": 1})

    ax.set_xticks(range(n_langs))
    ax.set_xticklabels([l.capitalize() for l in langs], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    results_dir = ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(results_dir, f"model_comparison_{metric}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_domain_bias(bias_results, save_path=None):
    """
    Plots top-N token coverage, Zipf alpha, and frequency entropy across languages.
    Reveals domain homogeneity — narrow corpora show high coverage and steep Zipf slope.
    """
    langs = [r["lang"].capitalize() for r in bias_results]
    coverage = [r["top_n_coverage"] for r in bias_results]
    zipf = [r["zipf_alpha"] if not math.isnan(r["zipf_alpha"]) else 0
            for r in bias_results]
    freq_ent = [r["freq_entropy"] for r in bias_results]

    x = list(range(len(langs)))
    cols = [LANG_COLORS.get(r["lang"], "#888888") for r in bias_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Domain Bias Indicators", fontsize=13, fontweight="bold")

    for ax, vals, title, ylabel in zip(
        axes,
        [coverage, zipf, freq_ent],
        ["Top-30 Token Coverage", "Zipf Exponent (α)", "Frequency Entropy (bits)"],
        ["Fraction of corpus", "α", "bits"],
    ):
        bars = ax.bar(x, vals, color=cols, width=0.5, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(langs, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    results_dir = ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(results_dir, "domain_bias.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_smoothing_curves(tuning_data, save_path=None):
    """
    Plots validation entropy vs alpha for each language and n-gram order.
    Shows sensitivity to the smoothing parameter.
    """
    langs = sorted(tuning_data.keys())
    n_vals = sorted({n for lang_d in tuning_data.values() for n in lang_d})

    fig, axes = plt.subplots(1, len(n_vals),
                             figsize=(6 * len(n_vals), 4.5),
                             squeeze=False)

    for col, n in enumerate(n_vals):
        ax = axes[0][col]
        for lang in langs:
            entry = tuning_data[lang].get(n)
            if entry is None:
                continue
            col_c = LANG_COLORS.get(lang, "#888888")
            ax.semilogx(entry["alphas"], entry["entropies"],
                        "o-", color=col_c, linewidth=1.8, markersize=5,
                        label=lang.capitalize())
        ax.set_xlabel("α (log scale)", fontsize=10)
        ax.set_ylabel("Validation entropy (bits/token)", fontsize=10)
        ax.set_title(f"n={n} Smoothing Sensitivity", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    results_dir = ensure_results_dir()
    if save_path is None:
        save_path = os.path.join(results_dir, "smoothing_curves.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")