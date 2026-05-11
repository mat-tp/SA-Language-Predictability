""" Plotting functions for the SA language n-gram study. """

import os
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# Colour palettes; One colour per n-gram order (n=1..5)
ORDER_COLORS = {
    1: "#AAAAAA",   
    2: "#0D9488",   
    3: "#1A3A5C",   
    4: "#F59E0B",   
    5: "#DC2626"
}

ORDER_LABELS = {
    1: "Unigram",
    2: "Bigram",
    3: "Trigram",
    4: "4-gram",
    5: "5-gram",
}

# One colour per tokenizer type
TOK_COLORS = {
    "char": "#0D9488",
    "word": "#1A3A5C",
}
# BPE entries are named "bpe_500", "bpe_2000", 
BPE_COLOR = "#F59E0B"

# One colour per language
LANG_COLORS = {
    "english":   "#0D9488",
    "afrikaans": "#1A3A5C",
    "sepedi":    "#F59E0B",
    "zulu":      "#DC2626",
}

def _ensure_results_dir():
    """Creates the results/ folder if needed and returns its path."""

    path = os.path.join(os.getcwd(), "results")
    os.makedirs(path, exist_ok=True)
    
    return path


def _tok_color(tok_name):
    """Returns a consistent colour for any tokenizer name."""

    if tok_name in TOK_COLORS:
        return TOK_COLORS[tok_name]
    return BPE_COLOR   # all BPE variants share the amber colour

# Plot 1 : How does entropy / perplexity change with n?
def plot_ngram_comparison(all_results, metric="entropy_token", save_path=None):
    """ Grouped bar chart: one group per language, one bar per n-gram order.

      * Does a higher-order model give lower entropy, and does this differ across languages? """
    
    # Collect the n-gram orders present in the data 
    orders = sorted({k for r in all_results for k in r if isinstance(k, int)})
    
    if not orders:
        print("  [plot_ngram_comparison] No numeric keys found — skipping.")
        return

    langs = [r["lang"].capitalize() for r in all_results]
    x     = list(range(len(langs)))
    n_orders = len(orders)
    bar_width = 0.7 / n_orders   # all bars for one language fit in 0.7 units

    if "entropy" in metric:
        ylabel = "Entropy  (bits / " + ("token" if "token" in metric else "character") + ")"
        title  = "Cross-Entropy by N-Gram Order"
    else:
        ylabel = "Perplexity"
        title  = "Perplexity by N-Gram Order"

    fig, ax = plt.subplots(figsize=(max(8, len(langs) * 2), 5))

    for idx, n in enumerate(orders):
        # Offset each order's bars so they sit side by side within each group
        offset = (idx - n_orders / 2 + 0.5) * bar_width
        vals   = [r.get(n, {}).get(metric, float("nan")) for r in all_results]
        x_pos  = [xi + offset for xi in x]

        bars = ax.bar(x_pos, vals, bar_width, label=ORDER_LABELS.get(n, f"{n}-gram"),color=ORDER_COLORS.get(n, "#888888"))
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(_ensure_results_dir(), f"ngram_comparison_{metric}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# Plot 2 : How do char vs word vs BPE compare?
def plot_tokenizer_comparison(all_results, metric="entropy_char", n_order=3, save_path=None):
    """ Grouped bar chart: one group per language, one bar per tokenizer.
    
    This is the central plot of the study — it shows whether character,
    word, or subword tokenization leads to lower (better) entropy when
    using the same n-gram order.

    Uses per-character entropy by default for fair cross-tokenizer comparison. """

    # Group results by language; within each language gather all tokenizers
    by_lang = {}
    for r in all_results:
        lang = r["lang"]
        by_lang.setdefault(lang, []).append(r)

    langs     = sorted(by_lang.keys())
    
    # Collect all tokenizer names in a stable order: char, word, then BPE variants
    tok_names = []
    for r in all_results:
        if r["tokenizer"] not in tok_names:
            tok_names.append(r["tokenizer"])

    n_toks = len(tok_names)
    x = list(range(len(langs)))
    bar_width = 0.7 / n_toks

    if "entropy" in metric:
        ylabel = "Entropy  (bits / " + ("token" if "token" in metric else "character") + ")"
        title  = (f"Cross-Entropy by Tokenizer  (n={n_order})\n"
                  "Per-character entropy enables fair cross-tokenizer comparison")
    else:
        ylabel = "Perplexity"
        title  = f"Perplexity by Tokenizer  (n={n_order})"

    fig, ax = plt.subplots(figsize=(max(9, len(langs) * 2.2), 5))

    for tidx, tok in enumerate(tok_names):
        offset = (tidx - n_toks / 2 + 0.5) * bar_width
        vals   = []
        for lang in langs:
            
            # Find the result dict for this (language, tokenizer) pair
            match = next(
                (r for r in by_lang[lang] if r["tokenizer"] == tok),
                None
            )
            val = match.get(n_order, {}).get(metric, float("nan")) if match else float("nan")
            vals.append(val)

        x_pos = [xi + offset for xi in x]
        bars  = ax.bar(x_pos, vals, bar_width,
                       label=tok,
                       color=_tok_color(tok))
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in langs], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path is None:
        save_path = os.path.join(
            _ensure_results_dir(),
            f"tokenizer_comparison_{metric}_n{n_order}.png"
        )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")