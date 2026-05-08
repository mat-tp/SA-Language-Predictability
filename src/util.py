"""
Utility functions for data splitting and basic evaluation.
"""

import matplotlib.pyplot as plt

from language_model import build_ngram_model, compute_entropy, compute_perplexity


def split_data(data, train_ratio=0.8, val_ratio=0.1):
    """Splits data into train, validation, and test sets. Default: 80/10/10."""
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    train = data[:train_size]
    val = data[train_size:train_size + val_size]
    test = data[train_size + val_size:]
    return train, val, test


def plot_results(all_results):
    """Plots entropy and perplexity comparisons across languages (bigram vs trigram)."""
    langs = [r["lang"].capitalize() for r in all_results]
    H2 = [r["entropy2"] for r in all_results]
    H3 = [r["entropy3"] for r in all_results]
    P2 = [r["perplexity2"] for r in all_results]
    P3 = [r["perplexity3"] for r in all_results]

    x = range(len(langs))
    width = 0.35
    colors = {"bigram": "#0D9488", "trigram": "#1A3A5C"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("N-Gram Language Model SA-Language Comparison",
                 fontsize=14, fontweight="bold", y=1.01)

    # Entropy
    ax = axes[0]
    b1 = ax.bar([i - width/2 for i in x], H2, width,
                label="Bigram", color=colors["bigram"])
    b2 = ax.bar([i + width/2 for i in x], H3, width,
                label="Trigram", color=colors["trigram"])
    ax.set_title("Entropy (bits/token)")
    ax.set_ylabel("Bits per token")
    ax.set_xticks(list(x))
    ax.set_xticklabels(langs)
    ax.legend()
    ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)
    ax.set_ylim(0, max(H2 + H3) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    # Perplexity
    ax = axes[1]
    b3 = ax.bar([i - width/2 for i in x], P2, width,
                label="Bigram", color=colors["bigram"])
    b4 = ax.bar([i + width/2 for i in x], P3, width,
                label="Trigram", color=colors["trigram"])
    ax.set_title("Perplexity")
    ax.set_ylabel("Perplexity")
    ax.set_xticks(list(x))
    ax.set_xticklabels(langs)
    ax.legend()
    ax.bar_label(b3, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(b4, fmt="%.1f", padding=3, fontsize=9)
    ax.set_ylim(0, max(P2 + P3) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.show()


def print_stats(lang, lang_data):
    """Prints corpus statistics for one language."""
    stats = lang_data["stats"]
    tokenizer = lang_data["tokenizer"]
    train = lang_data["train"]
    val = lang_data["val"]
    test = lang_data["test"]

    print(f"\n")
    print(f"  {lang.upper()}")
    print(f"{'='*60}")
    print(f"  Raw length    : {stats['raw_length']:,} chars")
    print(f"  Clean length  : {stats['clean_length']:,} chars")
    print(f"  Vocab size    : {tokenizer['vocab_size']} unique chars")
    print(f"  Vocabulary    : {''.join(stats['chars'])}")
    print(f"  Train tokens  : {len(train):,}")
    print(f"  Val   tokens  : {len(val):,}")
    print(f"  Test  tokens  : {len(test):,}")


def evaluate(lang, lang_data):
    """Evaluates bigram and trigram models on the test set."""
    train = lang_data["train"]
    test = lang_data["test"]
    vocab_size = lang_data["tokenizer"]["vocab_size"]

    model2 = build_ngram_model(train, n=2, vocab_size=vocab_size)
    H2 = compute_entropy(test, model2)
    P2 = compute_perplexity(H2)

    model3 = build_ngram_model(train, n=3, vocab_size=vocab_size)
    H3 = compute_entropy(test, model3)
    P3 = compute_perplexity(H3)

    print(f"\n  [TEST RESULTS]  {lang.upper()}")
    print(f"  Bigram  → Entropy: {H2:.4f} bits  |  Perplexity: {P2:.2f}")
    print(f"  Trigram → Entropy: {H3:.4f} bits  |  Perplexity: {P3:.2f}")

    return {
        "lang": lang,
        "model2": model2,
        "model3": model3,
        "entropy2": H2,
        "entropy3": H3,
        "perplexity2": P2,
        "perplexity3": P3,
    }