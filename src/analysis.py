"""
Corpus-level linguistic analysis for cross-language comparison.
Includes TTR, MATTR, vocabulary growth, Heaps' law fitting, and domain bias checks.
"""

import math
from collections import Counter


def type_token_ratio(tokens):
    """Unique types divided by total tokens. Sensitive to corpus size — use mattr() for comparisons."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def mattr(tokens, window_size=500):
    """
    Moving-Average Type-Token Ratio. Averages TTR over non-overlapping windows.
    Stable across different corpus sizes, safe for cross-language comparison.
    """
    if len(tokens) < window_size:
        return type_token_ratio(tokens)

    ttrs = []
    for start in range(0, len(tokens) - window_size + 1, window_size):
        window = tokens[start : start + window_size]
        ttrs.append(len(set(window)) / window_size)

    return sum(ttrs) / len(ttrs)


def vocabulary_growth(tokens, sample_points=None):
    """
    Records cumulative vocabulary size at given token positions.
    Defaults to 50 log-spaced points. Returns (token_counts, vocab_sizes).
    """
    n_total = len(tokens)
    if sample_points is None:
        import numpy as np
        sample_points = list(map(int, np.logspace(
            math.log10(max(100, n_total // 1000)),
            math.log10(n_total),
            num=50
        )))
        sample_points = sorted(set(sample_points))

    seen = set()
    ns, vs = [], []
    pt_iter = iter(sample_points)
    next_pt = next(pt_iter, None)

    for i, tok in enumerate(tokens, 1):
        seen.add(tok)
        if next_pt is not None and i >= next_pt:
            ns.append(i)
            vs.append(len(seen))
            next_pt = next(pt_iter, None)

    return ns, vs


def fit_heaps_law(ns, vs):
    """
    Fits Heaps' law: V(N) = K * N^beta via OLS in log-log space.
    Higher beta = faster vocabulary growth (morphological richness).
    Returns K, beta, and R².
    """
    if len(ns) < 2:
        return {"K": float("nan"), "beta": float("nan"), "r2": float("nan")}

    log_n = [math.log(n) for n in ns]
    log_v = [math.log(v) for v in vs]

    mean_n = sum(log_n) / len(log_n)
    mean_v = sum(log_v) / len(log_v)
    ss_nn = sum((x - mean_n) ** 2 for x in log_n)
    ss_nv = sum((x - mean_n) * (y - mean_v) for x, y in zip(log_n, log_v))

    if ss_nn == 0:
        return {"K": float("nan"), "beta": float("nan"), "r2": float("nan")}

    beta = ss_nv / ss_nn
    log_K = mean_v - beta * mean_n
    K = math.exp(log_K)

    ss_tot = sum((y - mean_v) ** 2 for y in log_v)
    preds = [log_K + beta * x for x in log_n]
    ss_res = sum((y - p) ** 2 for y, p in zip(log_v, preds))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {"K": K, "beta": beta, "r2": r2}


def corpus_statistics(lang, tokens_train, tokens_val, tokens_test,
                      tokenizer_name="char", cleaned_text=""):
    """Builds a stats dictionary for one language: token counts, TTR, Heaps' fit, word-level metrics."""
    all_tokens = tokens_train + tokens_val + tokens_test
    vocab_size = len(set(all_tokens))

    ns, vs = vocabulary_growth(tokens_train)
    heaps = fit_heaps_law(ns, vs)

    raw_ttr = type_token_ratio(tokens_train)
    ma_ttr = mattr(tokens_train, window_size=min(500, len(tokens_train) // 10))

    word_stats = {}
    if cleaned_text:
        words = cleaned_text.split()
        word_counts = Counter(words)
        word_stats = {
            "word_types": len(word_counts),
            "word_tokens": len(words),
            "word_ttr": len(word_counts) / len(words) if words else 0.0,
            "hapax_ratio": sum(1 for c in word_counts.values() if c == 1)
                           / len(word_counts) if word_counts else 0.0,
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0.0,
        }

    stats = {
        "lang": lang,
        "tokenizer": tokenizer_name,
        "total_tokens": len(all_tokens),
        "train_tokens": len(tokens_train),
        "val_tokens": len(tokens_val),
        "test_tokens": len(tokens_test),
        "vocab_size": vocab_size,
        "raw_ttr": raw_ttr,
        "mattr": ma_ttr,
        "heaps_K": heaps["K"],
        "heaps_beta": heaps["beta"],
        "heaps_r2": heaps["r2"],
        "heaps_ns": ns,
        "heaps_vs": vs,
        **word_stats,
    }
    return stats


def compare_corpora(all_stats):
    """Prints a formatted comparison table of corpus statistics across languages."""
    header = (
        f"{'Language':<12} {'Tokenizer':<12} {'Tokens':>10} "
        f"{'Vocab':>7} {'MATTR':>7} {'beta':>6} {'R2':>5} "
        f"{'WordTTR':>8} {'HapaxR':>7} {'AvgWLen':>8}"
    )
    print("\n" + "=" * len(header))
    print("  CORPUS STATISTICS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for s in all_stats:
        wt = s.get("word_ttr", float("nan"))
        hx = s.get("hapax_ratio", float("nan"))
        awl = s.get("avg_word_length", float("nan"))
        print(
            f"  {s['lang'].capitalize():<10} {s['tokenizer']:<12} "
            f"{s['train_tokens']:>10,} "
            f"{s['vocab_size']:>7} {s['mattr']:>7.4f} "
            f"{s['heaps_beta']:>6.3f} {s['heaps_r2']:>5.3f} "
            f"{wt:>8.4f} {hx:>7.4f} {awl:>8.2f}"
        )
    print("=" * len(header) + "\n")


def domain_bias_analysis(lang, tokens, top_n=30):
    """
    Analyzes token-frequency distribution to surface potential corpus bias.
    Narrow-domain corpora show low entropy, high top-K coverage, steep Zipf slope.
    """
    counts = Counter(tokens)
    total = len(tokens)
    most_common = counts.most_common(top_n)

    top_freqs = [c / total for _, c in most_common]
    top_coverage = sum(top_freqs)

    freq_entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values() if c > 0
    )

    sorted_freqs = [c / total for _, c in counts.most_common()]
    ranks = list(range(1, len(sorted_freqs) + 1))

    log_r = [math.log(r) for r in ranks if r <= len(sorted_freqs)]
    log_f = [math.log(f) for f in sorted_freqs if f > 0]
    min_len = min(len(log_r), len(log_f))
    log_r, log_f = log_r[:min_len], log_f[:min_len]

    zipf_alpha = float("nan")
    if len(log_r) > 2:
        mean_r = sum(log_r) / len(log_r)
        mean_f = sum(log_f) / len(log_f)
        ss_rr = sum((x - mean_r) ** 2 for x in log_r)
        ss_rf = sum((x - mean_r) * (y - mean_f) for x, y in zip(log_r, log_f))
        if ss_rr > 0:
            zipf_alpha = -ss_rf / ss_rr

    print(f"\n  [{lang.upper()} DOMAIN BIAS]")
    print(f"  Top-{top_n} token coverage : {top_coverage:.2%}")
    print(f"  Frequency entropy        : {freq_entropy:.4f} bits")
    print(f"  Zipf exponent (alpha)    : {zipf_alpha:.3f}" if not math.isnan(zipf_alpha)
          else "  Zipf exponent            : n/a")

    return {
        "lang": lang,
        "top_n_types": [t for t, _ in most_common],
        "top_n_freqs": top_freqs,
        "top_n_coverage": top_coverage,
        "freq_entropy": freq_entropy,
        "zipf_alpha": zipf_alpha,
    }