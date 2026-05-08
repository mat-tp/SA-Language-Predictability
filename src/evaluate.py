"""
Evaluating n-gram language models using cross-entropy, perplexity,
training-size experiments, alpha tuning, and bootstrap confidence intervals.
"""

import math
import random

from language_model import (
    build_ngram_model,
    build_unigram_model,
    build_kneser_ney_model,
    ngram_prob,
    unigram_prob,
    kneser_ney_prob,
)

def _calculate_total_log_prob(tokens, model, model_n=None, alpha=1.0):
    """
    Internal helper to compute total log probability for a token sequence.
    Extracts duplicate logic used by both token and char cross-entropy.
    """
    mtype = model.get("type", "ngram")
    n = model_n if model_n is not None else model["n"]

    if mtype == "unigram":
        return sum(math.log2(max(unigram_prob(w, model, alpha), 1e-300)) for w in tokens)

    n_contexts = len(tokens) - n + 1
    if n_contexts <= 0:
        raise ValueError("Test sequence too short for this n-gram order.")

    total_lp = 0.0
    for i in range(n_contexts):
        context = tuple(tokens[i : i + n - 1])
        word = tokens[i + n - 1]

        if mtype == "kneser_ney":
            prob = kneser_ney_prob(context, word, model)
        else:
            prob = ngram_prob(context, word, model, alpha=alpha)

        total_lp += math.log2(max(prob, 1e-300))

    return total_lp

def cross_entropy_tokens(tokens, model, model_n=None, alpha=1.0):
    """Per-token cross-entropy in bits. Lower is better."""
    total_lp = _calculate_total_log_prob(tokens, model, model_n, alpha)
    n = model_n if model_n is not None else model["n"]
    sequence_length = len(tokens) if model.get("type") == "unigram" else len(tokens) - n + 1
    return -total_lp / sequence_length

def cross_entropy_chars(tokens, model, tokenizer, model_n=None, alpha=1.0):
    """Per-character cross-entropy in bits. Fair comparison across tokenizers."""
    total_lp = _calculate_total_log_prob(tokens, model, model_n, alpha)
    full_text = tokenizer.decode(tokens)
    num_chars = len(full_text)
    
    if num_chars == 0:
        raise ValueError("Decoded test text has zero characters.")

    return -total_lp / num_chars

def perplexity(entropy_bits):
    """Converts entropy (bits) to perplexity."""
    return 2 ** entropy_bits

def bootstrap_confidence_interval(tokens, model, alpha=1.0, metric="token",
                                  tokenizer=None, n_bootstrap=500,
                                  ci_level=0.95, block_size=512, seed=42):
    """
    Block bootstrap confidence interval for cross-entropy.
    Uses contiguous chunks to preserve n-gram dependencies.
    """
    rng = random.Random(seed)
    blocks = [tokens[i : i + block_size] for i in range(0, len(tokens) - block_size + 1, block_size)]
    if not blocks:
        blocks = [tokens]

    def compute_entropy(seq):
        if metric == "char":
            return cross_entropy_chars(seq, model, tokenizer, alpha=alpha)
        return cross_entropy_tokens(seq, model, alpha=alpha)

    point_estimate = compute_entropy(tokens)
    bootstrap_vals = []
    n_blocks_needed = math.ceil(len(tokens) / block_size)

    for _ in range(n_bootstrap):
        sampled_blocks = rng.choices(blocks, k=n_blocks_needed)
        resampled = [tok for blk in sampled_blocks for tok in blk][:len(tokens)]
        
        if len(resampled) < model["n"]:
            continue
            
        try:
            bootstrap_vals.append(compute_entropy(resampled))
        except ValueError:
            continue

    bootstrap_vals.sort()
    tail = (1.0 - ci_level) / 2.0
    lo_idx = int(tail * len(bootstrap_vals))
    hi_idx = int((1.0 - tail) * len(bootstrap_vals))
    
    lower_bound = bootstrap_vals[lo_idx] if lo_idx < len(bootstrap_vals) else point_estimate
    upper_bound = bootstrap_vals[hi_idx - 1] if hi_idx <= len(bootstrap_vals) else point_estimate

    return (point_estimate, lower_bound, upper_bound)

def run_training_size_experiment(train_all, test_tokens, vocab_size,
                                 tokenizer, n=3, fractions=None,
                                 alpha=1.0, seed=42):
    """
    Evaluates model performance across increasing fractions of training data.
    Returns per-token and per-character entropy/perplexity for each fraction.
    """
    if fractions is None:
        fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {
        "frac": [],
        "entropy_token": [],
        "entropy_char": [],
        "perplexity_token": [],
        "perplexity_char": [],
    }

    total_len = len(train_all)

    for frac in fractions:
        sample_size = int(total_len * frac)
        if sample_size < n:
            print(f"  Skipping {frac:.0%} — too few tokens for n={n}")
            continue

        train_subset = train_all[:sample_size]
        model = build_ngram_model(train_subset, n=n, vocab_size=vocab_size)

        ent_tok = cross_entropy_tokens(test_tokens, model, alpha=alpha)
        ent_char = cross_entropy_chars(test_tokens, model, tokenizer, alpha=alpha)
        ppl_tok = perplexity(ent_tok)
        ppl_char = perplexity(ent_char)

        results["frac"].append(frac)
        results["entropy_token"].append(ent_tok)
        results["entropy_char"].append(ent_char)
        results["perplexity_token"].append(ppl_tok)
        results["perplexity_char"].append(ppl_char)

        print(f"  {frac:.0%}: {sample_size:,} tokens | "
              f"H_tok={ent_tok:.4f} H_char={ent_char:.4f} "
              f"PPL_tok={ppl_tok:.2f} PPL_char={ppl_char:.2f}")

    return results


def tune_alpha(train_tokens, val_tokens, vocab_size, n=3, alphas=None):
    """
    Selects best Add-alpha smoothing coefficient using validation set.
    Returns the alpha that minimizes per-token cross-entropy.
    """
    if alphas is None:
        alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    model = build_ngram_model(train_tokens, n=n, vocab_size=vocab_size)
    best_alpha = 1.0
    best_ent = float("inf")

    for alpha in alphas:
        ent = cross_entropy_tokens(val_tokens, model, alpha=alpha)
        if ent < best_ent:
            best_ent = ent
            best_alpha = alpha

    print(f"    Alpha tuning (n={n}): best alpha={best_alpha:.4f} "
          f"H_val={best_ent:.4f}")
    return best_alpha


def evaluate_all_models(train_enc, val_enc, test_enc, vocab_size,
                        tokenizer, run_bootstrap=False, n_bootstrap=300):
    """
    Trains and evaluates unigram, bigram, trigram (both Laplace and Kneser-Ney).
    Returns nested results keyed by model name with entropy and perplexity.
    """
    alpha2 = tune_alpha(train_enc, val_enc, vocab_size, n=2)
    alpha3 = tune_alpha(train_enc, val_enc, vocab_size, n=3)

    models = {
        "unigram": (build_unigram_model(train_enc, vocab_size), 1.0),
        "bigram_laplace": (build_ngram_model(train_enc, 2, vocab_size), alpha2),
        "trigram_laplace": (build_ngram_model(train_enc, 3, vocab_size), alpha3),
        "bigram_kn": (build_kneser_ney_model(train_enc, 2, vocab_size), None),
        "trigram_kn": (build_kneser_ney_model(train_enc, 3, vocab_size), None),
    }

    results = {}
    for name, (model, alpha) in models.items():
        kw = {} if alpha is None else {"alpha": alpha}
        ht = cross_entropy_tokens(test_enc, model, **kw)
        hc = cross_entropy_chars(test_enc, model, tokenizer, **kw)

        entry = {
            "entropy_token": ht,
            "entropy_char": hc,
            "perplexity_token": perplexity(ht),
            "perplexity_char": perplexity(hc),
        }

        if run_bootstrap:
            ci_t = bootstrap_confidence_interval(
                test_enc, model, metric="token",
                n_bootstrap=n_bootstrap, **kw)
            ci_c = bootstrap_confidence_interval(
                test_enc, model, tokenizer=tokenizer, metric="char",
                n_bootstrap=n_bootstrap, **kw)
            entry["ci_token"] = ci_t
            entry["ci_char"] = ci_c

        results[name] = entry
        print(f"  {name:<20} H_tok={ht:.4f} H_char={hc:.4f} "
              f"PPL_tok={entry['perplexity_token']:.2f} "
              f"PPL_char={entry['perplexity_char']:.2f}")

    return results