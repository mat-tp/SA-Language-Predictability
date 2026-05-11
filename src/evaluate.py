""" Evaluates n-gram models using cross-entropy and perplexity.
    Two normalisation modes are supported so results are comparable:
        1. Per-token, divides total log-probability by the number of predicted tokens. 
           This is the standard measure but depends on tokenizer granularity.
        2. Per-character, divides by the number of characters in the decoded text. This is 
           a fair comparison across tokenizers of different granularities (e.g. char vs word vs BPE). 
"""

import math

from language_model import (
    build_unigram_model,
    build_ngram_model,
    unigram_prob,
    ngram_prob,
    compute_perplexity,
)


def cross_entropy_per_token(tokens, model, alpha=1.0):
    """ Per-token cross-entropy H in bits.

        H = -(1/N) Σ log₂ P(wᵢ | context_i). """
    
    mtype = model["type"]
    n = model["n"]

    if mtype == "unigram":
        log_sum = sum(
            math.log2(max(unigram_prob(w, model, alpha), 1e-300))
            for w in tokens
        )
        return -log_sum / len(tokens)

    num_predictions = len(tokens) - n + 1
    if num_predictions <= 0:
        raise ValueError(
            f"Token sequence has {len(tokens)} tokens — too short for n={n}."
        )

    log_sum = 0.0
    for i in range(num_predictions):
        context = tuple(tokens[i : i + n - 1])
        word    = tokens[i + n - 1]
        prob    = ngram_prob(context, word, model, alpha)
        log_sum += math.log2(max(prob, 1e-300))

    return -log_sum / num_predictions


def cross_entropy_per_char(tokens, model, tokenizer, alpha=1.0):
    """ Per-character cross-entropy H in bits.

    Same log-probability sum as cross_entropy_per_token, but divided by
    the number of *characters* in the decoded text rather than the number
    of tokens. """
    mtype = model["type"]
    n = model["n"]

    # Compute total log-probability
    if mtype == "unigram":
        total_log_prob = sum(
            math.log2(max(unigram_prob(w, model, alpha), 1e-300))
            for w in tokens
        )
        relevant_tokens = tokens
    else:
        num_predictions = len(tokens) - n + 1
        if num_predictions <= 0:
            raise ValueError(
                f"Token sequence has {len(tokens)} tokens — too short for n={n}."
            )
        total_log_prob = 0.0
        for i in range(num_predictions):
            context = tuple(tokens[i : i + n - 1])
            word    = tokens[i + n - 1]
            prob    = ngram_prob(context, word, model, alpha)
            total_log_prob += math.log2(max(prob, 1e-300))
        # The predicted tokens start at position n-1
        relevant_tokens = tokens[n - 1:]

    # Count characters in the decoded text that was actually predicted
    decoded_text = tokenizer.decode(relevant_tokens)
    num_chars    = len(decoded_text)

    if num_chars == 0:
        raise ValueError("Decoded text has zero characters — cannot compute per-char entropy.")

    return -total_log_prob / num_chars


def perplexity(entropy_bits):
    """ Converts cross-entropy (bits) to perplexity: PP = 2^H. """
    return 2 ** entropy_bits


# Alpha (smoothing) tuning
def tune_alpha(train_tokens, val_tokens, vocab_size, n,alphas=None):
    """ Finds the Add-alpha smoothing value that minimises validation entropy. """

    if alphas is None:
        alphas = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

    if n == 1:
        model = build_unigram_model(train_tokens, vocab_size)
    else:
        model = build_ngram_model(train_tokens, n, vocab_size)

    best_alpha   = alphas[0]
    best_entropy = float("inf")

    for alpha in alphas:
        entropy = cross_entropy_per_token(val_tokens, model, alpha)
        if entropy < best_entropy:
            best_entropy = entropy
            best_alpha   = alpha

    print(f"    n={n}  best alpha={best_alpha:.4f}  "
          f"val entropy={best_entropy:.4f} bits/token")
    return best_alpha

# evaluation across all n-gram orders
def evaluate_all_orders(train_tokens, val_tokens, test_tokens, vocab_size, tokenizer, max_n=5, alphas=None):
    """ Trains and evaluates models for every n from 1 to max_n.

    For each n:
      1. Tune alpha on the validation set.
      2. Evaluate per-token and per-character entropy and perplexity
         on the held-out test set.

    Returns dict  { n (int) = { "alpha", "entropy_token", "entropy_char", "perplexity_token", "perplexity_char" } }
    """
    results = {}

    for n in range(1, max_n + 1):
        alpha = tune_alpha(train_tokens, val_tokens, vocab_size, n, alphas)

        if n == 1:
            model = build_unigram_model(train_tokens, vocab_size)
        else:
            model = build_ngram_model(train_tokens, n, vocab_size)

        h_tok  = cross_entropy_per_token(test_tokens, model, alpha)
        h_char = cross_entropy_per_char(test_tokens, model, tokenizer, alpha)
        pp_tok  = perplexity(h_tok)
        pp_char = perplexity(h_char)

        results[n] = {
            "alpha": alpha,
            "entropy_token": h_tok,
            "entropy_char": h_char,
            "perplexity_token": pp_tok,
            "perplexity_char": pp_char,
        }

        order_name = {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "4-gram",  5: "5-gram"}.get(n, f"{n}-gram")
        
        print(f"  {order_name:<10}  "
              f"H_tok={h_tok:.4f}  H_char={h_char:.4f}  "
              f"PPL_tok={pp_tok:.2f}  PPL_char={pp_char:.2f}")

    return results