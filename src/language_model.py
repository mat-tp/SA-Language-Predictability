"""
N-gram language model with Add-alpha (Laplace) and Kneser-Ney smoothing.
Supports unigram, bigram, trigram construction, probability estimation,
entropy calculation, and text generation.
"""

import math
import random
from collections import defaultdict, Counter


def build_ngram_model(tokens, n, vocab_size):
    """Builds an n-gram model (n >= 2) from token ids. Stores counts and context totals."""
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}. Use build_unigram_model for n=1.")
    if len(tokens) < n:
        raise ValueError(f"Training sequence too short for n={n}.")

    counts = defaultdict(Counter)
    context_counts = Counter()

    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i : i + n - 1])
        word = tokens[i + n - 1]
        counts[context][word] += 1
        context_counts[context] += 1

    return {
        "counts": counts,
        "context_counts": context_counts,
        "n": n,
        "vocab_size": vocab_size,
        "type": "ngram",
    }


def build_unigram_model(tokens, vocab_size):
    """Builds a unigram baseline model — context-free word probabilities."""
    counts = Counter(tokens)
    return {
        "counts": counts,
        "total": len(tokens),
        "n": 1,
        "vocab_size": vocab_size,
        "type": "unigram",
    }


def build_kneser_ney_model(tokens, n, vocab_size, discount=0.75):
    """
    Builds a Kneser-Ney smoothed model (n=2 or 3).
    Uses continuation counts for backing off — handles context-sparse words better.
    Discount defaults to 0.75 (Chen & Goodman, 1999).
    """
    if n not in (2, 3):
        raise ValueError("Kneser-Ney implemented for n=2 and n=3 only.")

    bigram_counts = defaultdict(Counter)
    trigram_counts = defaultdict(Counter)
    context1_totals = Counter()
    context2_totals = Counter()

    for i in range(len(tokens) - 1):
        ctx1 = (tokens[i],)
        w = tokens[i + 1]
        bigram_counts[ctx1][w] += 1
        context1_totals[ctx1] += 1

    if n == 3:
        for i in range(len(tokens) - 2):
            ctx2 = (tokens[i], tokens[i + 1])
            w = tokens[i + 2]
            trigram_counts[ctx2][w] += 1
            context2_totals[ctx2] += 1

    left_contexts = Counter()
    for ctx1, word_ctr in bigram_counts.items():
        for w in word_ctr:
            left_contexts[w] += 1

    total_bigram_types = sum(left_contexts.values())

    return {
        "n": n,
        "vocab_size": vocab_size,
        "discount": discount,
        "bigram_counts": bigram_counts,
        "trigram_counts": trigram_counts,
        "context1_totals": context1_totals,
        "context2_totals": context2_totals,
        "left_contexts": left_contexts,
        "total_bigram_types": total_bigram_types,
        "type": "kneser_ney",
    }


def unigram_prob(word, model, alpha=1.0):
    """Add-alpha smoothed unigram probability."""
    return (model["counts"].get(word, 0) + alpha) / (
        model["total"] + alpha * model["vocab_size"]
    )


def ngram_prob(context, word, model, alpha=1.0):
    """P(word | context) with Add-alpha smoothing."""
    counts = model["counts"]
    context_counts = model["context_counts"]
    vocab_size = model["vocab_size"]

    count_cw = counts.get(context, Counter()).get(word, 0)
    total_c = context_counts.get(context, 0)
    return (count_cw + alpha) / (total_c + alpha * vocab_size)


def kneser_ney_prob(context, word, model):
    """
    Interpolated Kneser-Ney probability.
    For unseen contexts, backs off gracefully to continuation unigram.
    """
    D = model["discount"]
    vocab_size = model["vocab_size"]
    n = model["n"]

    left_ctx = model["left_contexts"]
    total_bt = model["total_bigram_types"] or 1
    p_kn_uni = (left_ctx.get(word, 0) + 1e-10) / total_bt

    if n == 2:
        ctx1 = context
        bg_ctr = model["bigram_counts"].get(ctx1, Counter())
        total_c = model["context1_totals"].get(ctx1, 0)

        if total_c == 0:
            return p_kn_uni

        count_cw = bg_ctr.get(word, 0)
        num_types = len(bg_ctr)
        lambda_c = (D * num_types) / total_c
        return max(count_cw - D, 0.0) / total_c + lambda_c * p_kn_uni

    else:
        ctx2 = context
        tg_ctr = model["trigram_counts"].get(ctx2, Counter())
        total_c = model["context2_totals"].get(ctx2, 0)

        if total_c == 0:
            return kneser_ney_prob(context[-1:], word, {**model, "n": 2})

        count_cw = tg_ctr.get(word, 0)
        num_types = len(tg_ctr)
        lambda_c = (D * num_types) / total_c
        p_lower = kneser_ney_prob(context[-1:], word, {**model, "n": 2})
        return max(count_cw - D, 0.0) / total_c + lambda_c * p_lower


def compute_entropy(tokens, model, alpha=1.0):
    """Per-token cross-entropy. Dispatches to the right probability function based on model type."""
    mtype = model.get("type", "ngram")
    n = model["n"]

    if mtype == "unigram":
        total_lp = sum(math.log2(max(unigram_prob(w, model, alpha), 1e-300))
                       for w in tokens)
        return -total_lp / len(tokens)

    N = len(tokens) - n + 1
    if N <= 0:
        raise ValueError("Test sequence too short for this n-gram order.")

    total_lp = 0.0
    for i in range(N):
        context = tuple(tokens[i : i + n - 1])
        word = tokens[i + n - 1]
        if mtype == "kneser_ney":
            prob = kneser_ney_prob(context, word, model)
        else:
            prob = ngram_prob(context, word, model, alpha=alpha)
        total_lp += math.log2(max(prob, 1e-300))

    return -total_lp / N


def compute_perplexity(entropy):
    """Perplexity = 2 ^ entropy."""
    return 2 ** entropy


def get_log_probs(idx, model, alpha=1.0):
    """
    Log-probabilities over the vocabulary for the current context.
    Backs off to shorter contexts if the full context was unseen.
    Falls back to uniform when no context matches.
    """
    n = model["n"]
    vocab_size = model["vocab_size"]
    context_size = n - 1
    counts = model["counts"]
    context_counts = model["context_counts"]

    context = None
    for ctx_len in range(context_size, 0, -1):
        candidate = tuple(idx[-ctx_len:])
        if candidate in counts:
            context = candidate
            break

    logits = []
    for word in range(vocab_size):
        count_w = counts[context][word] if context is not None else 0
        total = context_counts[context] if context is not None else 0
        prob = (count_w + alpha) / (total + alpha * vocab_size)
        logits.append(math.log(prob))
    return logits


def softmax(logits):
    """Numerically stable softmax."""
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def generate(idx, model, max_new_tokens, decode, temperature=1.0, alpha=1.0, seed=None):
    """Autoregressively generates text from a seed sequence."""
    idx = list(idx)
    rng = random.Random(seed)  # <- INJECTED SEED FOR REPRODUCIBILITY

    for _ in range(max_new_tokens):
        log_probs = get_log_probs(idx, model, alpha=alpha)
        probs = softmax([lp / temperature for lp in log_probs])
        next_tok = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        idx.append(next_tok)

    return decode(idx)

def generate_from_test(lang, lang_data, model, max_new_tokens=200,
                       seed=None, temperature=1.0, alpha=1.0):
    """Picks a random context from the test set and generates text."""
    n = model["n"]
    context_size = n - 1
    test_tokens = lang_data["test"]
    decode = lang_data["tokenizer"].decode

    rng = random.Random(seed)
    max_start = len(test_tokens) - context_size - 1
    if max_start < 0:
        raise ValueError("Test set too small for the chosen n-gram order.")

    start_idx = rng.randint(0, max_start)
    idx = test_tokens[start_idx : start_idx + context_size]
    
    # Pass the seed down to generate() to guarantee repeatable outputs
    generated_text = generate(idx, model, max_new_tokens, decode,
                              temperature=temperature, alpha=alpha, seed=seed)

    print(f"\n  [GENERATE] {lang.upper()} | {n}-gram | temp={temperature} | "
          f"{max_new_tokens} tokens")
    print(f"  Seed: {repr(decode(idx))}")
    print(f"  {'─' * 60}")
    print(f"  {generated_text}")
    print(f"  {'─' * 60}")
    
    return generated_text