""" N-gram language models with Add-alpha (Laplace) smoothing.
    
    Supports n = 1 (unigram) through n = 5 (5-gram). """

import math
import random
from collections import Counter, defaultdict

# Model constructions

def build_unigram_model(tokens, vocab_size):
    """ Builds a unigram (n=1) model from a list of token ids. A unigram model ignores word order. It only records how often each token appears."""

    return {
        "type":       "unigram",
        "n":          1,
        "vocab_size": vocab_size,
        "counts":     Counter(tokens),
        "total":      len(tokens),
    }


def build_ngram_model(tokens, n, vocab_size):
    """ Builds an n-gram model (n ≥ 2) from a list of token ids. For each (n-1)-token context, the model stores a count of how many
    times each next token followed that context.  At prediction time, these counts are smoothed with Add-alpha to handle unseen events. """

    if n < 2:
        raise ValueError(f"n must be ≥ 2 for build_ngram_model; got {n}. Use build_unigram_model for n=1.")
    
    if len(tokens) < n:
        raise ValueError(f"Training sequence has only {len(tokens)} tokens, which is too short for n={n}.")

    counts = defaultdict(Counter)  # {next_token: count}
    context_totals = Counter() # total occurrences

    for i in range(len(tokens) - n + 1):
        context = tuple(tokens[i : i + n - 1])
        word    = tokens[i + n - 1]
        counts[context][word]    += 1
        context_totals[context]  += 1

    return {
        "type":           "ngram",
        "n":              n,
        "vocab_size":     vocab_size,
        "counts":         counts,
        "context_totals": context_totals,
    }


# Probability estimation
def unigram_prob(word, model, alpha=1.0):
    """ P(word) with Add-alpha smoothing for a unigram model. alpha=1.0 is standard Laplace (add-one) smoothing. """

    numerator = model["counts"].get(word, 0) + alpha
    denominator = model["total"] + alpha * model["vocab_size"]
    return numerator / denominator


def ngram_prob(context, word, model, alpha=1.0):
    """ P(word | context) with Add-alpha smoothing for an n-gram model. """
    
    count_context_word = model["counts"].get(context, Counter()).get(word, 0)
    total_context      = model["context_totals"].get(context, 0)

    numerator   = count_context_word + alpha
    denominator = total_context + alpha * model["vocab_size"]
    return numerator / denominator


# Metrics : Entropy and perplexity

def compute_entropy(tokens, model, alpha=1.0):
    """ Computes per-token cross-entropy in bits for the given token sequence.
        Cross-entropy H = - (1/N) Σ log₂ P(wᵢ | context). 
        
        Lower entropy means the model assigns higher probability to the actual next tokens — The language is more predictable under this model. """
    
    mtype = model["type"]
    n = model["n"]

    if mtype == "unigram":
        log_sum = sum(
            math.log2(max(unigram_prob(w, model, alpha), 1e-300))
            for w in tokens
        )
        return -log_sum / len(tokens)

    # For n-gram models we start at position n-1 (we need n-1 tokens of context)
    num_predictions = len(tokens) - n + 1
    if num_predictions <= 0:
        raise ValueError(
            f"Token sequence is too short ({len(tokens)} tokens) for n={n}."
        )

    log_sum = 0.0
    for i in range(num_predictions):
        context = tuple(tokens[i : i + n - 1])
        word    = tokens[i + n - 1]
        prob    = ngram_prob(context, word, model, alpha)
        log_sum += math.log2(max(prob, 1e-300))

    return -log_sum / num_predictions


def compute_perplexity(entropy_bits):
    """ Converts cross-entropy (bits) to perplexity: PP = 2^H.
        Perplexity is the effective vocabulary size the model is choosing from at each step.  Lower perplexity = better model. """
    return 2 ** entropy_bits


# Text generation 
def generate_from_model(lang, lang_data, model, tokenizer,max_new_tokens=100, seed=42,temperature=1.0, alpha=1.0):
    """ Generates a short text sample from the model, seeded by a random context taken from the test set.

    This is purely qualitative for inspecting whether the model has picked up real patterns in each language."""

    n = model["n"]
    context_size = n - 1
    test_tokens  = lang_data["test"]

    rng = random.Random(seed)

    # Pick a random starting position in the test set
    max_start = len(test_tokens) - context_size - 1
    if max_start < 0:
        raise ValueError("Test set is too small to seed generation for this n-gram order.")

    start = rng.randint(0, max_start)
    seed_ids = list(test_tokens[start : start + context_size])

    # Generate tokens one at a time
    generated_ids = list(seed_ids)
    for _ in range(max_new_tokens):
        log_probs  = _context_log_probs(generated_ids, model, alpha)
        probs = _softmax([lp / temperature for lp in log_probs])
        next_token = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        generated_ids.append(next_token)

    generated_text = tokenizer.decode(generated_ids)
    seed_text = tokenizer.decode(seed_ids)

    print(f"\n  [{lang.upper()}]  {n}-gram generation  (temp={temperature})")
    print(f"  Seed context : {seed_text!r}")
    print(f"  {'─' * 56}")
    print(f"  {generated_text}")
    print(f"  {'─' * 56}")

    return generated_text


def _context_log_probs(token_ids, model, alpha=1.0):
    """ Returns log P(w | context) for every token w in the vocabulary, given the current sequence of generated token ids. """

    n = model["n"]
    vocab_size = model["vocab_size"]
    counts = model["counts"]
    ctx_totals = model["context_totals"]

    # Try progressively shorter contexts until we find one that was seen
    context = None
    for ctx_len in range(n - 1, 0, -1):
        candidate = tuple(token_ids[-ctx_len:])
        if candidate in counts:
            context = candidate
            break

    log_probs = []
    for word in range(vocab_size):
        count_w = counts[context][word] if context is not None else 0
        total   = ctx_totals[context]   if context is not None else 0
        prob    = (count_w + alpha) / (total + alpha * vocab_size)
        log_probs.append(math.log(prob))

    return log_probs


def _softmax(logits):
    """Numerically stable softmax, subtracts the maximum before exponentiating."""
    max_l = max(logits)
    exps  = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]