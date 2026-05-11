""" A Comparative Study of N-Gram Predictability Across South African Languages Using Character, Word, and Subword Tokenization.
    
    Author : Tshephang Matlala  (223004635)

    Languages : English, Afrikaans, Sepedi, Zulu  (NCHLT corpora)
    Tokenizers: char · word · BPE (500 subwords) · BPE (2 000 subwords)
    N-gram orders: 1 (unigram) through 5 (5-gram)
    Metrics: cross-entropy (per-token and per-character) & perplexity

"""

import os
import sys
import json
import argparse

from preprocess import process_data
from tokenizers import CharTokenizer, WordTokenizer, BPETokenizer
from language_model import generate_from_model
from evaluate import evaluate_all_orders
from visualize import plot_ngram_comparison, plot_tokenizer_comparison

# Handling Command-line arguments
def parse_args():
    p = argparse.ArgumentParser(description="SA Language N-Gram Predictability Study")
    p.add_argument("--langs", nargs="+", default=None, help="Run only a subset of languages, e.g. --langs english zulu")
    p.add_argument("--max-n", type=int, default=5, help="Highest n-gram order to evaluate (default: 5)")
    p.add_argument("--bpe-sizes", nargs="+", type=int, default=[500, 2000], help="BPE vocabulary sizes to include (default: 500 2000)")
    p.add_argument("--no-generate", action="store_true", help="Skip the sample-text generation step")
    
    return p.parse_args()

# CLI Layout
def _banner(msg):
    """Prints a clearly visible section header to the console."""
    width = 64
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)

def _re_encode(lang_data, tokenizer, split_name):
    """Decodes char-encoded split back to text, then re-encodes."""
    
    original_text = lang_data["_char_decode"](lang_data[split_name])
    return tokenizer.encode(original_text)


def _build_tokenizers(cleaned_text, bpe_sizes):
    """ Constructs all tokenizers for one language.
        Returns a list in a consistent order: char -> word -> BPE variants. """
    
    tokenizers = [
        CharTokenizer(cleaned_text),
        WordTokenizer(cleaned_text),
    ]
    for size in bpe_sizes:
        tokenizers.append(BPETokenizer(cleaned_text, vocab_size=size))
    return tokenizers

# Entry point
def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    _banner("SA Language N-Gram Study")

    # Load corpora
    _banner("Loading Corpora")
    data = process_data()

    # restrict to a subset of languages
    if args.langs:
        data = {k: v for k, v in data.items() if k in args.langs}
        if not data:
            print(f"  ERROR: None of {args.langs} were found in the loaded data.")
            sys.exit(1)

    if not data:
        print("  ERROR: No language data loaded.  Check your data directory.")
        sys.exit(1)

    # Train and evaluate
    _banner("Training and Evaluation")

    # all_results collects one dict per (language, tokenizer) pair.
    # Each dict stores the results for every n-gram order.
    all_results = []

    for lang, lang_data in data.items():
        print(f"\n{'─' * 60}")
        print(f"  {lang.upper()}")
        print(f"{'─' * 60}")

        tokenizers = _build_tokenizers(lang_data["cleaned_text"], args.bpe_sizes)

        for tok in tokenizers:
            print(f"\n  Tokenizer: {tok.name}  (vocab size = {tok.vocab_size:,})")

            # Re-encode the train / val / test splits for this tokenizer
            train = _re_encode(lang_data, tok, "train")
            val = _re_encode(lang_data, tok, "val")
            test = _re_encode(lang_data, tok, "test")

            print(f"  Tokens — train: {len(train):,}  val: {len(val):,}  test: {len(test):,}")

            # Evaluate n=1..max_n, including alpha tuning
            order_results = evaluate_all_orders(
                train_tokens = train,
                val_tokens = val,
                test_tokens = test,
                vocab_size = tok.vocab_size,
                tokenizer = tok,
                max_n = args.max_n,
            )

            # Store alongside the language and tokenizer labels
            record = {"lang": lang, "tokenizer": tok.name}
            record.update(order_results)
            all_results.append(record)

    # Save results
    _banner("Saving Results")

    # Convert int keys to strings so json.dump can handle them
    serialisable = []
    for r in all_results:
        entry = {"lang": r["lang"], "tokenizer": r["tokenizer"]}
        for k, v in r.items():
            if isinstance(k, int):
                entry[str(k)] = v
        serialisable.append(entry)

    summary_path = os.path.join("results", "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Results saved to {summary_path}")

    # Plot results
    _banner("Generating Plots")

    # For the n-gram comparison we only use char tokenizer results
    # (char is the common baseline across all languages)
    char_results = [r for r in all_results if r["tokenizer"] == "char"]

    for metric in ("entropy_token", "entropy_char", "perplexity_token", "perplexity_char"):
        plot_ngram_comparison(
            char_results,
            metric=metric,
            save_path=os.path.join("results", f"ngram_comparison_{metric}.png"),
        )

    # Tokenizer comparison: use trigram (n=3) as a representative order and per-character entropy as the fair cross-tokenizer metric
    for n_order in (2, 3, 5):
        plot_tokenizer_comparison(
            all_results,
            metric="entropy_char",
            n_order=n_order,
            save_path=os.path.join("results", f"tokenizer_comparison_n{n_order}.png"),
        )

    # Generate sample text (qualitative inspection)
    if not args.no_generate:
        _banner("Sample Text Generation  (trigram, char tokenizer)")
        for lang, lang_data in data.items():
            # Rebuild the char tokenizer and trigram model for generation
            char_tok  = CharTokenizer(lang_data["cleaned_text"])
            train_enc = _re_encode(lang_data, char_tok, "train")

            from language_model import build_ngram_model
            model = build_ngram_model(train_enc, n=3, vocab_size=char_tok.vocab_size)

            # Retrieve the best alpha found during evaluation
            lang_record = next(
                (r for r in all_results
                 if r["lang"] == lang and r["tokenizer"] == "char"),
                None
            )
            alpha = lang_record[3]["alpha"] if lang_record and 3 in lang_record else 0.1

            generate_from_model(
                lang       = lang,
                lang_data  = lang_data,
                model      = model,
                tokenizer  = char_tok,
                max_new_tokens = 80,
                seed       = 42,
                alpha      = alpha,
            )

    _banner("Done : all results saved to results/")

if __name__ == "__main__":
    main()