"""
SA Language N-grams Comparison.
Author: Tshephang Matlala 223004635

Compares English, Afrikaans, Sepedi, and Zulu using n-gram models
with multiple tokenization strategies and smoothing methods.

"""

import os
import sys
import argparse
import json

from preprocess import process_data
from tokenizer import CharTokenizer, BPETokenizer
from language_model import build_ngram_model, generate_from_test
from evaluate import (
    cross_entropy_tokens,
    cross_entropy_chars,
    run_training_size_experiment,
    tune_alpha,
    evaluate_all_models,
)
from analysis import (
    corpus_statistics,
    compare_corpora,
    domain_bias_analysis,
)
from visualize import (
    plot_language_comparison,
    plot_training_size_curves,
    plot_heaps_law,
    plot_model_comparison,
    plot_domain_bias,
    plot_smoothing_curves,
)


def parse_args():
    """Parses command-line arguments for the experiments."""
    p = argparse.ArgumentParser(description="SA Language N-gram Experiments")
    p.add_argument("--no-bootstrap", action="store_true",
                   help="Skip bootstrap confidence intervals")
    p.add_argument("--n-bootstrap", type=int, default=300,
                   help="Number of bootstrap replicates")
    p.add_argument("--langs", nargs="+", default=None,
                   help="Subset of languages to process")
    p.add_argument("--bpe-sizes", nargs="+", type=int, default=[500, 2000],
                   help="BPE vocabulary sizes")
    return p.parse_args()


def re_encode(lang_data, tok, split):
    """Decodes a split from char ids, then re-encodes with the given tokenizer."""
    orig_decode = lang_data["tokenizer"]["decode"]
    text = orig_decode(lang_data[split])
    return tok.encode(text)


def print_banner(msg):
    """Prints a formatted section banner."""
    width = 64
    print("\n" + "=" * width)
    print(f"  {msg}")
    print("=" * width)


def main():
    args = parse_args()
    os.makedirs("results", exist_ok=True)

    print_banner("SA Language N-grams")

    # Loading data
    data = process_data()
    if args.langs:
        data = {k: v for k, v in data.items() if k in args.langs}
        if not data:
            print(f"ERROR: No matching languages for {args.langs}")
            sys.exit(1)

    # Corpus statistics and Heaps law
    print_banner("Corpus Statistics And Heap's Law")
    all_stats = []
    bias_results = []
    tuning_data = {}

    for lang, ld in data.items():
        stats = corpus_statistics(
            lang,
            tokens_train=ld["train"],
            tokens_val=ld["val"],
            tokens_test=ld["test"],
            tokenizer_name="char",
            cleaned_text=ld["cleaned_text"],
        )
        all_stats.append(stats)
        bias_results.append(domain_bias_analysis(lang, ld["train"]))

    compare_corpora(all_stats)
    plot_heaps_law(all_stats, save_path=os.path.join("results", "heaps_law.png"))
    plot_domain_bias(bias_results, save_path=os.path.join("results", "domain_bias.png"))

    # Per-language per-tokenizer experiments
    print_banner("Model Training & Evaluation")

    bar_chart_results = []
    results_by_lang = {}

    for lang, ld in data.items():
        print(f"\n{'─' * 60}")
        print(f"  {lang.upper()}")
        print(f"{'─' * 60}")

        char_tok = CharTokenizer(ld["cleaned_text"])
        bpe_tokenizers = [
            BPETokenizer(ld["cleaned_text"], vocab_size=sz, name_tag=f"{lang}_{sz}")
            for sz in args.bpe_sizes
        ]
        tokenizers = [char_tok] + bpe_tokenizers
        tuning_data[lang] = {}

        for tok in tokenizers:
            print(f"\n  [Tokenizer: {tok.name}]  vocab={tok.vocab_size}")

            train_enc = re_encode(ld, tok, "train")
            val_enc = re_encode(ld, tok, "val")
            test_enc = re_encode(ld, tok, "test")
            vsize = tok.vocab_size

            # Alpha tuning with detailed logging
            alphas_to_try = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
            for n_ord in (2, 3):
                model_tmp = build_ngram_model(train_enc, n=n_ord, vocab_size=vsize)
                ents = [cross_entropy_tokens(val_enc, model_tmp, alpha=a)
                        for a in alphas_to_try]
                tuning_data[lang].setdefault(n_ord, {})[tok.name] = {
                    "alphas": alphas_to_try, "entropies": ents
                }

            alpha2 = tune_alpha(train_enc, val_enc, vsize, n=2, alphas=alphas_to_try)
            alpha3 = tune_alpha(train_enc, val_enc, vsize, n=3, alphas=alphas_to_try)

            # Evaluate all model variants
            print(f"\n  Evaluating all models …")
            model_results = evaluate_all_models(
                train_enc, val_enc, test_enc, vsize, tok,
                run_bootstrap=(not args.no_bootstrap),
                n_bootstrap=args.n_bootstrap,
            )

            # Store char tokenizer results for cross-language comparison
            if tok.name == "char":
                results_by_lang[lang] = model_results
                mr = model_results

                def get_metric(key, model_name):
                    return mr[model_name].get(key, float("nan"))

                bar_chart_results.append({
                    "lang": lang,
                    "tokenizer": tok.name,
                    "entropy2": get_metric("entropy_token", "bigram_laplace"),
                    "entropy3": get_metric("entropy_token", "trigram_laplace"),
                    "perplexity2": get_metric("perplexity_token", "bigram_laplace"),
                    "perplexity3": get_metric("perplexity_token", "trigram_laplace"),
                    "entropy2_char": get_metric("entropy_char", "bigram_laplace"),
                    "entropy3_char": get_metric("entropy_char", "trigram_laplace"),
                    "perplexity2_char": get_metric("perplexity_char", "bigram_laplace"),
                    "perplexity3_char": get_metric("perplexity_char", "trigram_laplace"),
                    "ci_token_2": mr["bigram_laplace"].get("ci_token"),
                    "ci_token_3": mr["trigram_laplace"].get("ci_token"),
                    "ci_char_2": mr["bigram_laplace"].get("ci_char"),
                    "ci_char_3": mr["trigram_laplace"].get("ci_char"),
                })

            # Training-size experiment
            print(f"\n  Training-size experiment (trigram, alpha={alpha3:.4f}) …")
            exp = run_training_size_experiment(
                train_enc, test_enc, vsize, tok, n=3, alpha=alpha3, seed=42,
            )
            plot_training_size_curves(
                exp,
                title=f"{lang.upper()} – {tok.name} (trigram)",
                save_path=os.path.join("results", f"train_size_{lang}_{tok.name}.png"),
            )

    # Cross-language plots
    print_banner("Cross-Language Comparison Plots")

    for metric in ("entropy", "perplexity"):
        for norm in ("token", "char"):
            plot_language_comparison(
                bar_chart_results, metric=metric, token_norm=norm,
                save_path=os.path.join("results", f"comparison_{metric}_{norm}.png"),
            )

    for metric_key in ("entropy_token", "entropy_char",
                       "perplexity_token", "perplexity_char"):
        plot_model_comparison(
            results_by_lang, metric=metric_key,
            save_path=os.path.join("results", f"model_comparison_{metric_key}.png"),
        )

    # Smoothing curves
    sc_data = {}
    for lang, ord_dict in tuning_data.items():
        sc_data[lang] = {}
        for n_ord, tok_dict in ord_dict.items():
            char_entry = tok_dict.get("char")
            if char_entry:
                sc_data[lang][n_ord] = char_entry

    plot_smoothing_curves(sc_data,
                          save_path=os.path.join("results", "smoothing_curves.png"))

    # Save results
    print_banner("Saving Results")

    for s in all_stats:
        s.pop("heaps_ns", None)
        s.pop("heaps_vs", None)

    summary = {
        "corpus_stats": all_stats,
        "bias_analysis": [{k: v for k, v in b.items()
                           if k not in ("top_n_types",)} for b in bias_results],
        "model_results": {
            lang: {
                mname: {k: v for k, v in mres.items() if not isinstance(v, tuple)}
                for mname, mres in lang_res.items()
            }
            for lang, lang_res in results_by_lang.items()
        },
    }

    summary_path = os.path.join("results", "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None)
    print(f"  Results written to {summary_path}")

    print_banner("All experiments complete. Results saved to results/")

    print("\n--- Generating Sample Text ---")
    for lang, lang_data in results_by_lang.items():
        # Generating from a trained trigram model
        trained_model = lang_data["models"]["trigram_laplace"] 
    
        generate_from_test(
            lang=lang,lang_data=lang_data,
            model=trained_model,
            max_new_tokens=50, 
            seed=42 # Using the seed we added earlier for reproducibility
        )

if __name__ == "__main__":
    main()