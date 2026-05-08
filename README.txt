YouTube Link: https://youtu.be/gP3JngH-Sio

Pass alpha to ngram_prob in evaluation functions

File: evaluate.py

Lines: inside cross_entropy_tokens and cross_entropy_chars

Change prob = ngram_prob(context, word, model) → prob = ngram_prob(context, word, model, alpha=alpha)

Fix character whitelist in preprocessing

File: preprocess.py, function clean_text

Regex [^a-zà-ü ',\-!?.] removes letters like š (Sepedi/Zulu).

Replace with a dynamic character set that keeps all alphabetic letters from the corpus.

🟠 Important Improvements
Shuffle training data before sub‑sampling in training‑size experiment

File: evaluate.py, function run_training_size_experiment

Use rng.shuffle(train_all) before taking train_all[:sample_size] to avoid order bias.

Verify <fn> metadata removal doesn’t drop real text

File: preprocess.py, function remove_metadata

Check that no text is lost when skipping the first <fn> line.

Save evaluation metrics to a CSV file

In main.py, after computing metrics, append to a results/metrics.csv for later analysis without re‑running.

BPE vocabulary contains <unk> even if unused

File: tokenizer.py, BPETokenizer

Optionally remap ids to exclude <unk> (id 0) for cleaner vocabulary size reporting.

📝 Code Hygiene
Document all functions

Add docstrings to any remaining functions in main.py, visualize.py.

Use consistent random seeds across all experiments

Ensure seed is passed to random.Random() everywhere.

Add error handling for missing files

preprocess.py already prints a warning but continues; consider raising an exception.

🔧 For Your “Build Tokenizations Myself” Requirement
Tokenizers must implement: encode(text), decode(ids), vocab_size, name

You can design your own tokenizer class(es) from scratch.

Ensure they match this interface so the rest of the pipeline works without change.

Data splitting logic

Already implemented in util.py (split_data).

You can modify it (e.g., different ratios) as long as you return train, val, test lists of ints.

✅ How to Use This List
Start with the critical fixes (1 and 2) – they affect correctness.

Then apply the improvements (3‑6) to make your results robust and reproducible.

Finally, polish with the hygiene items (7‑9).

Save this list and tick off each item when you fix it. Let me know if you need help with any specific code change.

