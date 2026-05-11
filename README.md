# Comparing Predictability of South African Languages

**Author:** Tshephang P-A-N Matlala (223004635)  
**Affiliation:** Academy of Computer Science and Software Engineering, University of Johannesburg  
**Module:** Compiler Construction

**Research Focus:** Analysing the **predictability of South African languages** by computing  
**entropy** and **perplexity** with n‑gram language models.

The study compares English, Afrikaans, Sepedi, and isiZulu, asking how morphological  
structure affects predictability across languages.

---

## What this project does

- Loads the NCHLT corpora for four South African languages.
- Applies three tokenization strategies: **character**, **word**, and **subword (BPE)**.
- Trains n‑gram language models from n = 1 (unigram) up to n = 5 (5‑gram).
- Uses only **Add‑alpha (Laplace) smoothing** (α tuned on a validation set).
- Evaluates models with **per‑token** and **per‑character cross‑entropy** and **perplexity**.
- Provides both a **command‑line script** (`main.py`) and a **graphical desktop app** (`run.py`) that works with any plain‑text corpus.

---

## Project's library requirements

- They are stored in the `requirements.txt` file.

```bash
  pip install -r requirements.txt
```

---

## Running the application

- Usage:

1. For running on Graphical user interface

```bash
  python run.py
```

2. For running on Command line interface

```bash
  python main.py
```

3. Customizing the CLI - Providing arguments for selection of languages, N for n-gram, BPE-sizes [int]

```bash
      python main.py --help
```

## Methodical approach

1. Text cleaning : Remove metadata, lower‑case, handle punctuation, keep only desired characters.

2. Tokenization : Three strategies are applied independently to each language:

- Character – each unique character is a token.

- Word – whitespace‑separated words (closed vocabulary).

- Byte‑Pair Encoding (BPE) – subword tokens learned from the training text (vocabulary size configurable). Reference: Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units". ACL 2016.

3. Data splitting – The encoded sequence is split into train (80%) / validation (10%) / test (10%), with no shuffling to preserve order and reproducibility.

4. Model training – For each tokenizer and n‑gram order (1–5), an n‑gram count‑based model is built.

5. Smoothing – Only Add‑alpha (Laplace) smoothing is used. The α coefficient is tuned on the validation set by testing values {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0} and picking the one that minimises validation entropy.

6. Evaluation – The trained models are evaluated on the held‑out test set. Two normalised metrics are reported:

- Per‑token cross‑entropy / perplexity

- Per‑character cross‑entropy / perplexity (the only fair comparison across tokenizers of different granularities)

---

## Metrics

#### Cross-Entropy (H)

- Average information (bits) needed to predict the next token. Lower = more predictable.

- Measures the information from 2 probabilities.

#### Perplexity (PPL)

- 2^H; the effective number of equally likely choices the model considers at each step. Lower = better model.

- Measure how confused a model is in predicting the next word.

---

## Results

The key finding is that BPE subword tokenization (vocab size 500) yields the lowest per‑character entropy for all four languages, making it the most efficient choice for n‑gram modelling across these typologically diverse South African languages.

---

YouTube Video 1 Link: https://youtu.be/gP3JngH-Sio

Youtube Video 2 Link: \_

---

## References

1. Data Source : NCHLT (National Centre for Human Language Technology) corpora provided by SADiLaR

- @misc{20.500.12185/301, title = {{NCHLT} English Text Corpora}, author = {Martin Puttkammer and Martin Schlemmer and Wikus Pienaar and Ruan Bekker},
  url = {https://hdl.handle.net/20.500.12185/301}, note = {{SADiLaR} Language Resource Repository, License: Creative Commons Attribution 2.5 South Africa License: http://creativecommons.org/licenses/by/2.5/za/legalcode}, year = {2016}}

- @misc{20.500.12185/708, title = {{NCHLT} Sepedi {POS} and Lemma annotated corpus}, author = {Gaustad, Tanja},
  url = {https://hdl.handle.net/20.500.12185/708}, note = {{SADiLaR} Language Resource Repository, License: Creative Commons Attribution 4.0 International},year = {2026}}

- @misc{20.500.12185/701, title = {{isiZulu} Domain corpus {POS} annotated (5 domains)},author = {Gaustad, Tanja}, url = {https://hdl.handle.net/20.500.12185/701}, note = {{SADiLaR} Language Resource Repository, License: Creative Commons Attribution 4.0 International},year = {2026}}

- @misc{20.500.12185/142, title = {Afrikaans Part of Speech Data},url = {https://hdl.handle.net/20.500.12185/142}, note = {{SADiLaR} Language Resource Repository},year = {2015}}
