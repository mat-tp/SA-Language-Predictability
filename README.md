# Comparing Predictability of South African Languages

Author : Tshephang Matlala.
Affiliation: Department of Computer Science and Software Engineering, University of Johannesburg
Module : Compiler Construction

Research Focus : Analyzing the **predictability of South African languages** by computing **entropy** and **perplexity** using n-gram language models.

Comparing predictability of South African languages {English, Afrikaans, Sepedi, and isiZulu}. The goal is to compare how morphological structure of the languages are between languages.

## Running the application

- All the functionality is in the main by running (trains, and loads the results- this also trains the BPE which uses bootsraping which can take hours) :python main.py

- Usage:
  python main.py # full run
  python main.py --no-bootstrap # skip bootstrap
  python main.py --langs english zulu # subset of languages

- \_

## Methodical Approach

1. Text processing : Cleaning the data by removing the metadata, normalizing the text, and handling punctuations.
2. Tokenization : Adapted 2 methods

- Character tokenization = character sequences on the text.
- Sub-word tokenization = Using Byte Pair Encoding

3. Smoothing methods : Handling zero probabilities

- Laplace (Add- alpha)
- Kneser-Ney Smoothing

4. Evaluated using sliding window approach on the test set.

---

## Metrics

#### Cross-Entropy (H)

- Measures the information from 2 probabilities.

#### Perplexity (PPL)

- Measure how confused a model is in predicting the next word.

#### Zipf's Law Analysis

- Helps in evaluating distribution of word frequencies across different languages.

---

## Results

Here is a breakdown of the results.

---

YouTube Video 1 Link: https://youtu.be/gP3JngH-Sio
Youtube Video 2 Link: \_

---

## References

1. Data Source : NCHLT (National Centre for Human Language Technology) corpora provided by SADiLaR

@misc{20.500.12185/301,
title = {{NCHLT} English Text Corpora},
author = {Martin Puttkammer and Martin Schlemmer and Wikus Pienaar and Ruan Bekker},
url = {https://hdl.handle.net/20.500.12185/301},
note = {{SADiLaR} Language Resource Repository, License: Creative Commons Attribution 2.5 South Africa License: http://creativecommons.org/licenses/by/2.5/za/legalcode},
year = {2016}
}

More Reference for other languages...
