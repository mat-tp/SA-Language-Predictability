""" Loads and cleans the 4 NCHLT South African language corpora (English, Afrikaans, Sepedi, Zulu), then splits each one into
train / validation / test sets. """

import re
import os
from pathlib import Path

# Corpus file paths
BASE_DATA_DIR = Path(os.getenv("NCHLT_DATA_DIR", "Data/South_African Languages"))

LANGUAGES = {
    "english":   BASE_DATA_DIR / "corpora.nchlt.en/en/1.Corpus/CORP.NCHLT.eng.CLEAN.1.0.0.txt",
    "afrikaans": BASE_DATA_DIR / "corpora.nchlt.af/af/2.Corpora/CORP.NCHLT.af.CLEAN.2.0.txt",
    "sepedi":    BASE_DATA_DIR / "corpora.nchlt.nso/nso/2.Corpora/CORP.NCHLT.nso.CLEAN.2.0.txt",
    "zulu":      BASE_DATA_DIR / "corpora.nchlt.zu/zu/2.Corpora/CORP.NCHLT.zu.CLEAN.2.0.txt",
}


# Text cleaning helpers
def _remove_metadata(text):
    """ The NCHLT files start with header lines enclosed in <fn>…</fn> tags. This function skips everything up to and including those tags,
    returning only the actual corpus sentences. """
    lines = text.split("\n")
    body_lines = []
    past_header = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"<fn>.*?</fn>", stripped):
            past_header = True
            continue
        if past_header:
            body_lines.append(line)

    return "\n".join(body_lines)


def clean_text(text):
    """ Normalises raw corpus text into a clean, lowercase string. """

    text = text.lower()

    text = re.sub(r"[\x00-\x1F\x7F\x80-\x9F]", " ", text)  # Remove ASCII control characters and a common Windows-1252 range
    text = re.sub(r"_+", " ", text) # remove the underscores
    text = re.sub(r"<[^>]+>", " ", text) # Remove any XML/HTML tags
    text = re.sub(r"([.!?])", r" \1 ", text) # Pad sentence-ending punctuation so it splits cleanly as its own word

    # Keep only valid characters: Unicode letters, spaces, and the punctuation
    # set ' , - ! ? .
    allowed = " ',-\!?."
    filtered = []
    for ch in text:
        if ch.isalpha() or ch.isspace() or ch in allowed:
            filtered.append(ch)
    text = "".join(filtered)

    # Collapse multiple spaces
    return re.sub(r" +", " ", text).strip()

# Train / val / test splits
def split_data(encoded_tokens, train_ratio=0.8, val_ratio=0.1):
    """ Splits a list of token ids into train, validation, and test portions. Default split: 80 % train : 10 % val : 10 % test."""

    n = len(encoded_tokens)
    train_end = int(n * train_ratio)
    val_end   = train_end + int(n * val_ratio)

    train = encoded_tokens[:train_end]
    val   = encoded_tokens[train_end:val_end]
    test  = encoded_tokens[val_end:]

    return train, val, test

# Character encoding and decoding methods 

def _make_encoder(stoi):
    return lambda s: [stoi[c] for c in s]


def _make_decoder(itos):
    return lambda ids: "".join(itos[i] for i in ids)

# Main entry point
def process_data():
    """ Loading every language corpus, cleans the text, character-level encoding. """
    results = {}

    for lang, path in LANGUAGES.items():
        if not path.exists():
            print(f"  [WARNING] File not found: {path}")
            print(f"            Set NCHLT_DATA_DIR to your data root and retry.")
            continue

        print(f"  Loading {lang} …")
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()

        cleaned = clean_text(_remove_metadata(raw_text))
        char_vocab = sorted(set(cleaned))

        stoi = {ch: i for i, ch in enumerate(char_vocab)}
        itos = {i: ch for i, ch in enumerate(char_vocab)}

        encode = _make_encoder(stoi)
        decode = _make_decoder(itos)

        encoded = encode(cleaned)
        train, val, test = split_data(encoded)

        print(f"  {lang.upper():<12}  vocab={len(char_vocab):>3}  "
              f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

        results[lang] = {
            "cleaned_text": cleaned,
            "char_vocab":   char_vocab,
            "train": train,
            "val":   val,
            "test":  test,
            "_char_encode": encode,
            "_char_decode": decode,
        }

    return results