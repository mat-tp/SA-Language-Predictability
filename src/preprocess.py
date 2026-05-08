"""
Character-level preprocessing for NCHLT South African language corpora.
Supports English, Afrikaans, Sepedi, Zulu.
Keeps meaningful punctuation (. , ! ? - ') and removes noise.
"""

import re
import os
from pathlib import Path
from util import split_data

# Allows overriding the data directory via environment variable for reproducibility
BASE_DATA_DIR = Path(os.getenv("NCHLT_DATA_DIR", "Data/South_African Languages"))

LANGUAGES = {
    "english": BASE_DATA_DIR / "corpora.nchlt.en/en/1.Corpus/CORP.NCHLT.eng.CLEAN.1.0.0.txt",
    "afrikaans": BASE_DATA_DIR / "corpora.nchlt.af/af/2.Corpora/CORP.NCHLT.af.CLEAN.2.0.txt",
    "sepedi": BASE_DATA_DIR / "corpora.nchlt.nso/nso/2.Corpora/CORP.NCHLT.nso.CLEAN.2.0.txt",
    "zulu": BASE_DATA_DIR / "corpora.nchlt.zu/zu/2.Corpora/CORP.NCHLT.zu.CLEAN.2.0.txt",
}

def remove_metadata(text):
    """Strips corpus header lines — keeps only content after the first <fn> tag."""
    lines = text.split("\n")
    clean_lines = []
    past_header = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.search(r"<fn>.*?</fn>", stripped):
            past_header = True
            continue
        if past_header:
            clean_lines.append(line)

    return "\n".join(clean_lines)

def clean_text(text):
    """
    Normalizes raw corpus text to a clean string.
    Lowercases, removes control characters, separates sentence-ending punctuation,
    and keeps only letters, spaces, and selected punctuation.
    """
    text = text.lower()

    # Remove control characters and metadata tags
    text = re.sub(r"[\x00-\x1F\x7F\x80-\x9F]", " ", text)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)

    # Separate sentence-ending punctuation so it becomes a distinct token
    text = re.sub(r"([.!?])", r" \1 ", text)

    # Keep only lowercase letters (including diacritics), space, and selected punctuation
    text = re.sub(r"[^a-zà-ü ',\-!?.]", "", text, flags=re.UNICODE)

    # Collapse multiple whitespaces into a single space
    return re.sub(r" +", " ", text).strip()

def build_tokenizer(chars):
    """Builds a character-level tokenizer from a sorted character list."""
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    return {
        "stoi": stoi,
        "itos": itos,
        "encode": lambda s: [stoi[c] for c in s],
        "decode": lambda indices: "".join(itos[i] for i in indices),
        "vocab_size": len(chars),
    }

def process_data():
    """Loads, cleans, tokenizes, and splits every language corpus. Returns a dictionary keyed by language."""
    results = {}

    for lang, path in LANGUAGES.items():
        if not path.exists():
            print(f"[WARNING] File not found: {path}. Check your BASE_DATA_DIR.")
            continue

        print(f"  Loading {lang} …")
        with open(path, encoding="utf-8") as f:
            raw_text = f.read()

        cleaned_text = clean_text(remove_metadata(raw_text))
        chars = sorted(set(cleaned_text))
        tokenizer = build_tokenizer(chars)
        
        encoded = tokenizer["encode"](cleaned_text)
        train, val, test = split_data(encoded)

        print(f"\n{lang.upper():<12} vocab={len(chars):>3}  "
              f"train={len(train):,}  val={len(val):,}  test={len(test):,}")

        results[lang] = {
            "cleaned_text": cleaned_text,
            "stats": {
                "raw_length": len(raw_text),
                "clean_length": len(cleaned_text),
                "chars": chars,
                "vocab_size": len(chars),
            },
            "tokenizer": tokenizer,
            "train": train,
            "val": val,
            "test": test,
        }

    return results