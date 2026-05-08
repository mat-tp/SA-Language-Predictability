"""
Two tokenization strategies for the SA language study.
CharTokenizer — character-level baseline.
BPETokenizer — subword Byte Pair Encoding via SentencePiece.
Both expose the same interface: encode, decode, vocab_size, name.
"""

import io
import os
import tempfile

import sentencepiece as spm


class CharTokenizer:
    """Character-level tokenizer. Builds vocabulary from every unique character in the training text."""

    def __init__(self, text):
        chars = sorted(set(text))
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.name = "char"

    def encode(self, text):
        """Encodes a string to a list of character ids."""
        return [self._stoi[ch] for ch in text]

    def decode(self, ids):
        """Decodes a list of character ids back to a string."""
        return "".join(self._itos[i] for i in ids)

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer backed by SentencePiece.
    Trains on sentences split at punctuation boundaries.
    Keeps the model in memory — no files left behind.
    """

    def __init__(self, text, vocab_size=500, name_tag=""):
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")

        self.vocab_size = vocab_size
        self.name = f"bpe_{vocab_size}" + (f"_{name_tag}" if name_tag else "")

        # Split text into sentences using known token boundaries
        tokens = text.split()
        sentences = []
        buffer = []

        for token in tokens:
            buffer.append(token)
            if token in ('.', '!', '?'):
                sentences.append(' '.join(buffer))
                buffer = []

        # Any remaining tokens
        if buffer:
            sentences.append(' '.join(buffer))

        spm_text = "\n".join(sentences)

        # Train SentencePiece on a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         encoding="utf-8", delete=False) as tmp:
            tmp.write(spm_text)
            tmp_path = tmp.name

        try:
            model_buffer = io.BytesIO()
            spm.SentencePieceTrainer.train(
                input=tmp_path,
                model_writer=model_buffer,
                vocab_size=vocab_size,
                model_type="bpe",
                character_coverage=1.0,
                pad_id=-1,
                unk_surface="▁",
                max_sentence_length=1_000_000,
            )
        finally:
            os.unlink(tmp_path)

        # Load trained model from buffer
        model_buffer.seek(0)
        self._sp = spm.SentencePieceProcessor()
        self._sp.load_from_serialized_proto(model_buffer.read())

        # Use actual vocab size (may be slightly smaller for small corpora)
        self.vocab_size = self._sp.get_piece_size()

    def encode(self, text):
        """Encodes text to a list of BPE subword ids."""
        return self._sp.encode(text, out_type=int)

    def decode(self, ids):
        """Decodes a list of BPE subword ids back to a string."""
        return self._sp.decode(ids)

    def id_to_piece(self, id):
        """Returns the subword string for a given token id."""
        return self._sp.id_to_piece(id)

    def piece_to_id(self, piece):
        """Returns the token id for a given subword string."""
        return self._sp.piece_to_id(piece)

    def __repr__(self):
        return f"BPETokenizer(vocab_size={self.vocab_size}, name={self.name!r})"


def build_tokenizers(text, bpe_sizes=(500, 2000)):
    """Builds all tokenizers for a language in one call. Returns a dict keyed by tokenizer name."""
    tokenizers = {}

    char_tok = CharTokenizer(text)
    tokenizers[char_tok.name] = char_tok

    for size in bpe_sizes:
        bpe_tok = BPETokenizer(text, vocab_size=size)
        tokenizers[bpe_tok.name] = bpe_tok

    return tokenizers