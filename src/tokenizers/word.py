""" Word-level tokenizer. 
Splits text on whitespace and assigns one integer id to each unique
word type. This gives the largest vocabulary and the shortest token sequences.

Closed-vocabulary assumption where unseen words are silently skipped."""

class WordTokenizer:
    """ Whitespace-based word tokenizer. Every unique whitespace-separated word found here becomes part of the vocabulary. """

    name = "word"

    def __init__(self, text: str):

        words = sorted(set(text.split())) 
        self._stoi = {w: i for i, w in enumerate(words)}
        self._itos = {i: w for i, w in enumerate(words)}
        self.vocab_size = len(words)

    def encode(self, text: str) -> list:
        """ Converts a string into a list of integer word ids. Unseen words are omitted.
        """
        return [self._stoi[w] for w in text.split() if w in self._stoi]

    def decode(self, ids: list) -> str:
        """Converts a list of word ids back into a space-separated string."""
        return " ".join(self._itos[i] for i in ids)

    def __repr__(self):
        return f"WordTokenizer(vocab_size={self.vocab_size})"