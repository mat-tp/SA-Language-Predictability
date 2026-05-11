""" Character-level tokenizer.

Every unique character in the training text becomes one token.
This is the baseline tokenizer: it makes no assumptions about word
boundaries or morphology. """


class CharTokenizer:
    """ Splits text into individual characters. Every unique character found here becomes part of the vocabulary. """

    name = "char"

    def __init__(self, text: str):
        chars = sorted(set(text))   
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list:
        """Converts a string into a list of integer character ids."""
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list) -> str:
        """Converts a list of character ids back into a string."""
        return "".join(self._itos[i] for i in ids)

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size})"
