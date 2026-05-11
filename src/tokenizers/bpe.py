""" Byte Pair Encoding (BPE) tokenizer Implementation. """

from collections import Counter

class BPETokenizer:
    """ Byte Pair Encoding tokenizer. """

    def __init__(self, text: str, vocab_size: int = 500, name_tag: str = ""):
        
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")

        self.name = f"bpe_{vocab_size}" + (f"_{name_tag}" if name_tag else "")

        # Build initial character vocabulary
        base_chars = sorted(set(text))
        self._eow = "▁"   # end-of-word marker

        all_base = base_chars + [self._eow]
        self._vocab   = {ch: i for i, ch in enumerate(all_base)}
        self._id2tok  = {i: ch for i, ch in enumerate(all_base)}

        # Represent training corpus as word-tuples (with frequency counts)
        word_freqs = Counter()
        for word in text.split():
            tokenised = tuple(list(word) + [self._eow])
            word_freqs[tokenised] += 1

        # Merging loop
        num_merges = vocab_size - len(self._vocab)
        self._merges = {}  # merged token

        for _ in range(max(0, num_merges)):
            pair_counts = _count_pairs(word_freqs)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = best_pair[0] + best_pair[1]

            new_id = len(self._vocab)
            self._vocab[new_token]  = new_id
            self._id2tok[new_id]    = new_token
            self._merges[best_pair] = new_token

            word_freqs = _apply_merge(word_freqs, best_pair, new_token)

        self.vocab_size = len(self._vocab)

        # sort tokens for greedy longest-match encoding
        self._sorted_tokens = sorted(self._vocab.keys(),
                                     key=len, reverse=True)

    def encode(self, text: str) -> list:
        """Encode a string into a list of BPE subword ids."""
        ids = []
        for word in text.split():
            word_str = word + self._eow
            ids.extend(self._encode_word(word_str))
        return ids

    def decode(self, ids: list) -> str:
        """Decode a list of ids back into text, replacing EOW with space."""
        tokens = [self._id2tok.get(i, "") for i in ids]
        text   = "".join(tokens).replace(self._eow, " ")
        return text.strip()

    def __repr__(self):
        return f"BPETokenizer(vocab_size={self.vocab_size}, name={self.name!r})"

    def _encode_word(self, word_str: str) -> list:
        """Greedy longest-match encoding of a single word """
        ids = []
        pos = 0
        while pos < len(word_str):
            matched = False
            for token in self._sorted_tokens:
                end = pos + len(token)
                if word_str[pos:end] == token:
                    ids.append(self._vocab[token])
                    pos = end
                    matched = True
                    break
            if not matched:
                pos += 1   # fallback: skip one character
        return ids

def _count_pairs(word_freqs):
    """Count adjacent token pairs, weighted by word frequency."""
    counts = Counter()
    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            counts[(word_tuple[i], word_tuple[i + 1])] += freq
    return counts


def _apply_merge(word_freqs, pair, new_token):
    """Replace every occurrence of `pair` with `new_token` in the corpus."""
    left, right = pair
    new_freqs = {}

    for word_tuple, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word_tuple):
            if (i < len(word_tuple) - 1
                    and word_tuple[i] == left
                    and word_tuple[i + 1] == right):
                new_word.append(new_token)
                i += 2
            else:
                new_word.append(word_tuple[i])
                i += 1
        new_freqs[tuple(new_word)] = freq

    return new_freqs