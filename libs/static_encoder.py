from collections import Counter

import torch

from .reserved_tokens import EOS_INDEX
from .reserved_tokens import RESERVED_ITOS
from .reserved_tokens import UNKNOWN_INDEX

def _tokenize(s):
    return s.split()


class Encoder(object):
    """ Base class for a text encoder.
    """

    def __init__(self):  # pragma: no cover
        raise NotImplementedError

    def encode(self, string):  # pragma: no cover
        """ Returns a :class:`torch.LongTensor` encoding of the `text`. """
        raise NotImplementedError

    def batch_encode(self, strings, *args, **kwargs):
        """ Returns a :class:`list` of :class:`torch.LongTensor` encoding of the `text`. """
        return [self.encode(s, *args, **kwargs) for s in strings]

    def decode(self, tensor):  # pragma: no cover
        """ Given a :class:`torch.Tensor`, returns a :class:`str` representing the decoded text.
        Note that, depending on the tokenization method, the decoded version is not guaranteed to be
        the original text.
        """
        raise NotImplementedError

    @property
    def vocab_size(self):
        """ Return the size (:class:`int`) of the vocabulary. """
        return len(self.vocab)

    @property
    def vocab(self):  # pragma: no cover
        """ Returns the vocabulary (:class:`list`) used to encode text. """
        return NotImplementedError


class StaticTokenizerEncoder(Encoder):
    """ Encodes the text using a tokenizer.
    Args:
        sample (list of strings): Sample of data to build dictionary on.
        min_occurrences (int, optional): Minimum number of occurrences for a token to be added to
          dictionary.
        tokenize (callable): :class:``callable`` to tokenize a string.
        append_eos (bool, optional): If `True` append EOS token onto the end to the encoded vector.
        reserved_tokens (list of str, optional): Tokens added to dictionary; reserving the first
            `len(reserved_tokens)` indexes.
    Example:
        >>> encoder = StaticTokenizerEncoder(["This ain't funny.", "Don't?"],
                                             tokenize=lambda s: s.split())
        >>> encoder.encode("This ain't funny.")
         5
         6
         7
        [torch.LongTensor of size 3]
        >>> encoder.vocab
        ['<pad>', '<unk>', '</s>', '<s>', '<copy>', 'This', "ain't", 'funny.', "Don't?"]
        >>> encoder.decode(encoder.encode("This ain't funny."))
        "This ain't funny."
    """

    def __init__(self,
                 sample,
                 min_occurrences=1,
                 append_eos=False,
                 tokenize=_tokenize,
                 reserved_tokens=RESERVED_ITOS):
        if not isinstance(sample, list):
            raise TypeError('Sample must be a list of strings.')

        self.tokenize = tokenize
        self.append_eos = append_eos
        self.tokens = Counter()

        for text in sample:
            self.tokens.update(self.tokenize(text))

        self.itos = reserved_tokens.copy()
        self.stoi = {token: index for index, token in enumerate(reserved_tokens)}
        for token, count in self.tokens.items():
            if count >= min_occurrences:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

    @property
    def vocab(self):
        return self.itos

    def encode(self, text, eos_index=EOS_INDEX, unknown_index=UNKNOWN_INDEX):
        text = self.tokenize(text)
        vector = [self.stoi.get(token, unknown_index) for token in text]
        if self.append_eos:
            vector.append(eos_index)
        return torch.LongTensor(vector)

    def decode(self, tensor):
        tokens = [self.itos[index] for index in tensor]
        return ' '.join(tokens)
