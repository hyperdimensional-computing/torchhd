import torch
import torch.nn as nn
from torchhd import embeddings
import torchhd


class Encoder(nn.Module):
    def __init__(self, size, dimensions, encoding, name):
        super(Encoder, self).__init__()
        self.encoding = encoding
        if self.encoding == "bundle":
            if name == "EuropeanLanguages":
                self.embed = embeddings.Random(size, dimensions, padding_idx=0)
            else:
                self.embed = embeddings.Random(size, dimensions)
        if self.encoding == "hashmap":
            levels = 100
            if name == "EuropeanLanguages":
                self.keys = embeddings.Random(size, dimensions, padding_idx=0)
            else:
                self.keys = embeddings.Random(size, dimensions)
            self.embed = embeddings.Level(levels, dimensions)
        if self.encoding == "ngram":
            if name == "EuropeanLanguages":
                self.embed = embeddings.Random(size, dimensions, padding_idx=0)
            else:
                self.embed = embeddings.Random(size, dimensions)
        if self.encoding == "sequence":
            if name == "EuropeanLanguages":
                self.embed = embeddings.Random(size, dimensions, padding_idx=0)
            else:
                self.embed = embeddings.Random(size, dimensions)
        if self.encoding == "random":
            self.embed = embeddings.Projection(size, dimensions)
        if self.encoding == "sinusoid":
            self.embed = embeddings.Sinusoid(size, dimensions)
        if self.encoding == "density":
            self.embed = embeddings.Density(size, dimensions)
        if self.encoding == "flocet":
            self.embed = embeddings.DensityFlocet(size, dimensions)
        if self.encoding == "generic":
            levels = 100
            self.keys = embeddings.Random(size, dimensions)
            self.embed = embeddings.Level(levels, dimensions)
        if self.encoding == "fractional":
            self.fractional = torchhd.functional.FractionalPowerEncoding(
                dimensions, size, "sinc", 1.0, "FHRR"
            )
        self.flatten = torch.nn.Flatten()

    def forward(self, x, device=None):
        x = self.flatten(x).float()
        if self.encoding == "bundle":
            sample_hv = torchhd.multiset(self.embed(x.long()))
        if self.encoding == "hashmap":
            sample_hv = torchhd.hash_table(self.keys.weight, self.embed(x))
        if self.encoding == "ngram":
            sample_hv = torchhd.ngrams(self.embed(x.long()), n=3)
        if self.encoding == "sequence":
            sample_hv = torchhd.ngrams(self.embed(x.long()), n=1)
        if self.encoding == "random":
            sample_hv = self.embed(x).sign()
        if self.encoding == "sinusoid":
            sample_hv = self.embed(x).sign()
        if self.encoding == "density":
            sample_hv = self.embed(x).sign()
        if self.encoding == "flocet":
            sample_hv = self.embed(x).sign()
        if self.encoding == "generic":
            sample_hv = torchhd.functional.generic(self.keys.weight, self.embed(x), 3)
        if self.encoding == "fractional":
            sample_hv = self.fractional.encoding(x, device)
            return sample_hv
        return torchhd.hard_quantize(sample_hv)
