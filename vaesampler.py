import torch
from data import MonoTextData
from modules import VAE, LSTMEncoder, LSTMDecoder
from torch import nn, optim

from bpemb import BPEmb

import re

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):   
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

class VAESampler:

    def __init__(self, decode_from, params, cuda=False):
        self.decode_from = decode_from
        self.params = params
        params.enc_nh = params.dec_nh # not sure why this is necessary...

        self.train_data = MonoTextData(params.train_data, label=False)
        self.vocab = self.train_data.vocab
        self.vocab_size = len(self.vocab)

        # do I need these?
        model_init = uniform_initializer(0.01)
        emb_init = uniform_initializer(0.1)

        params.device = self.device = torch.device("cuda" if cuda else "cpu")

        self.encoder = LSTMEncoder(params, self.vocab_size, model_init,
                emb_init)
        self.decoder = LSTMDecoder(params, self.vocab, model_init, emb_init)

        self.vae = VAE(self.encoder, self.decoder, params).to(params.device)

        # assuming models were trained on a gpu...
        if cuda:
            self.vae.load_state_dict(torch.load(self.decode_from))
        else:
            self.vae.load_state_dict(torch.load(self.decode_from,
                map_location='cpu'))

    def to_s(self, decoded):
        return [' '.join(item) for item in decoded]

    def beam(self, z, K=5):
        decoded_batch = self.vae.decoder.beam_search_decode(z, K)
        return self.to_s(decoded_batch)

    def sample(self, z, temperature=1.0):
        decoded_batch = self.vae.decoder.sample_decode(z, temperature)
        return self.to_s(decoded_batch)

    def greedy(self, z):
        decoded_batch = self.vae.decoder.greedy_decode(z)
        return self.to_s(decoded_batch)

    def str2ids(self, s):
        "encode string s as list of word ids"
        raise NotImplemented

    def encode(self, t):
        """
        Returns (z, mu, log_var) from encoder given list of strings.

        z is a sample from gaussian specified with (mu, log_var)
        """
        str_ids = []
        for s in t:
            ids = self.str2ids(s)
            str_ids.append(ids)
        tensor = self.train_data._to_tensor(str_ids, True, self.device)[0]
        z, (mu, log_var) = self.vae.encoder.sample(tensor, 1)
        return z, mu, log_var

    def z(self, t):
        "return sampled latent zs for list of strings t"
        z, mu, logvar = self.encode(t)
        return z.squeeze(1)

    def mu(self, t):
        "return mean of latent gaussian for list of strings t"
        z, mu, logvar = self.encode(t)
        return mu.squeeze(1)


class BPEmbVaeSampler(VAESampler):

    def __init__(self, lang, vs, dim, decode_from, params, cuda=False):
        self.bp = BPEmb(lang=lang, vs=vs, dim=dim)
        super().__init__(decode_from, params, cuda)

    def to_s(self, decoded):
        out = []
        for item in decoded:
            s = self.bp.decode(item).replace('▁', ' ').strip()
            s = s[0].upper() + s[1:]
            s = re.sub(r'\bi\b', 'I', s)
            s = re.sub(r'[.!?]\s+(\w)',
                    lambda m: m.group()[:-1] + m.group()[-1].upper(),
                    s)
            out.append(s)
        return out

    def str2ids(self, s):
        """
        Encode string s with BPEmb. BPEmb has a fixed vocabulary size, but
        the model only has outputs for vocab items that are used in the
        training data, so this function replaces any BPEmb ids *not* in the
        training vocabulary with the model's "unknown" id.
        """
        encoded = self.bp.encode(s)
        ids = [self.vocab.word2id.get(item, self.vocab.unk_id) \
                for item in encoded]
        return ids
