import torch
import torch.nn as nn

import numpy as np

""" Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                if self.kwargs['normalize']:
                    embed_fns.append(lambda x, p_fn=p_fn,
                                     freq=freq: p_fn(x * freq) / freq)
                else:
                    embed_fns.append(lambda x, p_fn=p_fn,
                                            freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(multires, normalize=False, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'normalize': normalize,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


def positional_encoding(input,L): # [B,...,N]
    shape = input.shape
    freq = 2**torch.arange(L,dtype=torch.float32)*np.pi # [L]
    spectrum = input[...,None]*freq # [B,...,N,L]
    sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
    input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
    input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
    return input_enc


def positional_encoding_c2f(input,L, emb_c2f, alpha_ratio = -1): # [B,...,N]
    input_enc = positional_encoding(input,L=L) # [B,...,2NL]
    # coarse-to-fine: smoothly mask positional encoding for BARF
    weight = None
    if emb_c2f is not None:
        # set weights for different frequency bands
        start,end = emb_c2f
        alpha = (alpha_ratio-start)/(end-start)*L
        k = torch.arange(L,dtype=torch.float32) + 1
        weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
        # apply weights
        shape = input_enc.shape
        input_enc = (input_enc.view(-1,L)*weight).view(*shape)
    # input_enc = torch.zeros_like(input_enc)
    input_enc = torch.cat([input,input_enc],dim=-1) # [B,...,6L+3]
    return input_enc, weight