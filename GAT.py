#!/usr/bin/python

import jax
import flax
import flax.linen as nn
import jax.numpy as jnp

import jVMC.global_defs as global_defs
from jVMC.utils import HashableArray

from functools import partial
from typing import Sequence

from jVMC.nets.initializers import init_fn_args

class GAT(nn.Module):
    
    nbr_idx: HashableArray
    F: Sequence[int] = (32, 32, 32, 32, 32, 32)
    num_heads: int = 8
    enlyr: int = 3
    use_bias: bool = True
    
    def nodeEncoder(self, s):
        sp = jnp.zeros_like(s)
        sp = sp.at[::2, ::2].set(1)
        sp = sp.at[1::2, 1::2].set(1)
        sp = jnp.ravel(sp)
        sp = sp * 2 - 1
        
        s = jnp.ravel(s)
        s = s * 2 - 1
        node_fea = jnp.stack((s, sp), axis=1)
        return node_fea
    
    def setup(self):
        
        init_args = init_fn_args(dtype=global_defs.tReal, kernel_init=jax.nn.initializers.he_uniform(), use_bias=self.use_bias)
        
        self.encoders = [nn.Dense(self.F[0], **init_args) for _ in range(3)]
        self.enlns = [nn.LayerNorm(param_dtype=global_defs.tReal, dtype=global_defs.tReal) for _ in range(self.enlyr)]
        
        self.GATLayers = [GATLayer(f, self.num_heads, self.nbr_idx) for f in self.F]
        self.ln = [nn.LayerNorm(param_dtype=global_defs.tReal, dtype=global_defs.tReal) for _ in self.F]
        
        self.decoder_re = nn.Dense(1, param_dtype=global_defs.tReal, dtype=global_defs.tReal, use_bias=True, kernel_init=jax.nn.initializers.normal(stddev=2e-4), bias_init=jax.nn.initializers.ones)
        self.decoder_re2 = nn.Dense(self.F[-1], **init_args)
        self.ln_re = nn.LayerNorm(param_dtype=global_defs.tReal, dtype=global_defs.tReal)
        
        self.decoder_im = nn.Dense(1, param_dtype=global_defs.tReal, dtype=global_defs.tReal, use_bias=self.use_bias, kernel_init=jax.nn.initializers.normal(dtype=global_defs.tReal))
        self.decoder_im2 = nn.Dense(self.F[-1], **init_args)
        self.ln_im = nn.LayerNorm(param_dtype=global_defs.tReal, dtype=global_defs.tReal)
        
    def __call__(self, s):
        node_feats = self.nodeEncoder(s)
        for (encoder, enln) in zip(self.encoders, self.enlns):
            node_feats = encoder(node_feats)
            node_feats = nn.relu(node_feats)
            node_feats = enln(node_feats)

        init_feats = node_feats
        for (GATLayer, ln) in zip(self.GATLayers, self.ln):
            res = node_feats
            node_feats = GATLayer(node_feats)
            node_feats = nn.relu(node_feats)
            node_feats = res + node_feats
            node_feats = ln(node_feats)

        node_feats_re = self.ln_re(nn.relu(self.decoder_re2(node_feats)))
        node_feats_re = self.decoder_re(node_feats_re)
        node_feats_re = jnp.sum(jnp.log(jnp.abs(node_feats_re)))
        
        node_feats_im = self.ln_im(nn.relu(self.decoder_im2(node_feats)))
        node_feats_im = jnp.mod(self.decoder_im(node_feats_im).sum(), 2*jnp.pi)
        
        return node_feats_re + 1.j * node_feats_im

class GATLayer(nn.Module):
    c_out : int  
    num_heads : int  
    nbr_idx: HashableArray
    concat_heads : bool = True  
    alpha : float = 0.2  
    
    def setup(self):
        if self.concat_heads:
            assert self.c_out % self.num_heads == 0, "Number of output features must be a multiple of the count of heads."
            c_out_per_head = self.c_out // self.num_heads
        else:
            c_out_per_head = self.c_out

        self.projection = nn.Dense(c_out_per_head * self.num_heads,
                                   kernel_init=nn.initializers.glorot_uniform(dtype=global_defs.tReal),
                                   param_dtype=global_defs.tReal,
                                   dtype=global_defs.tReal)
        
        self.a = self.param('a',
                            nn.initializers.glorot_uniform(dtype=global_defs.tReal),
                            (self.num_heads, 2 * c_out_per_head), global_defs.tReal)  # One per head
        
    def __call__(self, node_feats):
        num_nodes, num_neighbors = self.nbr_idx.shape
        nbr_idx = self.nbr_idx.wrapped
        node_feats = self.projection(node_feats)
        node_feats = node_feats.reshape((num_nodes, self.num_heads, -1))
        logit_parent = (node_feats * self.a[None, :, :self.a.shape[1]//2]).sum(axis=-1)
        logit_child = (node_feats * self.a[None, :, self.a.shape[1]//2:]).sum(axis=-1)
        attn_logits = logit_parent[:, None, :] + logit_child[nbr_idx]
        attn_logits = nn.leaky_relu(attn_logits, self.alpha)
        
        attn_probs = jax.nn.softmax(attn_logits, axis=1)
        node_nbr_fea = node_feats[nbr_idx]
        node_feats = jnp.einsum("NMh,NMhc->Nhc", attn_probs, node_nbr_fea)
        
        if self.concat_heads:
            node_feats = node_feats.reshape(num_nodes, -1)
        else:
            node_feats = node_feats.mean(axis=1)

        return node_feats