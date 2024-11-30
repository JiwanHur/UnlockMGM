# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is heavily inspired by the original implementation from https://github.com/lucidrains/muse-maskgit-pytorch

import copy
import math
from functools import partial
from typing import Callable, Optional
import pdb

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, gumbel_sample, mask_by_random_topk, top_k, gumbel_sample, new_arange

try:
    import xformers.ops as xops

    is_xformers_available = True
except ImportError:
    is_xformers_available = False
    
try:
    from apex.normalization import FusedRMSNorm as RMSNorm  # noqa
except Exception:

    class RMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.variance_epsilon = eps

        def forward(self, input):
            variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
            input = input * torch.rsqrt(variance + self.variance_epsilon)

            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                input = input.to(self.weight.dtype)

            return self.weight * input


# layer norm without bias
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, use_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if use_bias else None
        self.eps = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, encoder_hidden_size=None, attention_dropout=0.0, use_bias=False, return_attn=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.hidden_size} and"
                f" `num_heads`: {self.num_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        kv_hidden_size = self.hidden_size if encoder_hidden_size is None else encoder_hidden_size
        self.key = nn.Linear(kv_hidden_size, self.hidden_size, bias=use_bias)
        self.value = nn.Linear(kv_hidden_size, self.hidden_size, bias=use_bias)

        self.out = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.attention_dropout = attention_dropout

        self.use_memory_efficient_attention_xformers = False
        self.xformers_attention_op = None
        
        self.return_attn = return_attn
    
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        if use_memory_efficient_attention_xformers and not is_xformers_available:
            raise ImportError("Please install xformers to use memory efficient attention")
        self.use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self.xformers_attention_op = attention_op

    def forward(self, hidden_states, valid_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, top_down_feat=None, return_attn=False):
        # if encoder_attention_mask is not None and self.use_memory_efficient_attention_xformers:
        #     raise ValueError("Memory efficient attention does not yet support encoder attention mask")
        
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, q_seq_len, _ = hidden_states.shape
        kv_seq_len = q_seq_len if encoder_hidden_states is None else encoder_hidden_states.shape[1]

        query = self.query(hidden_states)
        key = self.key(context)
        value = self.value(context)
        if top_down_feat is not None:
            value = value + self.value(top_down_feat)
        
        query = query.view(batch, q_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        key = key.view(batch, kv_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        value = value.view(batch, kv_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)

        query, key, value = map(lambda t: t.transpose(1, 2).contiguous(), (query, key, value))  # (B, nh, T, hs)      
           
        if return_attn: # return attn map from prompt to image tokens
            attn_output, out_attn = self.attention(
                query, key, value, 
                attn_mask=encoder_attention_mask, dropout_p=self.attention_dropout if self.training else 0.0)
        else:
            attn_output = F.scaled_dot_product_attention(
                query, key, value, 
                attn_mask=encoder_attention_mask, dropout_p=self.attention_dropout if self.training else 0.0)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, q_seq_len, self.hidden_size)

        attn_output = self.out(attn_output)
        
        if return_attn:
            return attn_output, out_attn
        return attn_output

    def attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        # Efficient implementation equivalent to the following:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        
        attn_weight = torch.softmax(attn_weight, dim=-1)
        out_attn = attn_weight.mean(dim=1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
                
        return attn_weight @ value, out_attn


# Normformer style GLU FeedForward
class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_dropout=0.0,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        use_normformer=True,
        use_bias=False,
    ):
        super().__init__()
        self.use_normformer = use_normformer
        self.pre_mlp_layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps, use_bias=use_bias)
        self.wi_0 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.wi_1 = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        if use_normformer:
            norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
            self.mid_mlp_layer_norm = norm_cls(intermediate_size, eps=layer_norm_eps)
        self.wo = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.pre_mlp_layer_norm(hidden_states)

        hidden_gelu = F.gelu(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        if self.use_normformer:
            hidden_states = self.mid_mlp_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        hidden_states = self.wo(hidden_states)
        return hidden_states

# Default Maskgit MLP
class Mlp(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout=0.0, layer_norm_eps=1e-5, use_bias=False):
        super().__init__()
        self.intermediate_output = nn.Linear(hidden_size, intermediate_size, bias=use_bias)
        self.layer_output = nn.Linear(intermediate_size, hidden_size, bias=use_bias)
        self.layer_norm = LayerNorm(hidden_size, eps=layer_norm_eps, use_bias=use_bias)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        layer_output = F.gelu(self.intermediate_output(hidden_states))
        layer_output = self.layer_output(layer_output)
        layer_output =  self.dropout(layer_output)
        hidden_states = self.layer_norm(layer_output+hidden_states)
        return hidden_states

# PreLN Transformer layer
class TransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        encoder_hidden_size=1024,
        add_cross_attention=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        use_normformer=True,
        use_bias=False,
        use_maskgit_mlp=False,
        return_attn=False,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.use_normformer = use_normformer
        self.return_attn = return_attn

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
        self.attn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)
        self.attention = Attention(
            self.hidden_size, self.num_attention_heads, attention_dropout=attention_dropout, use_bias=use_bias, return_attn=return_attn
        )
        self.use_maskgit_mlp = use_maskgit_mlp
        if use_maskgit_mlp:
            self.ffn = Mlp(self.hidden_size, self.intermediate_size, hidden_dropout, layer_norm_eps, use_bias)
        else:
            self.ffn = FeedForward(
                self.hidden_size,
                self.intermediate_size,
                hidden_dropout,
                norm_type,
                layer_norm_eps,
                use_normformer,
                use_bias
            )
        if use_normformer:
            self.post_attn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)

        if add_cross_attention:
            self.crossattn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)
            self.crossattention = Attention(
                self.hidden_size, self.num_attention_heads, encoder_hidden_size, attention_dropout, use_bias
            )
            if use_normformer:
                self.post_crossattn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)
     
    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None,
                return_hidden_states=False, prev_hidden_states=None, valid_index=None,
                top_down_feat=None, return_attn=False,):
        residual = hidden_states
        
        if not self.use_maskgit_mlp:
            hidden_states = self.attn_layer_norm(hidden_states)
        attention_output = self.attention(hidden_states, valid_mask=valid_index, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, 
                                          top_down_feat=top_down_feat, return_attn=return_attn)
        
        if return_attn:
            attention_output, attn = attention_output  # unpack output
        else:
            attn = None
            
        if self.use_normformer:
            attention_output = self.post_attn_layer_norm(attention_output)
        hidden_states = residual + attention_output
        
        if self.use_maskgit_mlp:
            hidden_states = self.attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        
        if not self.use_maskgit_mlp:            
            hidden_states = residual + hidden_states                
        
        return hidden_states, attn


class Embed(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        hidden_dropout=0.0,
        max_position_embeddings=512,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        use_bias=False,
        layer_norm_embeddings=False,
        use_embeddings_project=False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_embeddings = layer_norm_embeddings
        self.use_embeddings_project = use_embeddings_project

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.embedding_size)
        self.dropout = nn.Dropout(self.hidden_dropout)

        if layer_norm_embeddings:
            norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
            self.embeddings_ln = norm_cls(self.embedding_size, eps=layer_norm_eps)

        if use_embeddings_project:
            self.embedding_hidden_mapping = nn.Linear(self.embedding_size, self.hidden_size, bias=use_bias)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.shape
        
        word_embeddings = self.word_embeddings(input_ids)
        position_ids = torch.arange(seq_length)[None, :].to(input_ids.device)
        position_embeddings = self.position_embeddings(position_ids)
        input_embeddings = word_embeddings + position_embeddings
        if self.layer_norm_embeddings:
            input_embeddings = self.embeddings_ln(input_embeddings)
        if self.use_embeddings_project:
            input_embeddings = self.embedding_hidden_mapping(input_embeddings)

        input_embeddings = self.dropout(input_embeddings)
        return input_embeddings


class MlmLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        norm_type="layernorm",
        layer_norm_eps=1e-5,
        use_mlm_layernorm=True,
        use_bias=False,
        add_dropout=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_mlm_layernorm = use_mlm_layernorm
        self.add_dropout = add_dropout
        self.mlm_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        if use_mlm_layernorm:
            norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
            self.mlm_ln = norm_cls(self.hidden_size, eps=layer_norm_eps)
        if self.add_dropout:
            self.dropout = nn.Dropout(0.1)  # hardcode haha
        self.to_logits = nn.Linear(self.hidden_size, vocab_size, bias=use_bias)

    def forward(self, hidden_states):
        hidden_states = self.mlm_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        if self.use_mlm_layernorm:
            hidden_states = self.mlm_ln(hidden_states)
        if self.add_dropout:
            hidden_states = self.dropout(hidden_states)
        logits = self.to_logits(hidden_states)
        return logits
    
class TokenGenerator(nn.Module):
    def __init__(
        self,
        n_token: int, # Number of token (S)
        n_class: int, # Number of class (C)
        n_factor: int, # Number of factors (F)
        d_embed: int, # Embed dimension (P)
        d_token: int, # Token dimension (D)
    ):
        super().__init__()
        self.n_token = n_token
        self.n_class = n_class
        self.n_factor = n_factor
        self.d_embed = d_embed
        self.d_token = d_token
        
        self.mlp_p = nn.Embedding(self.n_token, self.d_embed * self.n_factor)
        self.mlp_c = nn.Embedding(self.n_class, self.d_embed * self.n_factor)
        self.mlp_f = nn.Embedding(1, self.n_factor)
        self.mlp_t = nn.Linear(self.d_embed, self.d_token)
        self.layer_norm = nn.LayerNorm(self.d_token)

    def forward(self, cls_ids):
        cls_ids = cls_ids - 1024 # vq codebook size 1024
        pos_ids = torch.arange(self.n_token).to(cls_ids.device)
        factor_ids = torch.arange(1).to(cls_ids.device)
        pos_embed = self.mlp_p(pos_ids).view(1, self.n_token, self.d_embed, self.n_factor) # 1 x S x P x F
        cls_embed = self.mlp_c(cls_ids).view(-1, 1, self.d_embed, self.n_factor) # B x 1 x P x F
        fac_embed = self.mlp_f(factor_ids).view(1, 1, 1, self.n_factor) # 1 x 1 x 1 x F
        embed = (fac_embed * (pos_embed + cls_embed)).sum(-1)
        
        return self.mlp_t(self.layer_norm(embed)) # B x S x P

class Decode_Block(nn.Module):
    def __init__(self, inplanes):
        super().__init__()
        self.linear = nn.Linear(inplanes, inplanes, bias=False)
        self.linear2 = nn.Linear(inplanes, inplanes, bias=False)

    def forward(self, x):
        x = self.linear(x)
        out = self.linear2(x)
        # out = x
        return x, out
    
class MaskGitTransformerTOAST(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        vocab_size,  # codebook_size + 1 (for the mask token), for class-conditioned generation it'll be codebook_size + num_classes + 1
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=256,  # for clas-conditioned generation it'll be 256 + 1 (for the class token)
        num_vq_tokens=256,
        add_cross_attention=False,
        encoder_hidden_size=1024,  # T5-large
        project_encoder_hidden_states=False,
        initializer_range=0.02,
        norm_type="layernorm",  # or rmsnorm
        layer_norm_eps=1e-5,
        use_normformer=True,
        use_encoder_layernorm=True,
        use_mlm_layer=True,
        use_mlm_layernorm=True,
        use_bias=False,
        layer_norm_embeddings=False,
        use_maskgit_mlp=False,
        codebook_size=1024,
        num_classes=None,  # set for class-conditioned generation
        tg_factor=16,
        use_toast=False,
        train_head=False,
        use_blank_second=False,
        predict_all=True,
        lambda_var=0.1,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.register_to_config(mask_token_id=vocab_size - 1)
        self.tg_factor = tg_factor
        self.use_toast = use_toast
        self.train_head = train_head
        self.use_blank_second = use_blank_second
        self.predict_all = predict_all
        self.lambda_var = lambda_var
        
        self.output_size = self.vocab_size
        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm

        self.embed = Embed(
            self.vocab_size,
            self.hidden_size,
            self.hidden_size,
            self.hidden_dropout,
            self.max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            use_bias=use_bias,
            layer_norm_embeddings=layer_norm_embeddings,
            norm_type=norm_type,
        )

        if add_cross_attention is not None and project_encoder_hidden_states:  # Cross attention
            self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size, bias=use_bias)
            self.encoder_proj_layer_norm = norm_type(hidden_size, eps=layer_norm_eps)
            
        self.transformer_layers = nn.ModuleList(
            [
                TransformerLayer(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    num_attention_heads=self.num_attention_heads,
                    encoder_hidden_size=encoder_hidden_size,
                    add_cross_attention=add_cross_attention,
                    hidden_dropout=self.hidden_dropout,
                    attention_dropout=self.attention_dropout,
                    norm_type=norm_type,
                    layer_norm_eps=layer_norm_eps,
                    use_normformer=use_normformer,
                    use_bias=use_bias,
                    use_maskgit_mlp=use_maskgit_mlp
                )
                for l in range(self.num_hidden_layers)
            ]
        )
        
        if use_encoder_layernorm:
            self.encoder_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)

        if use_mlm_layer:
            self.mlm_layer = MlmLayer(
                self.hidden_size, self.vocab_size, norm_type, layer_norm_eps, use_mlm_layernorm, use_bias
            )
        else:
            self.to_logits = nn.Linear(self.hidden_size, self.vocab_size, bias=use_bias)
            
        if self.use_toast:
            self.cls_anchor_gen = TokenGenerator(
                1, num_classes, tg_factor, hidden_size, hidden_size
            )
            self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(self.hidden_size), requires_grad=True)
            self.decoders = nn.ModuleList([Decode_Block(hidden_size) for _ in range(self.num_hidden_layers)])
            # Original TOAST update classifier head for transfer learning but we do not
                    
        self.gradient_checkpointing = False

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights according to the original implementation.
        https://github.com/google-research/maskgit/blob/main/maskgit/nets/maskgit_transformer.py#L37
        """
        # TODO: make this configurable
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        cond_dropout_prob=0.0,
        return_hidden_states=False,
        prev_hidden_states=None,
        image_tokens=None,
        predict_mask_token=False,
        force_no_toast=False,
        return_attn=False,
        **kwargs,
    ):
        if self.config.add_cross_attention and encoder_hidden_states is None:
            raise ValueError("If `add_cross_attention` is True, `encoder_hidden_states` should be provided.")
        
        hidden_stack = []        
        hidden_states = self.embed(input_ids)
        hidden_stack.append(hidden_states)

        for l, layer in enumerate(self.transformer_layers):
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(layer), hidden_states, encoder_hidden_states, encoder_attention_mask
                )
            else:
                valid_index = input_ids.ne(self.config.mask_token_id)
                                
                hidden_states, attn = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    prev_hidden_states=prev_hidden_states,
                    valid_index=valid_index,
                    return_attn=return_attn,
                )
                hidden_stack.append(hidden_states)
         
        out_hidden_states = hidden_states   
        if not force_no_toast and self.use_toast:
            # print('use toast')
            in_var = []
            out_var = []

            cls_hidden = hidden_states[:,0]
            x = hidden_states
            cls_anchor = self.cls_anchor_gen(input_ids[:,0])
            cos_sim = F.normalize(x, dim=-1) @ F.normalize(cls_anchor, dim=-1).transpose(1,2)
            mask = cos_sim.clamp(0, 1)
            x = x * mask
            x = x @ self.top_down_transform
            
            td = []
            for depth in range(len(self.decoders) - 1, -1, -1):
                x, out = self.decoders[depth](x)
                td = [out] + td            
            if self.use_blank_second:
                input_ids = copy.deepcopy(input_ids)
                input_ids.fill_(self.config.mask_token_id)
                if labels is not None and self.predict_all:
                    labels[:,1:] = image_tokens
            hidden_states = self.embed(input_ids)
            
            for l, layer in enumerate(self.transformer_layers):
                valid_index = input_ids.ne(self.config.mask_token_id)
                                
                in_var.append(hidden_states)
                hidden_states, attn = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    prev_hidden_states=prev_hidden_states,
                    valid_index=valid_index,
                    top_down_feat = td[l],
                )
                out_var.append(hidden_states)
        if self.config.use_encoder_layernorm:
            hidden_states = self.encoder_layer_norm(hidden_states)

        if self.config.use_mlm_layer:
            logits = self.mlm_layer(hidden_states)
        else:
            logits = self.to_logits(hidden_states)
            
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.output_size), labels.view(-1), ignore_index=-100, label_smoothing=label_smoothing
            )
            # calculate auxiliary loss
            if self.lambda_var > 0:
                loss_var = self.var_loss(in_var, out_var)
            else:
                loss_var = None
                
            # return logits, loss, loss_correct_mask
            return logits, loss, loss_var
        
        # only for inference
        if return_attn and return_hidden_states:
            return logits, out_hidden_states, attn
        if return_attn:
            return logits, attn
        if return_hidden_states: 
            return logits, out_hidden_states
        return logits
    
    def var_loss(self, in_var, out_var):
        recon_loss = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            recon, out = self.decoders[depth](out_var[depth].detach())
            target = in_var[depth].detach()
            recon_loss.append(F.mse_loss(recon, target))

        return self.lambda_var * sum(recon_loss)

    def forward_with_hidden_toast(
        self,
        input_ids,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        cond_dropout_prob=0.0,
        return_hidden_states=False,
        prev_hidden_states=None,
        image_tokens=None,
        predict_mask_token=False,
        return_attn=False,
        **kwargs,
    ):
        if self.config.add_cross_attention and encoder_hidden_states is None:
            raise ValueError("If `add_cross_attention` is True, `encoder_hidden_states` should be provided.")
        
        out_hidden_states = prev_hidden_states

        in_var = []
        out_var = []
        cls_hidden = out_hidden_states[:,0]
        x = out_hidden_states
        cls_anchor = self.cls_anchor_gen(input_ids[:,0])
        cos_sim = F.normalize(x, dim=-1) @ F.normalize(cls_anchor, dim=-1).transpose(1,2)
        mask = cos_sim.clamp(0, 1)
        x = x * mask
        x = x @ self.top_down_transform
        
        td = []
        for depth in range(len(self.decoders) - 1, -1, -1):
            x, out = self.decoders[depth](x)
            td = [out] + td            
        if self.use_blank_second:
            input_ids = copy.deepcopy(input_ids)
            input_ids.fill_(self.config.mask_token_id)
            if labels is not None and self.predict_all:
                labels[:,1:] = image_tokens
        hidden_states = self.embed(input_ids)
        
        for l, layer in enumerate(self.transformer_layers):
            valid_index = input_ids.ne(self.config.mask_token_id)                
            
            # print(hidden_states.shape, td[l].shape)
            hidden_states, attn = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                prev_hidden_states=prev_hidden_states,
                valid_index=valid_index,
                top_down_feat = td[l],
                return_attn=return_attn,
            )

        if self.config.use_encoder_layernorm:
            hidden_states = self.encoder_layer_norm(hidden_states)

        if self.config.use_mlm_layer:
            logits = self.mlm_layer(hidden_states)
        else:
            logits = self.to_logits(hidden_states)
            
        if return_attn:
            return logits, attn
        return logits, out_hidden_states

    def gen_corr_loss_toast(
        self,
        input_ids,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        cond_dropout_prob=0.0,
        substitution_rate=0.3,
        previous_logits=None,
        use_blank_token_for_corr=True,
        use_dynamic_substitution=False,
        num_substitute=None,
        gt_image_tokens=None,
        **kwargs,
    ):
        # generating correction loss based argmax/multinomial sampling
        # generating correction loss based on the CLMLC 
        # https://github.com/layer6ai-labs/CMLMC/blob/master/fairseq/models/nat/cmlm_transformer.py
        
        if self.config.add_cross_attention and encoder_hidden_states is None:
            raise ValueError("If `add_cross_attention` is True, `encoder_hidden_states` should be provided.")

        # gen masked input
        batch_size = input_ids.shape[0]
        seq_len = self.config.num_vq_tokens
        shape = (batch_size, seq_len)
        class_ids = input_ids[:,:1]
        image_tokens = input_ids[:,1:]
        
        with torch.no_grad():
            if use_blank_token_for_corr:
                # initialize with all image tokens masked, no labels so that the loss is not calculated
                masked_input = torch.ones(shape, dtype=torch.long, device=self.device) * self.config.mask_token_id
                masked_input = torch.cat([class_ids, masked_input], dim=1)  # prepend class ids to input_ids
                previous_output_logits = self(input_ids=masked_input, label_smoothing=label_smoothing, force_no_toast=True)
                
                # detach class_ids for sampling, sampling with multinomial distribution
                previous_output_logits = torch.exp(previous_output_logits[:, 1:, : self.config.codebook_size].log_softmax(dim=-1))
                previous_prob, _ = previous_output_logits.max(-1)
                previous_sample_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in previous_output_logits])
                
            else: # use previous step's output for correction            
                # detach class_ids for sampling, sampling with multinomial distribution
                previous_logits = torch.exp(previous_logits[:, 1:, : self.config.codebook_size].log_softmax(dim=-1))
                previous_prob, _ = previous_logits.max(-1)
                previous_sample_ids = torch.multinomial(previous_logits, 1)[:, 0].view(*previous_logits.shape[:-1])
                            
            # sample substitue mask with probability substitution_rate
            existing_mask = torch.eq(image_tokens, self.config.mask_token_id)
            if not use_dynamic_substitution:
                num_unmasked = (~existing_mask).sum(dim=1)
                num_substitute = (num_unmasked * substitution_rate).round().clamp(min=1)
            
            # can replace masked_scores with previous_prob so taht the least probable tokens are replaced
            previous_prob.uniform_()
            previous_prob.masked_fill_(existing_mask, self.config.mask_token_id)
            
            _, replace_rank = previous_prob.sort(-1)
            substitute_cutoff = new_arange(replace_rank) < num_substitute[:, None].long()
            substitute_mask = substitute_cutoff.scatter(1, replace_rank, substitute_cutoff)
                        
            substituted_output = torch.where(substitute_mask, previous_sample_ids, image_tokens)
            substituted_output = torch.cat([class_ids, substituted_output], dim=1) # re-attach class token_ids
            final_labels = torch.where(substitute_mask, image_tokens, -100)
            labels_mask = torch.ones_like(class_ids, device=class_ids.device).fill_(-100)
            final_labels = torch.cat([labels_mask, final_labels], dim=1) # re-attach class token_ids
            
                
        final_output, loss_corr, loss_correct_mask = self(input_ids=substituted_output, labels=final_labels, 
                             label_smoothing=label_smoothing, image_tokens=gt_image_tokens
        )
        
        return final_output, loss_corr, loss_correct_mask

    def generate2(
        self,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        temperature=1.0,
        temperature2=0.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        return_intermediate=False,
        **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        seq_len = self.config.num_vq_tokens

        batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
        shape = (batch_size, seq_len)

        if return_intermediate:
            intermediate = []
            mask_index = []
            
        # shift the class ids by the codebook size
        if class_ids is not None:
            class_ids += self.config.codebook_size

        # initialize with all image tokens masked
        input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id

        for step in range(timesteps):
            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            logits = self(input_ids, encoder_hidden_states=encoder_hidden_states, force_no_toast=True)
            logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])
            
            # Just updates the masked tokens.
            unknown_map = input_ids == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
            
            if return_intermediate:
                intermediate.append(sampled_ids)
                
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            probs = logits.softmax(dim=-1)
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )

            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio) + temperature2
            masking = mask_by_random_topk(mask_len, selected_probs, temperature)
            if return_intermediate:
                mask_index.append(masking)
                
            # Masks tokens with lower confidence.
            input_ids = torch.where(masking, mask_token_id, sampled_ids)

        if return_intermediate:
            return sampled_ids, intermediate, mask_index
        return sampled_ids
        
    def generate_sg(
        self,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        return_intermediate=False,
        guidance_anneal=False,
        **kwargs,
    ):
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        seq_len = self.config.num_vq_tokens

        batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
        shape = (batch_size, seq_len)

        if return_intermediate:
            intermediate = []
            mask_index = []
            
        # shift the class ids by the codebook size
        if class_ids is not None:
            class_ids += self.config.codebook_size

        # initialize with all image tokens masked
        input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id

        hidden_states = None
        for step in range(timesteps):
            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            logits, hidden_states = self(input_ids, encoder_hidden_states=encoder_hidden_states, force_no_toast=True, return_hidden_states=True)
            logits = logits[..., : self.config.codebook_size]
                
            if guidance_scale != 0: # guidance with prev state
                # TOAST refining
                toast_logits, _ = self.forward_with_hidden_toast(input_ids, prev_hidden_states=hidden_states, 
                                                                 force_no_first=True)
                toast_logits = toast_logits[..., : self.config.codebook_size]
                logits = toast_logits + (1 + guidance_scale) * (logits - toast_logits)

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])
            
            # Just updates the masked tokens.
            unknown_map = input_ids == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
            
            if return_intermediate:
                intermediate.append(sampled_ids)
                
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            probs = logits.softmax(dim=-1)
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )

            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature)
            if return_intermediate:
                mask_index.append(masking)
                
            # Masks tokens with lower confidence.
            input_ids = torch.where(masking, mask_token_id, sampled_ids)

        if return_intermediate:
            return sampled_ids, intermediate, mask_index
        return sampled_ids