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
from .sampling import cosine_schedule, gumbel_sample, mask_by_random_topk, top_k

try:
    import xformers.ops as xops

    is_xformers_available = True
except ImportError:
    is_xformers_available = False


# classifier free guidance functions


def uniform(shape, min=0, max=1, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


def make_attention_mask(
    query_input: torch.Tensor, key_input: torch.Tensor, pairwise_fn: Callable = torch.mul
) -> torch.Tensor:
    # [batch, len_q, len_kv]
    mask = pairwise_fn(
        # [batch, len_q] -> [batch, len_q, 1]
        torch.unsqueeze(query_input, axis=-1),
        # [batch, len_q] -> [batch, 1, len_kv]
        torch.unsqueeze(key_input, axis=-2),
    )
    # [batch, 1, len_q, len_kv]. This creates the head dim.
    mask = torch.unsqueeze(mask, axis=-3)
    return (1.0 - mask).type(torch.bool)


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
    def __init__(self, hidden_size, num_heads, encoder_hidden_size=None, attention_dropout=0.0, use_bias=False):
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
        self.dropout = nn.Dropout(attention_dropout)

        self.use_memory_efficient_attention_xformers = False
        self.xformers_attention_op = None

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        if use_memory_efficient_attention_xformers and not is_xformers_available:
            raise ImportError("Please install xformers to use memory efficient attention")
        self.use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
        self.xformers_attention_op = attention_op

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None):
        if encoder_attention_mask is not None and self.use_memory_efficient_attention_xformers:
            raise ValueError("Memory efficient attention does not yet support encoder attention mask")

        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch, q_seq_len, _ = hidden_states.shape
        kv_seq_len = q_seq_len if encoder_hidden_states is None else encoder_hidden_states.shape[1]

        query = self.query(hidden_states)
        key = self.key(context)
        value = self.value(context)

        query = query.view(batch, q_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        key = key.view(batch, kv_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)
        value = value.view(batch, kv_seq_len, self.num_heads, self.head_dim)  # (B, T, nh, hs)

        if self.use_memory_efficient_attention_xformers:
            attn_output = xops.memory_efficient_attention(query, key, value)
            attn_output = attn_output.view(batch, q_seq_len, self.hidden_size)
        else:
            attention_mask = None
            if encoder_attention_mask is not None:
                src_attn_mask = torch.ones(batch, q_seq_len, dtype=torch.long, device=query.device)
                attention_mask = make_attention_mask(src_attn_mask, encoder_attention_mask, dtype=query.dtype)
            attn_output = self.attention(query, key, value, attention_mask)

        attn_output = self.out(attn_output)
        return attn_output

    def attention(self, query, key, value, attention_mask=None):
        batch, seq_len = query.shape[:2]
        kv_seq_len = key.shape[1]
        query, key, value = map(lambda t: t.transpose(1, 2).contiguous(), (query, key, value))  # (B, nh, T, hs)

        attn_weights = torch.baddbmm(
            input=torch.zeros(batch * self.num_heads, seq_len, kv_seq_len, dtype=query.dtype, device=query.device),
            batch1=query.view(batch * self.num_heads, seq_len, self.head_dim),
            batch2=key.view(batch * self.num_heads, kv_seq_len, self.head_dim).transpose(1, 2),
            alpha=1 / self.scale_attn,
        )
        attn_weights = attn_weights.view(batch, self.num_heads, seq_len, kv_seq_len)  # -1 is kv_seq_len
        # Apply the attention mask
        if attention_mask is not None:
            attn_weights = torch.masked_fill(attn_weights, attention_mask, torch.finfo(query.dtype).min)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)
        return attn_output


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
        use_maskgit_mlp=False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.use_normformer = use_normformer

        norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
        self.attn_layer_norm = norm_cls(self.hidden_size, eps=layer_norm_eps)
        self.attention = Attention(
            self.hidden_size, self.num_attention_heads, attention_dropout=attention_dropout, use_bias=use_bias
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
                use_bias,
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

    def forward(self, hidden_states, encoder_hidden_states=None, encoder_attention_mask=None):
        residual = hidden_states
        if not self.use_maskgit_mlp:
            hidden_states = self.attn_layer_norm(hidden_states)
        attention_output = self.attention(hidden_states)
        if self.use_normformer:
            attention_output = self.post_attn_layer_norm(attention_output)
        hidden_states = residual + attention_output
        if self.use_maskgit_mlp:
            hidden_states = self.attn_layer_norm(hidden_states)


        if encoder_hidden_states is not None:
            residual = hidden_states
            # TODO: should norm be applied to encoder_hidden_states as well?
            hidden_states = self.crossattn_layer_norm(hidden_states)
            attention_output = self.crossattention(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
            if self.use_normformer:
                attention_output = self.post_crossattn_layer_norm(attention_output)
            hidden_states = residual + attention_output

        residual = hidden_states
        hidden_states = self.ffn(hidden_states)
        if self.use_maskgit_mlp:
            return hidden_states
        hidden_states = residual + hidden_states
        return hidden_states


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
        seq_length = input_ids.shape[-1]
        position_ids = torch.arange(seq_length)[None, :].to(input_ids.device)
        word_embeddings = self.word_embeddings(input_ids)
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_mlm_layernorm = use_mlm_layernorm
        self.mlm_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        if use_mlm_layernorm:
            norm_cls = partial(LayerNorm, use_bias=use_bias) if norm_type == "layernorm" else RMSNorm
            self.mlm_ln = norm_cls(self.hidden_size, eps=layer_norm_eps)
        self.to_logits = nn.Linear(self.hidden_size, vocab_size, bias=use_bias)

    def forward(self, hidden_states):
        hidden_states = self.mlm_dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        if self.use_mlm_layernorm:
            hidden_states = self.mlm_ln(hidden_states)
        logits = self.to_logits(hidden_states)
        return logits


class MaskGitTransformer(ModelMixin, ConfigMixin):
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
        num_vq_tokens=256,
        num_classes=None,  # set for class-conditioned generation
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
            norm_type=norm_type
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
                    use_maskgit_mlp=use_maskgit_mlp,
                )
                for _ in range(self.num_hidden_layers)
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
    ):
        if self.config.add_cross_attention and encoder_hidden_states is None:
            raise ValueError("If `add_cross_attention` is True, `encoder_hidden_states` should be provided.")

        hidden_states = self.embed(input_ids)

        if encoder_hidden_states is not None and self.config.project_encoder_hidden_states:
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
            encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)

        # condition dropout for classifier free guidance
        if encoder_hidden_states is not None and self.training and cond_dropout_prob > 0.0:
            batch_size = encoder_hidden_states.shape[0]
            mask = prob_mask_like((batch_size, 1, 1), 1.0 - cond_dropout_prob, encoder_hidden_states.device)
            encoder_hidden_states = encoder_hidden_states * mask
        for layer in self.transformer_layers:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(layer), hidden_states, encoder_hidden_states, encoder_attention_mask
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
        if self.config.use_encoder_layernorm:
            hidden_states = self.encoder_layer_norm(hidden_states)

        if self.config.use_mlm_layer:
            logits = self.mlm_layer(hidden_states)
        else:
            logits = self.to_logits(hidden_states)

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100, label_smoothing=label_smoothing
            )
            return logits, loss
        if return_hidden_states:
            return logits, hidden_states
        else:
            return logits

    def generate(
        self,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        temperature=1.0,
        topk_filter_thres=0.9,
        can_remask_prev_masked=False,  # TODO: implement this
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=3,
        noise_schedule: Callable = cosine_schedule,
    ):
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        seq_len = self.config.num_vq_tokens

        batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
        shape = (batch_size, seq_len)

        # shift the class ids by the codebook size
        if class_ids is not None:
            class_ids += self.config.codebook_size

        # initialize with all image tokens masked
        input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id
        scores = torch.zeros(shape, dtype=torch.float32, device=self.device)

        starting_temperature = temperature

        for timestep, steps_until_x0 in tqdm(
            zip(torch.linspace(0, 1, timesteps, device=self.device), reversed(range(timesteps))), total=timesteps
        ):
            rand_mask_prob = noise_schedule(timestep)
            num_token_masked = max(int((rand_mask_prob * seq_len).item()), 1)

            masked_indices = scores.topk(num_token_masked, dim=-1).indices
            input_ids = input_ids.scatter(1, masked_indices, mask_token_id)

            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            # classifier free guidance
            if encoder_hidden_states is not None and guidance_scale > 0:
                uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
                model_input = torch.cat([input_ids] * 2)
                condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
                cond_logits, uncond_logits = self(model_input, encoder_hidden_states=condition).chunk(2)
                cond_logits = cond_logits[..., : self.config.codebook_size]
                uncond_logits = uncond_logits[..., : self.config.codebook_size]
                logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            else:
                logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
                logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            filtered_logits = top_k(logits, topk_filter_thres)

            temperature = starting_temperature * (steps_until_x0 / timesteps)  # temperature is annealed

            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            is_mask = input_ids == mask_token_id

            input_ids = torch.where(is_mask, pred_ids, input_ids)

            probs_without_temperature = F.softmax(logits, dim=-1)

            scores = 1 - probs_without_temperature.gather(2, pred_ids[..., None])
            scores = rearrange(scores, "... 1 -> ...")  # TODO: use torch

        return input_ids
    def generate2(
        self,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        temperature=1.0,
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

            # classifier free guidance
            if encoder_hidden_states is not None and guidance_scale > 0:
                uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
                model_input = torch.cat([input_ids] * 2)
                condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
                cond_logits, uncond_logits = self(model_input, encoder_hidden_states=condition).chunk(2)
                cond_logits = cond_logits[..., : self.config.codebook_size]
                uncond_logits = uncond_logits[..., : self.config.codebook_size]
                logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            else:
                logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
                logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            # sampled_ids = torch.multinomial(logits.softmax(dim=-1), 1)
            sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])
            
            if return_intermediate:
                intermediate.append(sampled_ids)

            # Just updates the masked tokens.
            unknown_map = input_ids == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
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
    
    def generate_correct(
        self,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        return_intermediate=False,
        momentum=1.0,
        min_c_step=0,
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

            # classifier free guidance
            if encoder_hidden_states is not None and guidance_scale > 0:
                uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
                model_input = torch.cat([input_ids] * 2)
                condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
                cond_logits, uncond_logits = self(model_input, encoder_hidden_states=condition).chunk(2)
                cond_logits = cond_logits[..., : self.config.codebook_size]
                uncond_logits = uncond_logits[..., : self.config.codebook_size]
                logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
            else:
                logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
                logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            # sampled_ids = torch.multinomial(logits.softmax(dim=-1), 1)
            sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])
            
            if return_intermediate:
                intermediate.append(sampled_ids)

            # Just updates the masked tokens.
            unknown_map = input_ids == mask_token_id
            if step < min_c_step:
                sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            probs = logits.softmax(dim=-1)
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            if step < min_c_step:
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
    
    def pseudo_forward(
        self,
        input_ids,
        additional_input_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        label_smoothing=0.0,
        cond_dropout_prob=0.0,
    ):
        hidden_states = self.embed(input_ids)
        if additional_input_ids is not None:
            hidden_states = torch.cat([hidden_states, additional_input_ids], dim=1)
            print('hidden states length', hidden_states.shape[1])

        if encoder_hidden_states is not None and self.config.project_encoder_hidden_states:
            encoder_hidden_states = self.encoder_proj(encoder_hidden_states)
            encoder_hidden_states = self.encoder_proj_layer_norm(encoder_hidden_states)

        # condition dropout for classifier free guidance
        if encoder_hidden_states is not None and self.training and cond_dropout_prob > 0.0:
            batch_size = encoder_hidden_states.shape[0]
            mask = prob_mask_like((batch_size, 1, 1), 1.0 - cond_dropout_prob, encoder_hidden_states.device)
            encoder_hidden_states = encoder_hidden_states * mask
        for layer in self.transformer_layers:
            if self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = checkpoint(
                    create_custom_forward(layer), hidden_states, encoder_hidden_states, encoder_attention_mask
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
        if self.config.use_encoder_layernorm:
            hidden_states = self.encoder_layer_norm(hidden_states)

        if self.config.use_mlm_layer:
            logits = self.mlm_layer(hidden_states)
        else:
            logits = self.to_logits(hidden_states)
            
        return logits[:,:257]
    
    def generate_blank_canvas(
        self,
        class_ids: torch.LongTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        temperature=1.0,
        timesteps=18,  # ideal number of steps is 18 in maskgit paper
        guidance_scale=0,
        noise_schedule=cosine_schedule,
        return_intermediate=False,
        momentum=1.0,
        min_c_step=0,
        maximum_thinking_ratio = 0.2,
        max_thinking_step = 5,
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
        blank_canvas = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id
        sampled_ids_only = None
        sampled_pos = None
        additional_input_ids = None
        for step in range(max_thinking_step):
            # prepend class token to input_ids
            if sampled_ids_only is not None:
                # attach previous ids with position ids
                # sampled_pos = torch.cat([class_ids[:, None], sampled_pos], dim=1)
                pos_embed = self.embed.position_embeddings(sampled_pos + 1)
                tkn_embed = self.embed.word_embeddings(sampled_ids_only)
                additional_input_ids = pos_embed + tkn_embed
                
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], blank_canvas], dim=1)

            logits = self.pseudo_forward(input_ids, additional_input_ids = additional_input_ids, encoder_hidden_states=encoder_hidden_states)
            logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            # sampled_ids = torch.multinomial(logits.softmax(dim=-1), 1)
            sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])
            
            # if return_intermediate:
            #     intermediate.append(sampled_ids)

            # Just updates the masked tokens.
            unknown_map = input_ids == mask_token_id
            if step < min_c_step:
                sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            mask_ratio = mask_ratio.clamp(maximum_thinking_ratio, 1)
            # Computes the probabilities of each selected tokens.
            probs = logits.softmax(dim=-1)
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            if step < min_c_step:
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
            # if return_intermediate:
            #     mask_index.append(masking)
                
            # Masks tokens with lower confidence.
            # input_ids = torch.where(masking, mask_token_id, sampled_ids)
            sampled_ids_only = sampled_ids[~masking].reshape(batch_size, -1)
            sampled_pos = torch.where(masking==False)[1].reshape(batch_size, -1)
            
        input_ids = blank_canvas
        for t in range(timesteps):
            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            logits = self.pseudo_forward(input_ids, additional_input_ids = additional_input_ids, encoder_hidden_states=encoder_hidden_states)
            logits = logits[..., : self.config.codebook_size]

            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            # sampled_ids = torch.multinomial(logits.softmax(dim=-1), 1)
            sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])
            
            if return_intermediate:
                intermediate.append(sampled_ids)

            # Just updates the masked tokens.
            unknown_map = input_ids == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
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
    
    def generate_incremental_salient_prev(
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

        hidden_states = None
        logits=None
        for step in range(timesteps):
            prev_hidden_states = hidden_states
            prev_logits = logits
            # prepend class token to input_ids
            if class_ids is not None:
                input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

            logits, hidden_states = self(input_ids, encoder_hidden_states=encoder_hidden_states, return_hidden_states=True)
            logits = logits[..., : self.config.codebook_size]
                
            # remove class token
            if class_ids is not None:
                input_ids = input_ids[:, 1:]
                logits = logits[:, 1:]
                
            # print(guidance_scale)
            if guidance_scale > 0 and prev_logits is not None and mask_ratio.item() > 0.0 : # guidance with prev state
                logits = prev_logits + guidance_scale * (logits - prev_logits)

            # Samples the ids using categorical sampling: [batch_size, seq_length].
            # sampled_ids = torch.multinomial(logits.softmax(dim=-1), 1)
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
            # temperature = temperature * (1.0 - ratio) + temperature2
            # print('t: ', temperature, '1.0 - ratio: ', 1.0-ratio)
            temperature = temperature * (1.0 - ratio) + temperature2
            masking = mask_by_random_topk(mask_len, selected_probs, temperature)
            if return_intermediate:
                mask_index.append(masking)
                
            # Masks tokens with lower confidence.
            input_ids = torch.where(masking, mask_token_id, sampled_ids)

        if return_intermediate:
            return sampled_ids, intermediate, mask_index
        return sampled_ids
    # def generate2(
    #     self,
    #     class_ids: torch.LongTensor = None,
    #     encoder_hidden_states: torch.FloatTensor = None,
    #     temperature=1.0,
    #     timesteps=18,  # ideal number of steps is 18 in maskgit paper
    #     guidance_scale=0,
    #     noise_schedule=cosine_schedule,
    #     **kwargs,
    # ):
    #     """
    #     Generate 1:1 similar to the original MaskGit repo
    #     https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
    #     """
    #     # begin with all image token ids masked
    #     mask_token_id = self.config.mask_token_id
    #     seq_len = self.config.num_vq_tokens

    #     batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
    #     shape = (batch_size, seq_len)

    #     # shift the class ids by the codebook size
    #     if class_ids is not None:
    #         class_ids += self.config.codebook_size

    #     # initialize with all image tokens masked
    #     input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id

    #     for step in range(timesteps):
    #         # prepend class token to input_ids
    #         if class_ids is not None:
    #             input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

    #         # classifier free guidance
    #         if encoder_hidden_states is not None and guidance_scale > 0:
    #             uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
    #             model_input = torch.cat([input_ids] * 2)
    #             condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
    #             cond_logits, uncond_logits = self(model_input, encoder_hidden_states=condition).chunk(2)
    #             cond_logits = cond_logits[..., : self.config.codebook_size]
    #             uncond_logits = uncond_logits[..., : self.config.codebook_size]
    #             logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
    #         else:
    #             logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
    #             logits = logits[..., : self.config.codebook_size]

    #         # remove class token
    #         if class_ids is not None:
    #             input_ids = input_ids[:, 1:]
    #             logits = logits[:, 1:]

    #         # Samples the ids using categorical sampling: [batch_size, seq_length].
    #         # sampled_ids = torch.multinomial(logits.softmax(dim=-1), 1)
    #         sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])

    #         # Just updates the masked tokens.
    #         unknown_map = input_ids == mask_token_id
    #         sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
    #         # Defines the mask ratio for the next round. The number to mask out is
    #         # determined by mask_ratio * unknown_number_in_the_beginning.
    #         ratio = 1.0 * (step + 1) / timesteps
    #         mask_ratio = noise_schedule(torch.tensor(ratio))
    #         # Computes the probabilities of each selected tokens.
    #         probs = logits.softmax(dim=-1)
    #         selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
    #         selected_probs = selected_probs.squeeze(-1)

    #         # Ignores the tokens given in the input by overwriting their confidence.
    #         selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
    #         # Gets mask lens for each sample in the batch according to the mask ratio.
    #         mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
    #         # Keeps at least one of prediction in this round and also masks out at least
    #         # one and for the next iteration
    #         mask_len = torch.max(
    #             torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
    #         )

    #         # Adds noise for randomness
    #         temperature = temperature * (1.0 - ratio)
    #         masking = mask_by_random_topk(mask_len, selected_probs, temperature)
    #         # Masks tokens with lower confidence.
    #         input_ids = torch.where(masking, mask_token_id, sampled_ids)

    #     return sampled_ids
    
    # def generate3(
    #     self,
    #     class_ids: torch.LongTensor = None,
    #     encoder_hidden_states: torch.FloatTensor = None,
    #     temperature=1.0,
    #     timesteps=18,  # ideal number of steps is 18 in maskgit paper
    #     guidance_scale=0,
    #     noise_schedule=cosine_schedule,
    #     **kwargs,
    # ):
    #     """
    #     Generate 1:1 similar to the original MaskGit repo
    #     https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
    #     + temperature annealing
    #     """
    #     # begin with all image token ids masked
    #     mask_token_id = self.config.mask_token_id
    #     seq_len = self.config.num_vq_tokens

    #     batch_size = len(class_ids) if class_ids is not None else encoder_hidden_states.shape[0]
    #     shape = (batch_size, seq_len)

    #     # shift the class ids by the codebook size
    #     if class_ids is not None:
    #         class_ids += self.config.codebook_size

    #     # initialize with all image tokens masked
    #     input_ids = torch.ones(shape, dtype=torch.long, device=self.device) * mask_token_id

    #     for step in range(timesteps):
    #         # prepend class token to input_ids
    #         if class_ids is not None:
    #             input_ids = torch.cat([class_ids[:, None], input_ids], dim=1)

    #         # classifier free guidance
    #         if encoder_hidden_states is not None and guidance_scale > 0:
    #             uncond_encoder_states = torch.zeros_like(encoder_hidden_states)
    #             model_input = torch.cat([input_ids] * 2)
    #             condition = torch.cat([encoder_hidden_states, uncond_encoder_states])
    #             cond_logits, uncond_logits = self(model_input, encoder_hidden_states=condition).chunk(2)
    #             cond_logits = cond_logits[..., : self.config.codebook_size]
    #             uncond_logits = uncond_logits[..., : self.config.codebook_size]
    #             logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
    #         else:
    #             logits = self(input_ids, encoder_hidden_states=encoder_hidden_states)
    #             logits = logits[..., : self.config.codebook_size]

    #         # remove class token
    #         if class_ids is not None:
    #             input_ids = input_ids[:, 1:]
    #             logits = logits[:, 1:]

    #         # Samples the ids using categorical sampling: [batch_size, seq_length].
    #         # sampled_ids = torch.multinomial(logits.softmax(dim=-1), 1)
    #         sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])

    #         # Just updates the masked tokens.
    #         unknown_map = input_ids == mask_token_id
    #         sampled_ids = torch.where(unknown_map, sampled_ids, input_ids)
    #         # Defines the mask ratio for the next round. The number to mask out is
    #         # determined by mask_ratio * unknown_number_in_the_beginning.
    #         ratio = 1.0 * (step + 1) / timesteps
    #         mask_ratio = noise_schedule(torch.tensor(ratio))
    #         # Computes the probabilities of each selected tokens.
    #         probs = logits.softmax(dim=-1)
    #         selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
    #         selected_probs = selected_probs.squeeze(-1)

    #         # Ignores the tokens given in the input by overwriting their confidence.
    #         selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
    #         # Gets mask lens for each sample in the batch according to the mask ratio.
    #         mask_len = (seq_len * mask_ratio).floor().unsqueeze(0).to(logits.device)
    #         # Keeps at least one of prediction in this round and also masks out at least
    #         # one and for the next iteration
    #         mask_len = torch.max(
    #             torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
    #         )

    #         # Adds noise for randomness
    #         temperature = temperature * (1.0 - ratio)
    #         masking = mask_by_random_topk(mask_len, selected_probs, temperature)
    #         # Masks tokens with lower confidence.
    #         input_ids = torch.where(masking, mask_token_id, sampled_ids)

    #     return sampled_ids