# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    CLIPTextModel,
    PreTrainedTokenizer,
    T5EncoderModel,
)
import pdb

from .modeling_maskgit_vqgan import MaskGitVQGAN
from .modeling_taming_vqgan import VQGANModel
from .modeling_transformer import MaskGitTransformer
from .modeling_transformer_toast import MaskGitTransformerTOAST
from .sampling import cosine_schedule, linear_schedule, sqrt_schedule

def schedule_func(t, mu=0.8, v=2):
    eps = 1e-4
    t = t+eps
    some = (t*(1-mu)/(mu*(1-t)))**(-v)
    return 1-1/(1+some) + eps

class PipelineMuse:
    def __init__(
        self,
        vae: Union[MaskGitVQGAN, VQGANModel],
        transformer: Union[MaskGitTransformer, MaskGitTransformerTOAST],
        is_class_conditioned: bool = False,
        text_encoder: Optional[Union[T5EncoderModel, CLIPTextModel]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.transformer = transformer
        self.is_class_conditioned = is_class_conditioned
        self.device = "cpu"

    def to(self, device="cpu", dtype=torch.float32) -> None:
        if not self.is_class_conditioned:
            self.text_encoder.to(device, dtype=dtype)
        self.vae.to(device, dtype=dtype)
        self.transformer.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype
        return self

    @torch.no_grad()
    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        class_ids: torch.LongTensor = None,
        timesteps: int = 8,
        guidance_scale: float = 8.0,
        temperature: float = 1.0,
        temperature2: float = 0.0,
        topk_filter_thres: float = 0.9,
        num_images_per_prompt: int = 1,
        sampling_type: str = "self_guidance",
        return_intermediate: bool = False,
        min_c_step:int = 0,
        max_c_step:int = 999,
        momentum: float = 0.0,
        correct_every: int = -1,
        threshold: float = 0.0,
        use_toast_correct_only: bool = False,
        no_temp_for_correct: bool = False,
        substitution_rate: float = 0.2,
        schedule: str = "cosine",
        stop_tdib_step: int = 100,
        min_c_ratio=0.0,
        guidance_anneal=False,
        degub: bool = False,
    ):
        if text is None and class_ids is None:
            raise ValueError("Either text or class_ids must be provided.")

        if text is not None and class_ids is not None:
            raise ValueError("Only one of text or class_ids may be provided.")

        if class_ids is not None:
            if isinstance(class_ids, int):
                class_ids = [class_ids]

                class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.long)
                # duplicate class ids for each generation per prompt
                class_ids = class_ids.repeat_interleave(num_images_per_prompt, dim=0)
            elif isinstance(class_ids, torch.Tensor):
                class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.long)
                
            model_inputs = {"class_ids": class_ids}
        else:
            if isinstance(text, str):
                text = [text]

            input_ids = self.tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=16
            ).input_ids  # TODO: remove hardcode
            input_ids = input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(input_ids).last_hidden_state

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed, seq_len, _ = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            encoder_hidden_states = encoder_hidden_states.view(bs_embed * num_images_per_prompt, seq_len, -1)
            model_inputs = {"encoder_hidden_states": encoder_hidden_states}

        if sampling_type == 'maskgit':
            generate = self.transformer.generate2
        elif sampling_type == 'self_guidance':
            generate = self.transformer.generate_sg
        else:
            raise NotImplementedError
            
        if schedule == "cosine":
            noise_schedule = cosine_schedule
        elif schedule == "linear":
            noise_schedule = linear_schedule
        elif schedule == "sqrt":
            noise_schedule = sqrt_schedule
        elif schedule == "custom":
            noise_schedule = schedule_func
        else:
            raise NotImplementedError
            
        outputs = generate(
            **model_inputs,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            temperature=temperature,
            return_intermediate=return_intermediate,
            noise_schedule=noise_schedule,
        )
        
        if return_intermediate:
            generated_tokens, intermediate, mask_index = outputs
        else:
            generated_tokens = outputs

        images = self.vae.decode_code(generated_tokens)
        if return_intermediate:
            intermediate_images = [self.vae.decode_code(tokens) for tokens in intermediate]

        # Convert to PIL images
        images = [self.to_pil_image(image) for image in images]
        if return_intermediate:
            intermediate_images = [[self.to_pil_image(image) for image in images] for images in intermediate_images]
            # return images, intermediate_images, mask_index
            return images, intermediate_images, intermediate, mask_index
        return images

    def to_pil_image(self, image: torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = 2.0 * image - 1.0
        image = np.clip(image, -1.0, 1.0)
        image = (image + 1.0) / 2.0
        image = (255 * image).astype(np.uint8)
        image = Image.fromarray(image).convert("RGB")
        return image

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = None,
        text_encoder_path: Optional[str] = None,
        vae_path: Optional[str] = None,
        transformer_path: Optional[str] = None,
        is_class_conditioned: bool = False,
        use_toast: bool = False,
        use_taming: bool = False,
        debug: bool = False,
    ) -> None:
        """
        Instantiate a PipelineMuse from a pretrained model. Either model_name_or_path or all of text_encoder_path, vae_path, and
        transformer_path must be provided.
        """         
        MaskGitTransformer = MaskGitTransformerTOAST if use_toast else MaskGitTransformer
        # if debug:
        #     from .modeling_transformer_org_correct_debugging import MaskGitCorrTransformerOrg
        #     MaskGitTransformer = MaskGitCorrTransformerOrg
        #     print("set debug mode!")
        
        if model_name_or_path is None:
            if vae_path is None or transformer_path is None:
                raise ValueError(
                    "If model_name_or_path is None, then text_encoder_path, vae_path, and transformer_path must be"
                    " provided."
                )

            text_encoder = None
            tokenizer = None

            if not is_class_conditioned:
                text_encoder = T5EncoderModel.from_pretrained(text_encoder_path)
                tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)

            if use_taming:
                vae = VQGANModel.from_pretrained(vae_path)
            else:
                vae = MaskGitVQGAN.from_pretrained(vae_path)
            transformer = MaskGitTransformer.from_pretrained(transformer_path)
        else:
            text_encoder = None
            tokenizer = None

            if not is_class_conditioned:
                text_encoder = T5EncoderModel.from_pretrained(model_name_or_path, subfolder="text_encoder")
                tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, subfolder="text_encoder")
            if use_taming:
                vae = VQGANModel.from_pretrained(model_name_or_path, subfolder="vae")
            else:
                vae = MaskGitVQGAN.from_pretrained(model_name_or_path, subfolder="vae")
            transformer = MaskGitTransformer.from_pretrained(model_name_or_path, subfolder="transformer")

        return cls(
            vae=vae,
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            is_class_conditioned=is_class_conditioned,
        )
        
    @torch.no_grad()
    def correct_vq_code(
        self,
        vq_tokens: torch.Tensor = None,
        class_ids: torch.LongTensor = None,
        correct_step: int = 1,
    ):
        """
        # refine the given vq_code with MaskGIT Corrector
        # vq_codes: [batch_size, seq_len]
        """
        
        batch_size, seq_len = vq_tokens.shape
        if class_ids is not None:
            if isinstance(class_ids, int):
                class_ids = [class_ids]

                class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.long)
                # duplicate class ids for each generation per prompt
                class_ids = class_ids.repeat_interleave(batch_size, dim=0)
            elif isinstance(class_ids, torch.Tensor):
                class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.long)
                
        input_ids = torch.cat([class_ids.unsqueeze(1), vq_tokens], dim=1)
            
        for i in correct_step:
                # classifier free guidance
                logits = self(vq_tokens, encoder_hidden_states=encoder_hidden_states)
                logits = logits[..., : self.config.codebook_size]                
                    
                # remove class token
                if class_ids is not None:
                    input_ids = input_ids[:, 1:]
                    logits = logits[:, 1:]
                    
                # add gumbel noise for correction for more diversity
                # logits = logits + temperature * gumbel_noise(logits)
                # logits = logits + 1 * gumbel_noise(logits)
                unknown_map = input_ids == mask_token_id

                # Samples the ids using categorical sampling: [batch_size, seq_length].
                # sampled_ids = torch.stack([torch.multinomial(l.softmax(dim=-1), 1).squeeze(1) for l in logits])
                sampled_ids = gumbel_sample(logits, temperature=1.0 * (1.0 - ratio))
                
                # Replace the input_ids with highest probability tokens
                selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
                selected_probs = selected_probs.squeeze(-1)
                # Mark the masked region so that it is not replaced
                selected_probs = torch.where(unknown_map, torch.finfo(selected_probs.dtype).min, selected_probs)
                
                num_unmasked = (~unknown_map).sum(dim=1)
                # num_substitute = (num_unmasked * substitution_rate * (1.0 - ratio)).round().clamp(min=1)[..., None]
                num_substitute = (num_unmasked * substitution_rate).round().clamp(min=1)[..., None]
                
                # Negation of num_substitute means select only num_substitute tokens using mask_by_random_topk
                substitute_mask = mask_by_random_topk(seq_len-num_substitute, selected_probs, temperature=0.0)
                
                input_ids = torch.where(substitute_mask, input_ids, sampled_ids)

        if return_intermediate:
            return sampled_ids, intermediate, mask_index
        return sampled_ids
        
    @torch.no_grad()
    def transfer(
        self,
        text: Optional[Union[str, List[str]]] = None,
        class_ids: torch.LongTensor = None,
        timesteps: int = 8,
        guidance_scale: float = 8.0,
        temperature: float = 1.0,
        num_images_per_prompt: int = 1,
        sampling_type: str = "maskgit",
        return_intermediate: bool = False,
        schedule: str = "cosine",
    ):
        if text is None and class_ids is None:
            raise ValueError("Either text or class_ids must be provided.")

        if text is not None and class_ids is not None:
            raise ValueError("Only one of text or class_ids may be provided.")

        if class_ids is not None:
            if isinstance(class_ids, int):
                class_ids = [class_ids]

                class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.long)
                # duplicate class ids for each generation per prompt
                class_ids = class_ids.repeat_interleave(num_images_per_prompt, dim=0)
            elif isinstance(class_ids, torch.Tensor):
                class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.long)
                
            model_inputs = {"class_ids": class_ids}
        else:
            if isinstance(text, str):
                text = [text]

            input_ids = self.tokenizer(
                text, return_tensors="pt", padding="max_length", truncation=True, max_length=16
            ).input_ids  # TODO: remove hardcode
            input_ids = input_ids.to(self.device)
            encoder_hidden_states = self.text_encoder(input_ids).last_hidden_state

            # duplicate text embeddings for each generation per prompt, using mps friendly method
            bs_embed, seq_len, _ = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            encoder_hidden_states = encoder_hidden_states.view(bs_embed * num_images_per_prompt, seq_len, -1)
            model_inputs = {"encoder_hidden_states": encoder_hidden_states}

        if sampling_type == 'maskgit':
            generate = self.transformer.generate2
        elif sampling_type == 'sg':
            generate = self.transformer.generate_sg
        else:
            raise NotImplementedError

        if schedule == "cosine":
            noise_schedule = cosine_schedule
        elif schedule == "linear":
            noise_schedule = linear_schedule
        elif schedule == "sqrt":
            noise_schedule = sqrt_schedule
        elif schedule == "custom":
            noise_schedule = schedule_func
        else:
            raise NotImplementedError
            
        outputs = generate(
            **model_inputs,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            temperature=temperature,
            return_intermediate=return_intermediate,
            noise_schedule=noise_schedule,
        )
        # import pdb
        # pdb.set_trace()
        
        if return_intermediate:
            generated_tokens, intermediate, mask_index = outputs
        else:
            generated_tokens = outputs

        images = self.vae.decode_code(generated_tokens)
        if return_intermediate:
            intermediate_images = [self.vae.decode_code(tokens) for tokens in intermediate]

        # Convert to PIL images
        images = [self.to_pil_image(image) for image in images]
        if return_intermediate:
            intermediate_images = [[self.to_pil_image(image) for image in images] for images in intermediate_images]
            # return images, intermediate_images, mask_index
            return images, intermediate_images, intermediate, mask_index
        return images