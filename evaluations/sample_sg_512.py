import sys
import os
from tqdm import tqdm

sys.path.append('.')
import torch

import numpy as np
device="cuda:1"
from muse.pipeline_muse_toast import PipelineMuse


model_paths = [
    "results_corr/ft_512_toast_cls_b256_corr/checkpoint-50000/ema_model",
]
for model_path in model_paths:
    pipe = PipelineMuse.from_pretrained(transformer_path=model_path, 
                                        is_class_conditioned=True,
                                        vae_path="./scripts/tokenizer_imagenet512_torch/",
                                        use_toast=True).to(device)
    pipe.transformer.eval()
    pipe.vae.eval()

    sample_root = f"."      
    
    temperatures = [45]
    sampling_types = ['self_guidance'] # maskgit, self_guidance
    schedule = ["cosine"]
    sampling_steps = [18]
    guidance_scale = [1]
    
    for gs in guidance_scale:
        for schedule_type in schedule:
            for sampling_type in sampling_types:
                for t in temperatures:
                    for i in range(len(sampling_steps)):
                        step = sampling_steps[i]
                        batch_size = 32
                        # num_images = 50000
                        num_images = 500
                        num_iter = num_images // batch_size + 1
                        
                        model_name = model_path.split('/')[1]
                        checkpoint_name = model_path.split('/')[2]

                        save_dir = os.path.join(sample_root, f"{model_name}_{checkpoint_name}_{num_images//1000}k_{sampling_type}_{step}_{schedule_type}_{t}_{schedule_type}_gs_{gs}.npz")
                        print(save_dir)
                        
                        all_images = []
                        all_labels = []
                        for o in tqdm(range(1, num_iter+1)):
                            class_ids = torch.randint(0, 1000, (batch_size,))
                            images = pipe(class_ids=class_ids, num_images_per_prompt=batch_size, 
                                            return_intermediate=False, timesteps=step, temperature=t, 
                                            sampling_type=sampling_type, schedule=schedule_type, guidance_scale=gs, min_c_ratio=0.5)
                            
                            all_images += images
                            all_labels.append(class_ids.numpy())
                        
                        arr = np.array([np.array(image) for image in all_images])
                        arr = arr[:num_images]
                        label_arr = np.concatenate(all_labels)
                        label_arr = label_arr[:num_images]
                        
                        np.savez(save_dir, arr, label_arr)
                