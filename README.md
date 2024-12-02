# Unlocking the Capabilities of Masked Generative Models for Image Synthesis via Self-Guidance: Official PyTorch Implementation (NeurIPS 2024)
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2410.13136)&nbsp;

[NeurIPS 2024] Official implementation of the paper "Unlocking the Capabilities of Masked Generative Models for Image Synthesis via Self-Guidance".

![git](https://github.com/user-attachments/assets/dac17103-f85c-43eb-abd9-0eff738e58ec)

## Get Started
Our code is based on the [open-muse](https://github.com/huggingface/open-muse), an open pytorch reproduction of masked generative models such as MaskGIT and MUSE. Please refer the codebase for the more information and source code.

### Installation
We provide a `docker/Dockerfile` to simplify the setup of our repository. Once the Docker container is running, follow the scripts below to get started

``` 
source activate accelerate && pip install xformers

### only for fine-tuning ###
mkdir /tmp/unique_for_apex
cd /tmp/unique_for_apex
SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
cd /tmp/unique_for_apex/apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

### install project dependencies ###
cd /UnlockMGM
pip install -e ".[extra]"
```

Depending on your settings, [apex](https://github.com/NVIDIA/apex) could not be installed successfully. Please carefully check the CUDA version for the Dockerfile or you can fine-tune without fused_adam optimizer.

### Prepare Dataset
Please refer to [open-muse](https://github.com/huggingface/open-muse) to prepare `webdataset` for ImageNet.

## Converting VQGAN and MaskGIT Weights

1. **Download JAX Checkpoints**  
   Download the JAX checkpoints for the VQGAN tokenizer and MaskGIT from the [MaskGIT repository](https://github.com/google-research/maskgit). Place the downloaded files in the `scripts/` directory.

2. **Convert JAX Checkpoints to PyTorch**  
   Run the following scripts to convert the JAX checkpoints to PyTorch format:
  ```bash
  python scripts/convert_maskgit_vqgan.py
  python scripts/convert_maskgit_transformer.py
  ```
  
3. **Adjust Resolution**  
  Update the `resolution` parameter in the converted files to support both 256x256 and 512x512 resolutions as required.

## Fine-tuning TOAST
For ImageNet256,
```
accelerate launch --config_file acc_config_mult.yaml training/ft_maskgit_org_toast.py config=configs/ft_256_toast_cls_b256_corr.yaml
```
For ImageNet512,
```
accelerate launch --config_file acc_config_mult.yaml training/ft_maskgit_org_toast.py config=configs/ft_512_toast_cls_b256_corr.yaml
```
Please adjust the `batch_size` in the config files and the `num_processes` in the `acc_config_mult.yaml` file to ensure that the total batch size matches 256.


## Sample Images with Self-Guidance

We provide [jupyter notebook](https://github.com/JiwanHur/UnlockMGM/blob/main/evaluations/maskgit_toast.ipynb) to sample and visualize the images using self-guidance.

To sample images for the evaluations, use `./evaluations/sample_sg_256.py` and `./evaluations/sample_sg_512.py`.

## Fine-tuning checkpoints
|   model    |  weights  |
|:----------:|:---------:|
|ImageNet-256|[checkpoint](https://huggingface.co/HURJIWAN/UnlockMGM/resolve/main/UnlockMGM_imagenet_256.zip)|
|ImageNet-512|[checkpoint](https://huggingface.co/HURJIWAN/UnlockMGM/resolve/main/UnlockMGM_imagenet_512.zip)|
 

## Acknowledgements
This code is heavily based on the following repositories. Thanks for all authors for their amazing works!
- [open-muse](https://github.com/huggingface/open-muse)
- [maskgit](https://github.com/google-research/maskgit)
- [TOAST](https://github.com/bfshi/TOAST)
- [DiffFit](https://github.com/mkshing/DiffFit-pytorch)
- [guided-diffusion](https://github.com/openai/guided-diffusion)
- [CMLMC](https://github.com/layer6ai-labs/CMLMC)
- [webdataset](https://github.com/webdataset/webdataset)
- [apex](https://github.com/NVIDIA/apex)

## Citation
```
@Article{hur2024unlocking,
  title={Unlocking the Capabilities of Masked Generative Models for Image Synthesis via Self-Guidance},
  author={Hur, Jiwan and Lee, Dong-Jae and Han, Gyojin and Choi, Jaehyun and Jeon, Yunho and Kim, Junmo},
  journal={arXiv preprint arXiv:2410.13136},
  year={2024}
}
```

 