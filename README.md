# UnlockMGM
[NeurIPS 2024] Official implementation of "Unlocking the Capabilities of Masked Generative Models for Image Synthesis via Self-Guidance"

1. Setup
 - open-muse
2. model setup
 - quick start
 - 학습해볼사람은 MaskGIT official repo에서 jax 받은 후에 script 실행해서 변환사용
3. fine-tuning
 - 실행 script 주기
 accelerate launch --config_file acc_config_mult.yaml training/ft_maskgit_org_toast.py config=configs/ft_256_toast_cls_b256_corr.yaml
 accelerate launch --config_file acc_config_mult.yaml training/ft_maskgit_org_toast.py config=configs/ft_512_toast_cls_b256_corr.yaml
4. sampling
 - 실행 script 주기
