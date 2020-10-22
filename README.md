# gard-adversarial-hybrid-spkid

# Installation

Create your virtual env, and then install the dependencies.

conda create -n adv_spkr python=3.6

conda activate adv_spkr

First, install pytorch 1.4.0 yourself depending on the configuration and GPU availability of your machine. Then,

pip install -r requirements.txt

# Training

1. Standard model training: CUDA_VISIBLE_DEVICES=0 python train_standard_libri.py

2. FGSM model training: CUDA_VISIBLE_DEVICES=0 python train_fgsm_libri.py

3. PGD model training: CUDA_VISIBLE_DEVICES=0 python train_pgd_libri.py

4. Feature Scattering model training: CUDA_VISIBLE_DEVICES=0 python train_fea_scatter_libri.py

5. Hybrid model training: CUDA_VISIBLE_DEVICES=0 python train_fea_scatter_libri_hybrid.py

# Testing

1. All untargeted white-box attack testing: CUDA_VISIBLE_DEVICES=0 python test_fea_scatter_libri.py

2. All untargeted black-box attack testing: CUDA_VISIBLE_DEVICES=0 python test_blackbox.py
