# ADVERSARIAL DEFENSE FOR DEEP SPEAKER RECOGNITION USING HYBRID ADVERSARIAL TRAINING

We propose a new adversarial defense mechanism based on a hybrid adversarial training (HAT) setup. In contrast to existing works on countermeasures against adversarial attacks in deep speaker recognition that only use class boundary information by supervised cross entropy (CE) loss, we propose to exploit additional information from supervised and unsupervised cues to craft diverse and stronger perturbations for adversarial training. Specifically, we employ multi-task objectives using CE, feature scattering (FS), and margin losses to create adversarial perturbations and include them for adversarial training to enhance the robustness of the model. 

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

2. All black-box attack testing: CUDA_VISIBLE_DEVICES=0 python test_blackbox.py
