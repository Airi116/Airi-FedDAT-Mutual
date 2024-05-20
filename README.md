# FedDAT-Mutual (Federated Dual-Adapter Teacher)

A refined approach for foundation model finetuning in multi-modal heterogeneous federated learning. [ [Pre-print]](https://arxiv.org/pdf/2308.12305.pdf)

![Problem Setup](/assets/fedvqa.png "Magic Gardens")

We propose the Dual-Adapter Teacher (DAT) module and apply Mutual Knowledge Distillation (MKD) to mitigate the client local data heterogeneity in different modality.

![Method](/assets/dat.png "Method")


---

## Setup

1. Create Conda environment with Python 3.8

```
conda create -n feddat python=3.8
conda activate feddat
```

2. Install requirements

```
 git clone https://github.com/Airi116/FedDAT.git
pip install -r requirements.txt
 pip install -U adapters
pip install accelerate
```
3. Prepare datasets and pretrained-models

Put the datasets and the models in the folders /data and /models respectively. The links for the datasets and models are provided in the original repository.

---

## Run

```
# Training with ViLT
bash src/train_vilt.sh

# Training with ALBEF
bash src/train_albef.sh
```

---

## Citation

```bibtex
@article{chen2023feddat,
  title={FedDAT: An Approach for Foundation Model Finetuning in Multi-Modal Heterogeneous Federated Learning},
  author={Chen, Haokun and Zhang, Yao and Krompass, Denis and Gu, Jindong and Tresp, Volker},
  journal={arXiv preprint arXiv:2308.12305},
  year={2023}
}
```