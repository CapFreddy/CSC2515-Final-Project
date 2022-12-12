# Learning Disentangled Features for Domain Generalization

## Setup

### Dataset

Download dataset from https://drive.google.com/file/d/15V7EsHfCcfbKgsDmzQKj_DfXt_XYp_P7/view?pli=1. 

Unzip it to root of this repo.

### Python Environment

```bash
conda env create -f environment.yml
```

Install PyTorch from https://pytorch.org/get-started/locally/ based on your hardware.

## Instruction

### $\text{DisEnt}_{\text{FINAL}}$
```bash
python src/train.py --disentangle_layer 3 --target_domain mnist --logdir result/method1
```

### $\text{DisEnt}_{\text{LATENT}}$
```bash
python src/train.py --disentangle_layer 2 --target_domain mnist --logdir result/method2
```

### $\text{DisEnt}_{\text{FINAL-FINETUNE}}$
```bash
python src/train.py --disentangle_layer 2 --finetune --target_domain mnist --logdir result/method3
```


### Baseline Methods

To run non-neural-network baseline methods for each domains and 3 seeds

```bash
python src/baseline/baseline.py --target_domain mnist -s 1 -o output/target-mnist-seed-1.csv
python src/baseline/baseline.py --target_domain mnist -s 2 -o output/target-mnist-seed-2.csv
python src/baseline/baseline.py --target_domain mnist -s 3 -o output/target-mnist-seed-3.csv

python src/baseline/baseline.py --target_domain mnist_m -s 1 -o output/target-mnist_m-seed-1.csv
python src/baseline/baseline.py --target_domain mnist_m -s 2 -o output/target-mnist_m-seed-2.csv
python src/baseline/baseline.py --target_domain mnist_m -s 3 -o output/target-mnist_m-seed-3.csv

python src/baseline/baseline.py --target_domain svhn -s 1 -o output/target-svhn-seed-1.csv
python src/baseline/baseline.py --target_domain svhn -s 2 -o output/target-svhn-seed-2.csv
python src/baseline/baseline.py --target_domain svhn -s 3 -o output/target-svhn-seed-3.csv

python src/baseline/baseline.py --target_domain syn -s 1 -o output/target-syn-seed-1.csv
python src/baseline/baseline.py --target_domain syn -s 2 -o output/target-syn-seed-2.csv
python src/baseline/baseline.py --target_domain syn -s 3 -o output/target-syn-seed-3.csv
```

Results are saved to `./output` in the root of the repo as csv files.

- [src/baseline/baseline.py](src/baseline/baseline.py) is for running experiments.
- [src/baseline/baseline.ipynb](src/baseline/baseline.ipynb) was used to develop `baseline.py` and visualize the results. It also contains the hyper parameters we used.
- [src/baseline/parse_output.ipynb](src/baseline/parse_output.ipynb) can be used to parse the output csv files and aggregate the data.


