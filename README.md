# **Multi-Period Condition-Enhanced Diffusion Model for Traffic Flow Imputation**





## Requirements

```
info_nce_pytorch==0.1.4
matplotlib==3.11.2
numpy==1.23.5
PyYAML==6.0.1
PyYAML==6.0.3
scipy==1.18.1
thop==0.1.1.post2209072238
timm==1.0.30
torch==2.0.1+cu118
torchinfo==1.8.0
tqdm==4.66.4
```

Please install the packages in requirements.txt

```
pip install -r requirements.txt
```



## Data

The datasets used in this experiment (PEMS03, PEMS04, PEMS08, METR-LA) are all publicly available. You can access them through the following links:

- [PEMS03、PEMS04、PEMS08](https://github.com/Davidham3/STSGCN)
- [METR-LA](https://github.com/liyaguang/DCRNN)

Alternatively, you can access the datasets through the [BasicTS](https://gitcode.com/gh_mirrors/ba/BasicTS/tree/master/datasets) dataset link.



## Usage



**step1:** Download the datasets to the `dataset` folder. The final directory structure should be as follows:

```
dataset
│
├── METR_LA
│   ├── METR_LA.csv
│   └── METR_LA.npz
│
├── PEMS03
│   ├── PEMS03.csv
│   ├── PEMS03.npz
│   └── PEMS03.txt
│
├── PEMS04
│   ├── PEMS04.csv
│   └── PEMS04.npz
│
└── PEMS08
    ├── PEMS08.csv
    └── PEMS08.npz
```

**step2:** In the `configuration\MPCDI` folder, provide the following configuration files:

```shell
MPCDI_METR_LA_Block_0.05.yaml
MPCDI_METR_LA_Point_0.2.yaml
MPCDI_PEMS03_Block_0.05.yaml
MPCDI_PEMS03_Point_0.2.yaml
MPCDI_PEMS04_Block_0.05.yaml
MPCDI_PEMS04_Point_0.2.yaml
MPCDI_PEMS08_Block_0.05.yaml
MPCDI_PEMS08_Point_0.2.yaml
```

The names of the files to be executed should be listed in the `BatchConfig.yaml`.

**step3:** Execute the following command to run the program:

```shell
Python run.py
```







