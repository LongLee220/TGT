# TGT: A Temporal Gating Transformer for Smartphone App Usage Prediction

This project implements the TATNet model, a temporal attention-based Transformer framework designed to improve app usage prediction by integrating temporal encoding and feature encoding.

<img src=Atten_model.jpg width=900>

---

## Table of Contents
1. [Overview](#overview)
2. [Environment Requirements](#environment-requirements)
3. [Installation Instructions](#installation-instructions)
4. [Data](#data)
5. [Preprocessing](#paraments)
6. [Run](#run)
7. [Result](#result)


---

## Overview

Accurately predicting app usage patterns can significantly enhance user experience and system performance by preloading app-related resources, reducing startup latency, and improving energy efficiency. TGT combines a Transformer with a temporal attention mechanism to prioritize the most informative app usage sequences, significantly improving prediction accuracy.

---

## Environment Requirements

This project requires the following environment and dependencies to be installed:

### Base Environment
- **Operating System:** Ubuntu 22.04
- **Python Version:** 3.10.12
- **CUDA Version:** 11.8.0 (optional for GPU support)

### Python Dependencies
Below are the required Python packages and their versions:

| Package         | Version    |
|-----------------|------------|
| `torch`         | Compatible with CUDA 11.8 (installed from [PyTorch official index](https://download.pytorch.org/whl/cu118)) |
| `torchvision`   | Compatible with CUDA 11.8 |
| `torchaudio`    | Compatible with CUDA 11.8 |
| `logzero`       | 1.7.0      |
| `networkx`      | 3.1        |
| `pandas`        | 2.0.3      |
| `pyyaml`        | 6.0.2      |
| `scikit-learn`  | 1.3.2      |
| `matplotlib`    | 3.8.0      |

---

## Installation Instructions

To set up the environment locally, follow these steps:

### 1. Install Python and CUDA
Ensure you have Python 3.10.12 installed on your system. If GPU acceleration is needed, make sure CUDA 11.8 is installed and configured properly.

### 2. Install Required Python Dependencies
Run the following command to install all necessary Python packages:

```bash
pip install -U \
    logzero==1.7.0 \
    networkx==3.1 \
    pandas==2.0.3 \
    pykan==0.2.8 \
    pyyaml==6.0.2 \
    scikit-learn==1.3.2 \
    matplotlib==3.8.0
```
### 3. Install PyTorch with CUDA 11.8 Support


## Dataset

[Tsinghua App Usage Dataset](http://fi.ee.tsinghua.edu.cn/appusage/)


```
@article{yu2018smartphone,
    title={Smartphone app usage prediction using points of interest},
    author={Yu, Donghan and Li, Yong and Xu, Fengli and Zhang, Pengyu and Kostakos, Vassilis},
    journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
    volume={1},
    number={4},
    pages={174},
    year={2018},
    publisher={ACM}}
```
Extract `App_usage_trace.txt` and put in in the `./Tsinghua/data` directory.


[LSApp Dataset](https://github.com/aliannejadi/LSApp)
```
@inproceedings{AliannejadiTOIS21,
    author    = {Aliannejadi, Mohammad and Zamani, Hamed and Crestani, Fabio and Croft, W. Bruce},
    title     = {Context-Aware Target Apps Selection and Recommendation for Enhancing Personal Mobile Assistants},
    booktitle = {{ACM} Transactions on Information Systems ({TOIS})},
    year      = 2021
  }
```
Extract `lsapp.tsv` and put in in the `./LSapp/data` directory.

## paraments
If you want to change sequence length, change `--seq_length` in main.py.
If you want to change split method, change `--split` in main.py.


## Run

After extracting dataset and preprocessing,

```bash
python main.py
```

