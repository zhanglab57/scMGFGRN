# scMGFGRN

scMGFGRN: A multi-modal deep learning framework for gene regulatory network inference from single-cell transcriptomic data integrating sequence and functional hierarchy through gated attention fusion 
## Overview

scMGF-GRNS is a deep learning framework for reconstructing Gene Regulatory Networks (GRNs) from single-cell RNA sequencing (scRNA-seq) data and multi-soure biological knowledge fusion.

Unlike conventional GRN inference methods that rely solely on transcriptomic information, scMGF-GRNS integrates:

* Single-cell transcriptomic profiles
* Gene Ontology (GO) hierarchical relationships 
* gene sequence 
* Pseudotime information

through a multi-source feature fusion architecture.

The framework combines:

* Denoising Autoencoder (DAE)
* Graph attention netowrk(GAT)
* pretrained DNABERT-2
* Bidirectional Gated Recurrent Unit (Bi-GRU)
* Gated Multi-Head Attention (Gated MHA)

to extract robust regulatory representations and accurately predict transcription factor (TF)–target gene interactions.

## Benchmark Datasets

The framework was evaluated on seven widely used scRNA-seq datasets:

| Species | Datasets |
|----------|----------|
| Human | hESC, hHep |
| Mouse | mESC, mDC, mHSC-E, mESC, mHSC-GM |

Ground-truth networks include:

* STRING
* Non-specific ChIP-seq
* Cell-type-specific ChIP-seq
* LOF/GOF

## Installation

```bash
git clone https://github.com/zhanglab57/scMGF-GRNS.git

cd scMGF-GRNS

conda create -n scmgf python=3.10

conda deactivate
conda activate scmgf

#Make sure those packages are installed before reproducing the code
numpy
scipy
torch
pytorch-lightning
scikit-learn
pandas
tensorboardX
pytorchtools
torch-geometric
torchvision

#You can download them using the following command
pip install -r requirements.txt

#installation
pip install torch torchvision
pip install torch-geometric
pip install numpy pandas scikit-learn
```

## Directory Structure

```text
scMGF-GRNS/
│
├── dataset/
│   ├── scRNA-seq/
│   ├── GO hierarchical and annotation/
│   ├── gene sequence/
│   └── pseudotime/
│
├── GatedAttention.py/
│   ├── GLU
│   ├── Gated multi-head attention
│   └── ERetNet
│ 
├── go_embedding.py/      #go DAG constrcution and graph enhancement
│   ├── GAT             
│ 
├── SequenceEncoder.py/
│   ├── DNABERT-2
│ 
├── main.py/
│   ├── DAE
│   ├── Bi-GRU
│   └── scMGF_GRNS
│
├── preproceessing.ipynb   # To generate go adjacent matrices
├── sample.py              # To generate negative samples
├── requirements.txt       
└── README.md
```

### Data preprocessing
The DNA sequence embedding need to be processed by SequenceEncoder.py and saved to be .npy file
```
python SequenceEncoder.py 
```

We have provided the full dataset on the STRING ground truth network, you can run the code without any preprocessing. For Non-specific  ChIP-seq、Cell-type-specific ChIP-seq、LOF/GOF, The go hierarchy and adjacency matrices for the three Gene Ontology (GO) categories (GO:BP, GO:MF, and GO:CC) are omitted from this repository due to their large size. Users can regenerate these matrices using the provided preprocessing.ipynb notebook.


### Train and test the scMGF-GRNS on seven benchmarking scRNAseq, current dataset is mESC-500

```bash
python main.py
```
### To apply scMGF-GRNS to other datasets or your own data, you can reset the training data path to different scRNA-seq ,biological priors and ground truth network by replace the dataset name mESC to another one, for example hESC, mDC.
```
#multi-source data
expression=pd.read_csv('./dataset/scRNA-Seq/STRING Dataset/mESC/TFs+500/ExpressionData.csv')
pesudo_time = pd.read_csv('./dataset/expression_orderd_mESC.csv')
bp = pd.read_csv('./dataset/scRNA-Seq/STRING Dataset/mESC/TFs+500/mESC_BP.csv')

#ground truth network
network_TF_target = pd.read_csv('./dataset/scRNA-Seq/STRING Dataset/mESC/TFs+500/AllPair.csv')
```
   

### Output

The model generates:

* TF-gene interaction scores
* Predicted GRNs
* Cell-type-specific GRNs
* Evaluation metrics (AUROC, AUPRC)

---

## Results

scMGF-GRNS consistently outperforms existing GRN inference methods, including:

* GENIE3
* GRNBoost2
* SCENIC
* DeepSEM
* DeepRIG
* GENELink
* scMGATGRN

Across multiple benchmark datasets, scMGF-GRNS achieves superior AUROC and AUPRC scores while maintaining robustness under highly sparse single-cell conditions.

---

Project Homepage:

https://github.com/zhanglab57/scMGFGRN

---

## License

MIT License

```
```
