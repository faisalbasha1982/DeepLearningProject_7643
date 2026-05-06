# GraphShield: Explainable Temporal Graph Neural Networks for Cryptocurrency Fraud Detection

CS 7643 Final Project — Georgia Institute of Technology

**Authors:** Faisal Bin Basha · Mewael Hailu

This project investigates whether graph neural networks — particularly temporal variants — can detect illicit Bitcoin transactions in the Elliptic dataset, and whether their decisions can be explained well enough for regulatory use. We compare static GNNs (GCN, GraphSAGE) against temporal GNNs (EvolveGCN, TGN) under a strict temporal evaluation protocol, then apply GNNExplainer and PGExplainer with quantitative faithfulness metrics on top of the trained models.

See **[CS7643_Final_Project_Report.pdf](CS7643_Final_Project_Report.pdf)** for the full writeup.

---

## Repository structure

DeepLearningProject_7643/
├── faisalbasha_codebase_graphshield/   # Faisal: EvolveGCN + Explainability framework
├── mewael_codebase_graphshield/        # Mewael: Data preprocessing + Baselines + TGN
├── CS7643_Final_Project_Report.pdf     # Final report (the deliverable)
├── ProjectProposal.pdf                 # Original project proposal
├── ProjectProposal.docx                # Editable proposal source
└── README.md                           # This file


Each codebase folder is self-contained — its notebooks load data from a sibling `data/raw/` directory (not committed, see Setup below).

## Headline Results

| Model     | Val F1 | Test F1 | ROC-AUC | PR-AUC |
|-----------|--------|---------|---------|--------|
| GraphSAGE | 0.862  | **0.540** | **0.884** | 0.524  |
| TGN       | 0.807  | 0.516   | 0.849   | **0.529**  |
| GCN       | 0.701  | 0.452   | 0.844   | 0.347  |
| EvolveGCN | 0.553  | 0.284   | 0.831   | 0.247  |
| LogReg    | 0.557  | 0.253   | 0.861   | 0.213  |

**Key findings:**
- GraphSAGE was the strongest overall classifier on illicit-class F1.
- TGN was the strongest temporal model and best on PR-AUC.
- Illicit predictions require 2.4–6.2× more important edges than licit ones in GNNExplainer outputs — empirical signature of money-laundering layering patterns.
- GCN explanations are dramatically more faithful than GraphSAGE explanations (Fidelity+ 0.544 vs 0.0001), revealing a precision-vs-explainability trade-off.
- EvolveGCN is incompatible with standard edge-mask explainers — a methodological negative result for regulated deployments.

## Setup

### Dataset

The Elliptic Bitcoin Dataset is **not committed** to this repo (657 MB, redistribution restrictions). Download from Kaggle:

```bash
kaggle datasets download -d ellipticco/elliptic-data-set
```

Unzip into `faisalbasha_codebase_graphshield/data/raw/`. The expected layout is:

### Environment

- Python 3.10+
- PyTorch 2.x
- PyTorch Geometric
- PyTorch Geometric Temporal
- scikit-learn, pandas, numpy, matplotlib, networkx

notebooks faisal_Projet_DL_7643.ipynb & explainer_framework.ipynb were developed and run on Google Colab (NVIDIA T4 GPU, free tier sufficient).

### Running

Open the notebooks directly in Colab (mount Drive, set `PROJECT_DIR` to the codebase folder location). Each notebook is self-contained and saves checkpoints + metrics to `<codebase_folder>/results/`.

## Methods

We trained and evaluated five models under a three-way temporal split (train: timesteps 1–30, validation: 31–34, test: 35–49):

- **Logistic Regression** — non-graph baseline on raw node features.
- **GCN** (Kipf & Welling, 2017) — static graph convolution.
- **GraphSAGE** (Hamilton et al., 2017) — inductive sampling-based aggregation.
- **EvolveGCN** (Pareja et al., 2020) — discrete-time temporal GNN with weight-evolution RNN.
- **TGN** (Rossi et al., 2020) — continuous-time temporal GNN with per-node memory modules.

We applied two explainability methods to the trained models:

- **GNNExplainer** (Ying et al., 2019) — instance-level edge mask optimization.
- **PGExplainer** (Luo et al., 2020) — parameterized explainer with shared MLP.

Faithfulness metrics (Fidelity+, Fidelity−, sparsity) follow Yuan et al. (2022).

## Contributions

| Contributor | Areas |
|---|---|
| Mewael Hailu | Data preprocessing pipeline, baseline models (LogReg/GCN/GraphSAGE), TGN implementation |
| Faisal Bin Basha | Data preprocessing, EvolveGCN implementation, Explainability framework (GNNExplainer + PGExplainer + faithfulness metrics) |

## References

Key references — see `CS7643_Final_Project_Report.pdf` for the full bibliography.

- Hamilton, W. L., Ying, R., & Leskovec, J. (2017). *Inductive representation learning on large graphs.* NeurIPS.
- Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks.* ICLR.
- Luo, D. et al. (2020). *Parameterized explainer for graph neural network.* NeurIPS.
- Pareja, A. et al. (2020). *EvolveGCN: Evolving graph convolutional networks for dynamic graphs.* AAAI.
- Rossi, E. et al. (2020). *Temporal graph networks for deep learning on dynamic graphs.* arXiv:2006.10637.
- Weber, M. et al. (2019). *Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics.* arXiv:1908.02591.
- Ying, R. et al. (2019). *GNNExplainer: Generating explanations for graph neural networks.* NeurIPS.
- Yuan, H., Yu, H., Gui, S., & Ji, S. (2022). *Explainability in graph neural networks: A taxonomic survey.* IEEE TPAMI.

## License

Course project for CS 7643 (Deep Learning), Georgia Tech. Code is provided as-is for academic reference.