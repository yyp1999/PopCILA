# 🧬 PopCILA

**PopCILA** is a multimodal computational framework designed to decompose phenotype-associated intercellular signaling. Guided by diverse phenotypes, PopCILA identifies phenotype-associated signaling at population scale—beginning with ligand–receptor interactions and extendable to downstream transcriptional cascades—and then projects these signals onto single-cell or spatial data to pinpoint specific cellular actors and tissue niches and to resolve intercellular signaling events that underlie phenotypic variation.

PopCILA supports multiple phenotype types, including **binary**, **continuous**, **ordinal**, and **right-censored survival** outcomes.

![Overview](https://github.com/yyp1999/PopCILA/blob/main/PopCILA.jpg)


## 🧩 System requirements

PopCILA requires Python 3.8 or later and is distributed as
a platform-independent Python package.

PopCILA has been successfully tested on Ubuntu 24.04.4 LTS, Windows 10 (64-bit) and macOS 26.5 (arm64). Typical installation time is approximately 5-15 minutes on a standard desktop computer with a stable internet connection.

The principal dependency versions used in this tested environment were:

- pandas 2.2.3
- NumPy 1.26.4
- Matplotlib 3.10.1
- scikit-learn 1.5.2
- SciPy 1.15.2
- seaborn 0.13.2
- statsmodels 0.14.4
- Plotly 6.3.0
- IPython 8.30.0
- Scanpy 1.11.0
- qnorm 0.9.0
- PyTorch 2.6.0
- Numba 0.61.0
- AnnData 0.11.4
- openchord 0.1.7
- adjustText 1.3.0
- PyComplexHeatmap 1.8.2
- libpysal 4.13.0
- Squidpy 1.6.5
- esda 2.7.0
- tqdm 4.67.1
- NetworkX 3.4.2

The `openchord` dependency is optional and only required for chord diagrams.

No non-standard hardware is required. A GPU is not required.

## 🔧 Installation


PopCILA is implemented in Python 3 and can be installed via:

```bash
pip install popcila
```

To use `PopCILA.cci_chord`, install the optional chord dependency:

```bash
pip install popcila[chord]
```

## 📘 Usage Guide

This repository provides two end-to-end tutorials (two tracks):

### 📜 Tutorial Links

| Track | Notebook Link |
|----------------|----------------|
| Single-cell RNA-seq         | [🔗 View Tutorial](https://github.com/yyp1999/PopCILA/blob/main/tutorial/PopCILA_for_Single-Cell_RNA-seq.ipynb) |
| Spatial Transcriptomics         |  [🔗 View Tutorial](https://github.com/yyp1999/PopCILA/blob/main/tutorial/PopCILA_for_Spatial_Transcriptomics.ipynb) |
> 💡 Tip: The notebooks are written to be self-contained. Follow the sections in order within each notebook.

## 📦 Toy Dataset

You can download the example dataset for tutorials here: [https://drive.google.com/drive/folders/17RgFhzNYNzFHYUq1Oo0bjhOZDNkfUtff?usp=sharing](https://drive.google.com/drive/folders/17RgFhzNYNzFHYUq1Oo0bjhOZDNkfUtff?usp=sharing)

## ✨ Citation

If you use **PopCILA** in your research, please consider citing our paper (coming soon).



## 📮 Contact

- Maintainer: Youpeng Yang
- Email: yangyp33@alumni.sysu.edu.cn
- Issues: Please open a GitHub Issue (recommended)
