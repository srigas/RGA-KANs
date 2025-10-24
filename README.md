# Introduction

This repository contains the code required to produce the experimental results and plots for the paper "[Towards Deep Physics-Informed Kolmogorov–Arnold Networks](TODO)".


# Getting Started

After cloning the repository,

```bash
git clone https://github.com/srigas/RGA-KANs.git rgakan
cd rgakan
```

create a Python virtual environment, activate it and install all dependencies:

```bash
python3 -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
pip3 install -r requirements.txt
```

Then launch JupyterLab:

```bash
jupyter lab
```

Open and run the notebook `auxiliaries/Data Preparation.ipynb` to create the `data` directory and populate it with the reference solutions for the PDEs studied in the paper.

Then open the notebooks in the parent directory in order (`1.*.ipynb` → `12.*.ipynb`) to reproduce all experiments and generate the data (`results` directory) and plots (`plots` directory) presented in the paper.


# Citation

If the code and/or results presented in this work helped you for your own work, please cite our work as:

```
@misc{rgakan, 
	title = {Towards deep physics-informed {K}olmogorov–{A}rnold networks}, 
	author = {Spyros Rigas and Fotios Anagnostopoulos and Michalis Papachristou and Georgios Alexandridis}, 
	year = {2025}, 
	eprint = {TODO}, 
	archivePrefix = {arXiv}, 
	primaryClass = {cs.LG}, 
	url = {TODO}
}
```
