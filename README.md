# Introduction

This repository contains the code required to produce the experimental results and plots for the paper "[Training Deep Physics-Informed Kolmogorov–Arnold Networks](https://www.sciencedirect.com/science/article/pii/S0045782526000356)".


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

Open and run the notebook `auxiliaries/Data Preparation.ipynb` to create the `data` directory and populate it with the reference solutions for the PDEs studied in the paper. Do not forget to also run the `auxiliaries/Navier-Stokes Data Generator.ipynb` file to create the data file for the Navier-Stokes equation.

Then open the notebooks in the parent directory in order (`1.*.ipynb` → `14.*.ipynb`) to reproduce all experiments and generate the data (`results` directory) and plots (`plots` directory) presented in the paper.


# Citation

If the code and/or results presented in this work helped you for your own work, please cite our work as:

```
@article{rgakan,
	title = {Training deep physics-informed Kolmogorov–Arnold networks},
	author = {Spyros Rigas and Fotios Anagnostopoulos and Michalis Papachristou and Georgios Alexandridis},
	journal = {Computer Methods in Applied Mechanics and Engineering},
	volume = {452},
	pages = {118761},
	year = {2026},
	issn = {0045-7825},
	doi = {https://doi.org/10.1016/j.cma.2026.118761},
}
```
