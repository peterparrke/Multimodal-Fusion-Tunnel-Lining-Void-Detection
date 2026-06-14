# Multimodal NDT Fusion for Tunnel Lining Void Detection

This repository contains the research code for multimodal nondestructive testing (NDT)-based internal void detection in tunnel lining structures. The current implementation combines impact/knock signal processing, ground-penetrating radar (GPR) feature extraction, autoencoder-based anomaly scoring, and Bayesian pixel-level fusion for tunnel lining void mapping.

This repository is released to support academic transparency, methodological traceability, and further research on multimodal NDT fusion for tunnel lining inspection.

## Main Features

* Impact/knock signal preprocessing and feature extraction.
* GPR trace feature extraction from raw matrix data.
* Transformer Autoencoder, CNN Autoencoder, and DNN Autoencoder baselines.
* Intra-modal and cross-modal Bayesian fusion.
* Threshold-based mask generation and comparison with ground-truth masks.
* Figure generation utilities for flowcharts, mask comparisons, and runtime breakdowns.
* Probabilistic void mapping for tunnel lining inspection.

## Repository Structure

```text
.
|-- final_TAE_Autoencoder_withGPR_transformer_deconv.py
|-- GPR_data/
|   |-- GPR_plot_A_scan.m
|   |-- GPR_plot_cscan.m
|   `-- GPR_reader_resample.m
|-- Impact/
|   |-- hammer_deconvolution.m
|   |-- sdcq_time_fre.m
|   `-- sdcq_time_fre_deconv.m
|-- main_TAE_result/
|   |-- make_tae_flowchart.py
|   |-- make_threshold_decision.py
|   |-- plot_runtime_breakdown_pie.py
|   |-- V2_make_threshold_decision.py
|   |-- CNNFinal_Mask.csv
|   |-- DNNFinal_Mask.csv
|   |-- TAEFinal_Mask.csv
|   |-- Groundtruth_4.xlsx
|   `-- runtime_breakdown_measured.csv
|-- requirements.txt
`-- .gitignore
```

Large generated result directories, intermediate arrays, model checkpoints, and figures are intentionally excluded by `.gitignore`. The Python scripts inside `main_TAE_result/` are kept because they are part of the result generation workflow.

## Environment

The code is written for Python 3.10 or later.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation can depend on the CPU/GPU and CUDA version. If the default `pip install -r requirements.txt` does not install the desired PyTorch build, install the matching PyTorch package first, then install the remaining requirements.

## Data and Path Configuration

The main script currently uses absolute Windows paths from the original experiment machine. Before running, update the configuration blocks near the top of:

```text
final_TAE_Autoencoder_withGPR_transformer_deconv.py
```

Important configuration entries include:

* `PROJECT_CFG["gt_path"]`
* `PATH_CFG["knock_root"]`
* `PATH_CFG["gpr_root"]`
* `PATH_CFG["export_root"]`
* `RUN_SWITCH`
* `EXPERIMENT_CFG`

The plotting and post-processing scripts in `main_TAE_result/` also contain a `ROOT` variable that should point to the local result directory.

Raw experimental data are not included in this repository. If reproducibility from raw data is required, please prepare the corresponding impact/knock and GPR data folders and update the path configuration accordingly.

## Running the Main Pipeline

After updating the paths, run the complete pipeline:

```powershell
python final_TAE_Autoencoder_withGPR_transformer_deconv.py
```

To run from intermediate files and skip raw TXT to NPY conversion:

```powershell
$env:PIPELINE_RUN_MODE="profile_from_intermediate"
python final_TAE_Autoencoder_withGPR_transformer_deconv.py
```

To force the full raw-data workflow:

```powershell
$env:PIPELINE_RUN_MODE="profile_full_raw"
python final_TAE_Autoencoder_withGPR_transformer_deconv.py
```

The main workflow includes:

1. Impact/knock TXT to NPY conversion.
2. Impact/knock feature extraction.
3. Feature matrix transformation and stacking.
4. GPR feature extraction and stacking.
5. Autoencoder training and inference.
6. Bayesian intra-modal and cross-modal fusion.
7. Runtime profiling and result export.

## Figure and Result Scripts

The scripts in `main_TAE_result/` are used to regenerate post-processing outputs and figures:

```powershell
python main_TAE_result\make_tae_flowchart.py
python main_TAE_result\make_threshold_decision.py
python main_TAE_result\V2_make_threshold_decision.py
python main_TAE_result\plot_runtime_breakdown_pie.py
```

Update each script's `ROOT`, input file paths, and output directory before use.

## MATLAB Utilities

The `GPR_data/` and `Impact/` folders contain MATLAB utilities for reading, resampling, plotting, and deconvolving GPR and impact data. These files are included for traceability with the original experimental workflow.

## Reproducibility Notes

* The default random seed is set in the Python code through `PROJECT_CFG["base_seed"]`.
* Some generated outputs are not tracked by Git to keep the repository small.
* If exact figures are needed, restore the original result files or regenerate them using the provided scripts.
* For publication or external reuse, record the operating system, Python version, PyTorch version, and GPU/CPU details used for the reported results.
* The reported results may depend on the selected random seed, software environment, and hardware configuration.

## Data Availability

Raw experimental data are not included in this repository. The repository mainly provides the implementation of the multimodal NDT processing, anomaly scoring, Bayesian fusion, and result-generation workflow.

## License

No license is provided at this stage. All rights are reserved by the author unless a license file is added in the future.
