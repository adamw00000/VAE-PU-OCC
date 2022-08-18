# VAE-PU+OCC: One-Class Classification Approach to Variational Learning from Biased Positive Unlabeled Data.

Official implementation for VAE-PU+OCC. Will not be actively maintained.

# How to run

1. Install required Python dependencies (check out `requirements.txt`), using virtual environment is recommended.
2. Download preprocessed datasets from [link](https://drive.google.com/file/d/1yXvTtftD0PTzYm2jBDZ-gBX2Jk5AaGuy/view?usp=sharing) and place `data` folder in the main directory.
3. (If using SAR-EM, optional) Follow the steps in SAR-EM readme in order to install the library, it is located under `external/SAR-EM` directory.
4. You can run the experiments form the paper using `main-paper.py` script, e.g. `python .\main-paper.py --method "VAE-PU+OCC" --dataset "MNIST 3v5"`

# Repository structure

## VAE-PU+OCC

Main module (VAE-PU and VAE-PU+OCC) implementation is contained in `vae_pu_occ` directory, with `model.py` containing model code and two trainer files controlling the training process.

## Results

Experiment results will be found in `result` directory. They are separated by dataset, label frequency and experiment number. Raw data can be processed using `parse-vae-pu-results.py` script in the repository root.

## Configuration

Common configuration file is located under `configs/base_config.json`. These settings are applied in each training scenario. `configs/datasets` directory contains settings specific for each dataset, and `configs/methods` - to each method. `configs/custom_settings.json` overwrites settings contained in other files, if needed.

## External methods

External methods used during experiments are contained in `external` directory, including:
- $A^3$ algorithm ([original link](https://github.com/Fraunhofer-AISEC/A3))
- CCV p-values calculation ([original link](https://github.com/msesia/conditional-conformal-pvalues))
- custom ECOD implementation (modified from [original link](https://github.com/yzhao062/pyod/blob/master/pyod/models/ecod.py))
- slightly modified SAR-EM code (numerical fixes, [original link](https://github.com/ML-KULeuven/SAR-PU))
- several simple wrappers.
