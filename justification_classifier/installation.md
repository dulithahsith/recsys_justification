# Installation Guide

This document outlines the steps required to set up the environment for this project using Conda and Python 3.10. All dependencies are listed in `requirements.txt`.

## Prerequisites

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution)
- Python 3.10 (installed via Conda)

## Setup Instructions

### 1. Create and activate the Conda environment

```bash
conda create -n classifier_env python=3.10
conda activate classifier_env

pip install -r requirements.txt

from git bash -> ./run_xgb.sh // To run xgb_bow classifier
```
