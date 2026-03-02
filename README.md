# Federated Circuit Learning (FCL) for Quantum Sensor Networks

This repository provides an implementation of **Federated Circuit Learning (FCL)** for **variational quantum sensing** using parameterized quantum circuits. The framework enables multiple quantum sensors to collaboratively optimize their sensing circuits while preserving local autonomy and robustness to heterogeneous noise.

The implementation includes:

- Standalone variational quantum sensing optimization
- Homogeneous federated learning (IID clients)
- Heterogeneous federated learning (Non-IID clients)
- Personalized federated circuit learning for heterogeneous quantum sensor networks

The sensing protocol is based on **Ramsey spectroscopy**, and optimization is performed using a Fisher-information-based objective (Cramér–Rao bound minimization).

---

## Repository structure

| File | Description |
|-----|-------------|
| standalone.py | Standalone variational sensing optimization |
| FCL_IID.py | Homogeneous federated learning |
| FCL_NonIID.py | Heterogeneous federated learning |
| demo.ipynb | Demo notebook |
| Ratun11_Demo.ipynb | Additional demo |
| metadata.json | Experiment configuration |
| requirements.txt | Dependencies |
| README.md | Documentation |

---

## Installation

### Create virtual environment (recommended)

```bash
python -m venv venv

### Activate environment

Linux / Mac

```bash
source venv/bin/activate

Windows

```bash
venv\Scripts\activate

Install dependencies

```bash
pip install -r requirements.txt

---

## Usage

### 1. Standalone variational sensing

Runs optimization for a single quantum sensor:

```bash
python standalone.py

This performs:

- Circuit initialization
- Variational optimization
- Cost minimization (Cramér–Rao bound)
