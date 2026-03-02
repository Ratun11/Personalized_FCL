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

Personalized_FCL/
│
├── standalone.py # Standalone variational sensing optimization
├── FCL_IID.py # Homogeneous federated learning (IID clients)
├── FCL_NonIID.py # Heterogeneous federated learning (Non-IID clients)
├── demo.ipynb # Demonstration notebook
├── Ratun11_Demo.ipynb # Additional demo notebook
├── metadata.json # Experiment configuration metadata
├── requirements.txt # Python dependencies
└── README.md # Documentation
