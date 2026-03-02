# Federated Circuit Learning (FCL) for Quantum Sensor Networks

This repository provides an implementation of **Federated Circuit Learning (FCL)** for **variational quantum sensing** using parameterized quantum circuits. The framework enables multiple quantum sensors to collaboratively optimize sensing circuits while preserving robustness under homogeneous and heterogeneous noise conditions.

The implementation supports:

- Standalone variational quantum sensing optimization
- Homogeneous federated learning (IID clients)
- Heterogeneous federated learning (Non-IID clients)
- Personalized federated circuit learning
- Reproducible experiments and visualization

---

## Repository structure

```
Personalized_FCL/
│
├── standalone.py        # Standalone variational sensing optimization
├── FCL_IID.py          # Homogeneous federated learning (IID clients)
├── FCL_NonIID.py       # Heterogeneous federated learning (Non-IID clients)
├── demo.ipynb          # Demonstration notebook
├── Ratun11_Demo.ipynb  # Additional demo notebook
├── metadata.json       # Experiment configuration metadata
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

---

## Installation

### Create virtual environment (recommended)

```bash
python -m venv venv
```

### Activate environment

Linux / Mac

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Standalone optimization

Runs variational sensing optimization for a single quantum sensor:

```bash
python standalone.py
```

---

### Homogeneous Federated Learning (IID)

Runs federated learning where all clients have identical noise:

```bash
python FCL_IID.py
```

---

### Heterogeneous Federated Learning (Non-IID)

Runs federated learning with heterogeneous client noise:

```bash
python FCL_NonIID.py
```

---

### Demo notebook

Interactive demonstration:

```
demo.ipynb
```

---

## Method overview

The sensing protocol is based on **Ramsey spectroscopy** using parameterized quantum circuits.

Each client performs:

- Local quantum circuit optimization
- Cost minimization using Fisher information / CRB
- Local gradient-based updates

The server performs:

- Model aggregation
- Global parameter synchronization

Personalized FCL allows adaptation to heterogeneous sensor noise.

---

## Dependencies

Main libraries:

- PennyLane
- NumPy
- Matplotlib
- Pandas

Install using:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

Typical configuration:

- Number of qubits: 3
- Number of layers: 2
- Clients: 3–10
- Optimization rounds: 20
- Learning rate: 0.1

---

## Applications

This framework applies to:

- Quantum sensor networks
- Distributed quantum sensing
- Variational quantum metrology
- Noise-robust quantum sensing
- Federated quantum learning

---

## Citation

If you use this code, please cite the corresponding research work.

---

## License

This project is intended for academic and research use.
