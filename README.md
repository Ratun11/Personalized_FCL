# Federated Circuit Learning (FCL) for Quantum Sensor Networks

This repository provides an implementation of **Federated Circuit
Learning (FCL)** for **variational quantum sensing** using parameterized
quantum circuits. The framework enables multiple quantum sensors to
collaboratively optimize sensing circuits while preserving robustness
under homogeneous and heterogeneous noise conditions.

The implementation supports:

-   Standalone variational quantum sensing optimization
-   Homogeneous federated learning (IID clients)
-   Heterogeneous federated learning (Non-IID clients)
-   Personalized federated circuit learning (pFCL)
-   Ablation studies for qubits and circuit depth
-   Reproducible experiments and visualization

------------------------------------------------------------------------

## Repository structure

```
Personalized_FCL/
│
├── standalone.py        # Standalone variational sensing optimization
├── FCL_IID.py          # Homogeneous federated learning (IID clients)
├── FCL_NonIID.py       # Heterogeneous federated learning (Non-IID clients)
├── demo.ipynb          # Demonstration notebook
├── metadata.json       # Experiment configuration metadata
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

------------------------------------------------------------------------

## Installation

### Create virtual environment (recommended)

``` bash
python -m venv venv
```

### Activate environment

Linux / Mac

``` bash
source venv/bin/activate
```

Windows

``` bash
venv\Scripts\activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

### Standalone optimization

Runs variational sensing optimization for a single quantum sensor:

``` bash
python standalone.py
```

Output:

    results/standalone_results.csv

------------------------------------------------------------------------

### Homogeneous Federated Learning (IID)

Runs federated learning where all clients have identical noise:

``` bash
python FCL_IID.py
```

Output:

    results/fcl_iid_results.csv

------------------------------------------------------------------------

### Heterogeneous Federated Learning (Non-IID)

Runs federated learning with heterogeneous client noise:

``` bash
python FCL_NonIID.py
```

Output:

    results/fcl_non_iid_results.csv

Contains:

-   round
-   hetero_fedavg_cost
-   pfcl_cost

------------------------------------------------------------------------

## Ablation Study

Recommended parameter sweep:

-   Qubits: 1, 2, 3, 4
-   Layers: 1, 2, 3, 4

Best observed configuration:

-   Qubits: 3
-   Layers: 2
-   CRB: 1.5185

------------------------------------------------------------------------

## Method overview

Based on Ramsey spectroscopy and Fisher information optimization.

Supports:

-   Distributed sensing
-   Federated optimization
-   Personalized sensing protocols

------------------------------------------------------------------------

## Applications

-   Quantum sensor networks
-   Distributed quantum sensing
-   Variational quantum metrology
-   Federated quantum learning

------------------------------------------------------------------------

## License

Academic and research use.
