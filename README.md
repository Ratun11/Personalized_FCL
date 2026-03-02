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

## Running Experiments (Standalone, Homogeneous FL, Heterogeneous FL)

This project supports three experiment modes: (1) standalone
single-sensor optimization, (2) homogeneous federated learning (IID
clients), and (3) heterogeneous federated learning (Non-IID clients)
with both FedAvg and personalized FCL (pFCL).

> **Note:** In this repository, scripts are currently located in the
> root directory (e.g., `standalone.py`, `FCL_IID.py`,
> `FCL_NonIID.py`).\
> If you later move them under `src/`, update the commands accordingly.

------------------------------------------------------------------------

### 1) Standalone (single sensor)

Runs variational optimization for a single sensing device.

**Command (root structure):**

``` bash
python standalone.py
```

**Command (if using src/ structure):**

``` bash
python src/standalone.py \
  --shots 1000 \
  --qubits 3 \
  --layers 2 \
  --gamma 0.2 \
  --iters 20 \
  --lr 0.1 \
  --seed 395
```

**Output:**

    results/standalone_results.csv

Contains columns like:

    step, cost, runtime_sec, ...

------------------------------------------------------------------------

### 2) Homogeneous FL (IID, K clients)

All clients share the same noise setting (e.g., γ = 0.2). FedAvg
aggregates local updates.

**Command (root structure):**

``` bash
python FCL_IID.py
```

**Command (if using src/ structure):**

``` bash
python src/fcl_iid.py \
  --clients 5 \
  --rounds 20 \
  --local_steps 1 \
  --gamma 0.2 \
  --shots 1000 \
  --qubits 3 \
  --layers 2 \
  --lr 0.1 \
  --seed 395
```

**Output:**

    results/fcl_iid_results.csv

Contains columns like:

    round, global_cost, avg_client_cost, ...

------------------------------------------------------------------------

### 3) Heterogeneous FL (Non-IID, K clients)

Clients experience different noise conditions (e.g., γ_k varies across
sensors). We compare:

-   Heterogeneous FedAvg (baseline)
-   Personalized FCL (pFCL)

**Command (root structure):**

``` bash
python FCL_NonIID.py
```

**Command (if using src/ structure):**

``` bash
python src/fcl_non_iid.py \
  --clients 5 \
  --rounds 20 \
  --local_steps 1 \
  --gamma_min 0.15 \
  --gamma_max 0.35 \
  --shots 1000 \
  --qubits 3 \
  --layers 2 \
  --lr 0.1 \
  --seed 395
```

**Output:**

    results/fcl_non_iid_results.csv

Contains columns like:

    round
    hetero_fedavg_cost
    pfcl_cost

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
