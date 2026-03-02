
import argparse
import itertools
import time
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import pennylane as qml

# ============================================================
# Quantum Sensing + (Federated) training harness with:
#  - legacy objective (matches tutorial scale for n_qubits=3)
#  - generic objective (works for any n_qubits)
# Supports ablations over shots, qubits, layers, clients, noise.
# CSV outputs are saved in the current directory by default.
# ============================================================

def parse_csv_list(s: str, cast=float) -> List:
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]

def parse_vec(s: str, n: int) -> np.ndarray:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]
    if len(vals) != n:
        raise ValueError(f"--phi must have exactly {n} comma-separated values (got {len(vals)}).")
    return np.array(vals, dtype=float)

def make_device(device: str, wires: int, shots: Optional[int]):
    if shots is not None and shots <= 0:
        shots = None
    return qml.device(device, wires=wires, shots=shots)

# ---------- Legacy Jacobian (tutorial) for n_qubits=3 -> 2D target ----------
def omega_matrix():
    omega = np.exp((-1j * 2.0 * np.pi) / 3.0)
    Omega = np.array([[1, 1, 1],
                      [1, omega, omega**2]], dtype=np.complex128) / np.sqrt(3.0)
    return Omega

def jacobian_fourier(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3,)
    Omega = omega_matrix()  # 2x3 complex
    a = Omega @ phi         # 2 complex
    # d |a_j|^2 / d phi_l = 2 Re(conj(a_j) * Omega_{j,l})
    J_T = np.stack([2.0 * np.real(np.conj(a[j]) * Omega[j, :]) for j in range(2)], axis=0)  # 2x3
    return J_T.T  # 3x2

# ---------- Circuit ----------
def build_qnode(dev, n_qubits: int, layers: int):
    wires = list(range(n_qubits))

    def probe(w_probe):
        qml.StronglyEntanglingLayers(w_probe, wires=wires)

    def meas(w_meas):
        for i in range(n_qubits):
            qml.Rot(w_meas[i,0], w_meas[i,1], w_meas[i,2], wires=i)

    def encoding(phi, gamma):
        for i in range(n_qubits):
            qml.RZ(phi[i], wires=i)
            qml.PhaseDamping(gamma, wires=i)

    @qml.qnode(dev)
    def qnode(w_probe, w_meas, phi, gamma=0.0):
        probe(w_probe)
        encoding(phi, gamma)
        meas(w_meas)
        return qml.probs(wires=wires)

    return qnode

# ---------- CFIM ----------
def cfim(qnode, w_probe, w_meas, phi: np.ndarray, gamma: float, p_clip=1e-12):
    phi = np.asarray(phi, dtype=float)
    p = qnode(w_probe, w_meas, phi, gamma=gamma)
    p = qml.math.clip(p, p_clip, 1.0)

    dps = []
    for idx in range(len(phi)):
        shift = np.zeros_like(phi)
        shift[idx] = np.pi / 2.0
        plus = qnode(w_probe, w_meas, phi + shift, gamma=gamma)
        minus = qnode(w_probe, w_meas, phi - shift, gamma=gamma)
        dps.append(0.5 * (plus - minus))

    n = len(phi)
    entries = []
    for i in range(n):
        for j in range(n):
            entries.append(qml.math.sum(dps[i] * dps[j] / p))
    return qml.math.reshape(qml.math.stack(entries), (n, n))

# ---------- Objectives ----------
def sensing_cost_generic(qnode, w_probe, w_meas, phi: np.ndarray, gamma: float, epsilon=1e-10):
    F = cfim(qnode, w_probe, w_meas, phi, gamma)
    n = len(phi)
    M = F + qml.math.eye(n) * epsilon
    return qml.math.trace(qml.math.linalg.inv(M))

def sensing_cost_legacy(qnode, w_probe, w_meas, phi: np.ndarray, gamma: float, epsilon=1e-10):
    """
    Matches the tutorial-style objective scale:
      Tr( W * (J^T F J + eps I)^-1 ), with W = I_2 and J from Fourier mapping.
    Only valid when n_qubits == 3.
    """
    phi = np.asarray(phi, dtype=float).reshape(3,)
    J = jacobian_fourier(phi)   # 3x2
    W = np.eye(2)
    F = cfim(qnode, w_probe, w_meas, phi, gamma)  # 3x3
    M = J.T @ F @ J + qml.math.eye(2) * epsilon   # 2x2
    return qml.math.trace(W @ qml.math.linalg.inv(M))

def pick_objective(objective: str, n_qubits: int):
    obj = (objective or "legacy").lower()
    if obj == "legacy":
        return "legacy" if n_qubits == 3 else "generic"
    return "generic"

# ---------- Param helpers ----------
def flatten_params(w_probe, w_meas) -> qml.numpy.tensor:
    return qml.numpy.concatenate([qml.numpy.ravel(w_probe), qml.numpy.ravel(w_meas)])

def unflatten_params(theta, n_qubits: int, layers: int):
    theta = qml.numpy.array(theta, requires_grad=True)
    probe_size = layers * n_qubits * 3
    meas_size = n_qubits * 3
    if len(theta) != probe_size + meas_size:
        raise ValueError("Parameter vector has wrong size for the given (layers, n_qubits).")
    w_probe = qml.numpy.reshape(theta[:probe_size], (layers, n_qubits, 3))
    w_meas  = qml.numpy.reshape(theta[probe_size:], (n_qubits, 3))
    return w_probe, w_meas

def save_csv(path: str, df):
    df.to_csv(path, index=False)

def init_theta(seed: int, n_qubits: int, layers: int):
    rng = np.random.default_rng(seed)
    probe = rng.uniform(0, 2*np.pi, size=(layers, n_qubits, 3))
    meas  = rng.uniform(0, 2*np.pi, size=(n_qubits, 3))
    return qml.numpy.array(flatten_params(qml.numpy.array(probe), qml.numpy.array(meas)), requires_grad=True)

def config_product(**grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def run_one(device: str, shots: Optional[int], seed: int, n_qubits: int, layers: int,
            iters: int, lr: float, gamma: float, phi: np.ndarray, objective: str):
    dev = make_device(device, wires=n_qubits, shots=shots)
    qnode = build_qnode(dev, n_qubits=n_qubits, layers=layers)

    theta = init_theta(seed, n_qubits, layers)
    opt = qml.AdagradOptimizer(stepsize=lr)

    obj_kind = pick_objective(objective, n_qubits)

    def obj(th):
        w_probe, w_meas = unflatten_params(th, n_qubits, layers)
        if obj_kind == "legacy":
            return sensing_cost_legacy(qnode, w_probe, w_meas, phi=phi, gamma=gamma)
        return sensing_cost_generic(qnode, w_probe, w_meas, phi=phi, gamma=gamma)

    t0 = time.time()
    rows = []
    c0 = float(obj(theta))
    rows.append({"step": 0, "cost": c0})
    for i in range(1, iters+1):
        theta, c = opt.step_and_cost(obj, theta)
        rows.append({"step": i, "cost": float(c)})
    dt = time.time() - t0

    import pandas as pd
    df = pd.DataFrame(rows)
    df["runtime_sec"] = dt
    df["device"] = device
    df["shots"] = None if (shots is None or shots <= 0) else shots
    df["n_qubits"] = n_qubits
    df["layers"] = layers
    df["iters"] = iters
    df["lr"] = lr
    df["gamma"] = gamma
    df["seed"] = seed
    df["objective"] = obj_kind
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="default.mixed")
    p.add_argument("--objective", default="legacy", choices=["legacy","generic"],
                   help="legacy matches tutorial scale for n_qubits=3; generic works for any n_qubits.")
    p.add_argument("--shots_list", default="1000")
    p.add_argument("--qubits_list", default="3")
    p.add_argument("--layers_list", default="2")
    p.add_argument("--gamma_list", default="0.2")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=395)
    p.add_argument("--phi", default="1.1,0.7,-0.6", help="comma-separated vector of length n_qubits")
    p.add_argument("--out_csv", default="standalone_results.csv")
    args = p.parse_args()

    shots_list = [int(x) for x in parse_csv_list(args.shots_list, cast=float)]
    qubits_list = [int(x) for x in parse_csv_list(args.qubits_list, cast=float)]
    layers_list = [int(x) for x in parse_csv_list(args.layers_list, cast=float)]
    gamma_list = [float(x) for x in parse_csv_list(args.gamma_list, cast=float)]

    import pandas as pd
    all_runs = []
    run_id = 0
    for cfg in config_product(shots=shots_list, n_qubits=qubits_list, layers=layers_list, gamma=gamma_list):
        run_id += 1
        n = int(cfg["n_qubits"])
        phi = parse_vec(args.phi, n=n)
        df = run_one(
            device=args.device,
            shots=None if cfg["shots"] <= 0 else int(cfg["shots"]),
            seed=args.seed,
            n_qubits=n,
            layers=int(cfg["layers"]),
            iters=args.iters,
            lr=args.lr,
            gamma=float(cfg["gamma"]),
            phi=phi,
            objective=args.objective,
        )
        df["run_id"] = run_id
        all_runs.append(df)
        print(f"[Standalone] run {run_id}: obj={df['objective'].iloc[0]} shots={cfg['shots']} qubits={n} layers={cfg['layers']} gamma={cfg['gamma']} final_cost={df['cost'].iloc[-1]:.6f}")

    out = pd.concat(all_runs, ignore_index=True)
    save_csv(args.out_csv, out)
    print(f"[Standalone] saved: {args.out_csv} (rows={len(out)})")

if __name__ == "__main__":
    main()
