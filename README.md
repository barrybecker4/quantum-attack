# Breaking Small Elliptic Curve Keys with Shor's Algorithm

This project demonstrates Shor's algorithm for the Elliptic Curve Discrete Logarithm Problem (ECDLP) using Qiskit quantum simulators. It's an educational tool to understand the quantum threat to Bitcoin and other ECDSA-based systems.

## The Attack in Plain Terms

Bitcoin uses ECDSA (Elliptic Curve Digital Signature Algorithm) with the secp256k1 curve:
- **Private key**: A 256-bit number `k`
- **Public key**: A point `Q = k * G` where G is the generator point
- **The problem**: Given `Q` and `G`, find `k`

Classically, this requires ~2^128 operations (infeasible). Shor's algorithm solves it in polynomial time on a quantum computer.

## What This Demo Does

Since we can't simulate 2300+ qubits needed for real Bitcoin keys, we use toy curves:
- **Tiny curve (p=5)**: ~15 qubits, runs in seconds
- **Small curve (p=251)**: ~80 qubits, pushes simulator limits

Both demonstrate the same principle that would break Bitcoin given sufficient qubits.

## Prerequisites

```bash
pip install qiskit qiskit-aer numpy
pip install https://p51lee.github.io/assets/python/wheel/qiskit_ecdlp-0.1-py3-none-any.whl
```

For GPU acceleration (optional, recommended for larger curves):
```bash
pip install qiskit-aer-gpu
```

## Project Structure

```
ecdlp_quantum_attack/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── classical_ec.py              # Classical elliptic curve operations
├── tiny_curve_attack.py         # Attack on p=5 curve (~15 qubits)
├── small_curve_attack.py        # Attack on p=251 curve (~80 qubits)
└── bitcoin_extrapolation.py     # Resource estimates for real Bitcoin
```

## Quick Start

**Works immediately (no external dependencies beyond numpy):**
```bash
# Classical EC operations demo
python classical_ec.py

# Resource estimates for real Bitcoin
python bitcoin_extrapolation.py

# Simplified quantum attack simulation
python simplified_demo.py --explain
```

**Requires full Qiskit + qiskit_ecdlp library:**
```bash
# Full quantum circuit attack (see Colab notebook)
python tiny_curve_attack.py
```

## Full Implementation (Google Colab)

The complete working implementation with real quantum circuits is available in this
Google Colab notebook (runs in browser, no local setup needed):

**[Open in Google Colab](https://colab.research.google.com/drive/1w5DFKPIMQemzDK3x1xzq-8omYj7hMcjI?usp=sharing)**

This notebook uses the qiskit_ecdlp library to build actual quantum circuits for
elliptic curve point addition and runs them on Qiskit's Aer simulator.

## Understanding the Output

The attack outputs:
1. **Circuit statistics**: Number of qubits, gates, and depth
2. **Measurement results**: Raw quantum measurements
3. **Recovered key**: The private key extracted via post-processing
4. **Verification**: Confirms `k * G = Q`

## Why This Matters for Bitcoin

Bitcoin addresses that have spent transactions have exposed public keys. A quantum computer with ~2300 logical qubits could:
1. Read the public key from the blockchain
2. Run Shor's algorithm
3. Recover the private key
4. Steal the funds

Current quantum computers have ~100-1000 physical qubits with high error rates. The timeline to 2300+ logical qubits is estimated at 10-20 years, but this is uncertain.

## References

- Proos & Zalka (2003): "Shor's discrete logarithm quantum algorithm for elliptic curves"
- Roetteler et al. (2017): "Quantum Resource Estimates for Computing Elliptic Curve Discrete Logarithms"
- BIP-360: Bitcoin's proposed quantum-resistant address format

## Author's Note

This is educational software demonstrating a well-known vulnerability. It cannot attack real Bitcoin keys—the curves used here are millions of times smaller than secp256k1.
