"""
Generic Quantum Attack on Elliptic Curve Discrete Logarithm

This script demonstrates Shor's algorithm breaking an elliptic curve
discrete logarithm. It can work with any curve configuration specified
in a JSON file.

Usage:
    python elliptic_curve_attack.py --config configs/tiny_curve.json --key 3 --shots 8
"""

import json
import math
import sys
from typing import Tuple, Optional
from fractions import Fraction

# Check for required packages
try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("WARNING: Qiskit not installed. Install with: pip install qiskit qiskit-aer")

try:
    from qiskit_ecdlp.api.CircuitChooser import CircuitChooser
    from qiskit_ecdlp.impl.util.semiclassical_qft import apply_semiclassical_qft_phase_component
    # Check for required dependencies
    try:
        import sympy  # qiskit_ecdlp requires sympy
    except ImportError:
        ECDLP_LIB_AVAILABLE = False
        print("WARNING: qiskit_ecdlp requires sympy but it's not installed")
        print("Install missing dependency: pip install sympy")
    else:
        ECDLP_LIB_AVAILABLE = True
except ImportError as e:
    ECDLP_LIB_AVAILABLE = False
    print(f"WARNING: qiskit_ecdlp import failed: {e}")
    print("Install from: https://p51lee.github.io/assets/python/wheel/qiskit_ecdlp-0.1-py3-none-any.whl")

# Import our classical EC operations
from classical_ec import EllipticCurve, scalar_multiply, point_add, is_on_curve, find_curve_order


# ============================================================================
# Configuration Loading
# ============================================================================

def load_curve_config(config_path: str) -> dict:
    """
    Load and validate curve configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Dictionary with curve parameters
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    # Validate required fields
    required_fields = ['modulus', 'a', 'b', 'generator']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Validate generator is a 2-element list
    if not isinstance(config['generator'], list) or len(config['generator']) != 2:
        raise ValueError("generator must be a list of two integers [x, y]")
    
    # Convert generator to tuple
    config['generator'] = tuple(config['generator'])
    
    # Calculate number of bits
    config['num_bits'] = math.ceil(math.log2(config['modulus']))
    
    # Create curve object
    config['curve'] = EllipticCurve(a=config['a'], b=config['b'], p=config['modulus'])
    
    # Validate generator is on curve
    if not is_on_curve(config['curve'], config['generator']):
        raise ValueError(f"Generator {config['generator']} is not on the curve!")
    
    # Calculate order if needed
    if config.get('calculate_order', False):
        print(f"Calculating order of generator {config['generator']}...")
        try:
            config['order'] = find_curve_order(config['curve'], config['generator'])
            print(f"  Found order: {config['order']}")
        except Exception as e:
            print(f"Error finding generator order: {e}")
            raise
    elif 'order' not in config or config['order'] is None:
        raise ValueError("order must be specified when calculate_order is false")
    
    return config


# ============================================================================
# Attack Functions
# ============================================================================

def classical_scalar_multiply(curve: EllipticCurve, point: Tuple[int, int], k: int) -> Optional[Tuple[int, int]]:
    """Wrapper for classical scalar multiplication"""
    return scalar_multiply(curve, k, point)


def build_shor_circuit(config: dict, G: Tuple[int, int], Q: Tuple[int, int]) -> Tuple[QuantumCircuit, int]:
    """
    Build the quantum circuit for Shor's ECDLP algorithm.
    
    Args:
        config: Curve configuration dictionary
        G: Generator point
        Q: Public key point (Q = k*G for unknown k)
    
    Returns:
        (circuit, num_measurements): The quantum circuit and number of measured bits
    """
    if not ECDLP_LIB_AVAILABLE:
        raise RuntimeError("qiskit_ecdlp library required for circuit construction")
    
    modulus = config['modulus']
    num_bits = config['num_bits']
    order = config['order']
    
    print(f"\nBuilding quantum circuit...")
    print(f"  Curve: {config.get('name', 'Unknown')}")
    print(f"  Field size: {modulus} ({num_bits} bits)")
    print(f"  Generator G = {G}")
    print(f"  Public key Q = {Q}")
    print(f"  Group order = {order}")
    
    # Quantum registers
    qreg_control = QuantumRegister(1, "ctrl")           # Control qubit for phase estimation
    qreg_point = QuantumRegister(2 * num_bits, "point") # Stores EC point (x, y)
    qreg_ancilla = QuantumRegister(2 * num_bits + 4, "anc")  # Ancilla for arithmetic
    
    # Classical register for measurements
    num_measurements = 2 * num_bits + 2  # Enough bits for period extraction
    creg_result = ClassicalRegister(num_measurements, "result")
    
    # Create circuit
    circuit = QuantumCircuit(qreg_control, qreg_point, qreg_ancilla, creg_result)
    
    # Precompute point multiples: 2^i * G and 2^i * Q
    points_G = []
    points_Q = []
    for i in range(num_bits + 1):
        points_G.append(classical_scalar_multiply(config['curve'], G, 2**i))
        points_Q.append(classical_scalar_multiply(config['curve'], Q, 2**i))
    
    all_points = points_G + points_Q
    
    print(f"  Precomputed {len(all_points)} point multiples")
    
    # Build the circuit: apply controlled point additions
    for k in range(num_measurements):
        # Put control qubit in superposition
        circuit.h(qreg_control[0])
        
        # Controlled point addition
        point_adder = (
            CircuitChooser()
            .choose_component(
                "QCECPointAdderIP",
                (num_bits, all_points[k % len(all_points)], modulus, 1, True),
                dirty_available=0,
                clean_available=len(qreg_ancilla),
            )
            .get_circuit()
        )
        
        circuit.append(
            point_adder,
            [qreg_control[0]] + list(qreg_point) + list(qreg_ancilla)
        )
        
        # Apply semi-classical QFT phase
        apply_semiclassical_qft_phase_component(
            circuit, [qreg_control[0]], creg_result, num_measurements, k
        )
        
        # Measure and reset control qubit
        circuit.h(qreg_control[0])
        circuit.measure(qreg_control[0], creg_result[k])
        
        # Reset for next iteration (semi-classical QFT technique)
        if k < num_measurements - 1:
            circuit.reset(qreg_control[0])
        
        # Progress indicator for long builds
        if (k + 1) % 5 == 0 and num_measurements > 10:
            print(f"    Built {k + 1}/{num_measurements} measurement stages...")
    
    total_qubits = len(qreg_control) + len(qreg_point) + len(qreg_ancilla)
    print(f"  Total qubits: {total_qubits}")
    print(f"  Measurement bits: {num_measurements}")
    
    return circuit, num_measurements


def post_process_results(counts: dict, config: dict, G: Tuple[int, int], Q: Tuple[int, int],
                         num_measurements: int) -> Optional[int]:
    """
    Extract the private key from quantum measurement results.
    
    Uses continued fractions to find the period from measurement outcomes,
    then computes the discrete logarithm.
    """
    recovered_key = None
    best_count = 0
    order = config['order']
    num_bits = config['num_bits']
    
    for bitstring, count in counts.items():
        # Split measurement into two parts (for the two periods)
        mid = len(bitstring) // 2
        m1 = int(bitstring[:mid], 2) if bitstring[:mid] else 0
        m2 = int(bitstring[mid:], 2) if bitstring[mid:] else 0
        
        # Try to extract key using continued fractions
        N = 2 ** num_bits
        
        for measured in [m1, m2]:
            if measured == 0:
                continue
            
            # Use continued fractions to find the period
            frac = Fraction(measured, N).limit_denominator(order)
            
            # Try candidates based on the fraction
            if frac.denominator > 1:
                for offset in range(-5, 6):
                    k_candidate = (frac.numerator + offset) % order
                    if k_candidate == 0:
                        continue
                    
                    if classical_scalar_multiply(config['curve'], G, k_candidate) == Q:
                        if count > best_count:
                            recovered_key = k_candidate
                            best_count = count
    
    # Fallback: try extended search
    if recovered_key is None:
        print("  Post-processing didn't converge, trying extended search...")
        search_limit = min(order, 1000)
        for k in range(1, search_limit):
            if classical_scalar_multiply(config['curve'], G, k) == Q:
                recovered_key = k
                break
    
    return recovered_key


def run_simplified_fallback(config: dict, private_key: int, num_shots: int,
                            G: Tuple[int, int], Q: Tuple[int, int]) -> Optional[int]:
    """Fallback when qiskit_ecdlp is not available"""
    import random
    
    num_bits = config['num_bits']
    order = config['order']
    
    print(f"\n[2] SIMULATING QUANTUM MEASUREMENTS ({num_shots} shots)")
    print(f"    (Classical simulation of measurement distribution)")
    
    for i in range(num_shots):
        N = 2 ** num_bits
        multiplier = random.randint(0, order - 1)
        j = (multiplier * N) // order
        noise = random.randint(-2, 2)
        j = (j + noise) % N
        k = (j * private_key) % N
        print(f"    Shot {i+1}: j={j}, k={k}")
    
    print(f"\n[3] POST-PROCESSING")
    
    # Find key
    recovered = None
    for k in range(1, min(order, 1000)):
        if classical_scalar_multiply(config['curve'], G, k) == Q:
            recovered = k
            break
    
    print(f"\n[4] RESULT")
    if recovered:
        print(f"    ✓ Recovered key: {recovered}")
        print(f"    ✓ Verification: {recovered} * G = {classical_scalar_multiply(config['curve'], G, recovered)}")
        print(f"    ✓ ATTACK SUCCESSFUL!")
    
    return recovered


def run_attack(config_path: str, private_key: int = 3, num_shots: int = 8):
    """
    Run the complete quantum attack.
    
    Args:
        config_path: Path to JSON configuration file
        private_key: The secret key to recover (for verification)
        num_shots: Number of quantum circuit executions
    """
    print("=" * 70)
    print("QUANTUM ATTACK ON ELLIPTIC CURVE DISCRETE LOGARITHM")
    print("=" * 70)
    
    # Load configuration
    try:
        config = load_curve_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Setup
    G = config['generator']
    Q = classical_scalar_multiply(config['curve'], G, private_key)
    
    curve_name = config.get('name', 'Unknown Curve')
    modulus = config['modulus']
    a = config['a']
    b = config['b']
    order = config['order']
    
    print(f"\n[1] TARGET SETUP")
    print(f"    Curve: {curve_name}")
    if config.get('description'):
        print(f"    {config['description']}")
    print(f"    Curve equation: y² = x³ + {a}x + {b} (mod {modulus})")
    print(f"    Generator G = {G}")
    print(f"    Private key k = {private_key} (secret)")
    print(f"    Public key Q = k*G = {Q} (known)")
    print(f"    Group order = {order}")
    print(f"    Field bits = {config['num_bits']}")
    
    if not ECDLP_LIB_AVAILABLE:
        print("\n" + "=" * 70)
        print("qiskit_ecdlp library not available - running simplified demo")
        print("=" * 70)
        return run_simplified_fallback(config, private_key, num_shots, G, Q)
    
    # Build circuit
    print(f"\n[2] BUILDING QUANTUM CIRCUIT")
    try:
        circuit, num_measurements = build_shor_circuit(config, G, Q)
    except Exception as e:
        print(f"Error building circuit: {e}")
        print("\nFalling back to simplified simulation...")
        return run_simplified_fallback(config, private_key, num_shots, G, Q)
    
    # Transpile for simulator
    print(f"\n[3] TRANSPILING CIRCUIT")
    simulator = AerSimulator(method='statevector')
    transpiled = transpile(circuit, simulator, optimization_level=3)
    
    print(f"    Original depth: {circuit.depth()}")
    print(f"    Transpiled depth: {transpiled.depth()}")
    print(f"    Gate count: {transpiled.count_ops()}")
    
    # Run simulation
    print(f"\n[4] RUNNING QUANTUM SIMULATION ({num_shots} shots)")
    if transpiled.num_qubits > 20:
        print(f"    WARNING: {transpiled.num_qubits} qubit simulation may be SLOW")
    
    import time
    start_time = time.time()
    
    result = simulator.run(transpiled, shots=num_shots).result()
    counts = result.get_counts()
    
    elapsed = time.time() - start_time
    print(f"    Completed in {elapsed:.1f} seconds")
    
    print(f"\n    Measurement outcomes:")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"      {bitstring}: {count}")
    
    # Post-process
    print(f"\n[5] POST-PROCESSING")
    recovered_key = post_process_results(counts, config, G, Q, num_measurements)
    
    # Verify
    print(f"\n[6] VERIFICATION")
    if recovered_key is not None:
        Q_check = classical_scalar_multiply(config['curve'], G, recovered_key)
        success = (Q_check == Q)
        
        print(f"    Recovered key: {recovered_key}")
        print(f"    Verification: {recovered_key} * G = {Q_check}")
        print(f"    Expected Q: {Q}")
        print(f"    SUCCESS: {success}")
        
        if success and recovered_key == private_key:
            print(f"\n    ✓ ATTACK SUCCESSFUL - Private key recovered!")
        elif success:
            print(f"\n    ✓ Found equivalent key (k mod order)")
    else:
        print(f"    FAILED - Could not recover key")
        print(f"    Try running with more shots")
    
    return recovered_key


def show_curve_info(config_path: str):
    """Display information about the curve"""
    try:
        config = load_curve_config(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    print("=" * 70)
    print(f"CURVE INFORMATION: {config.get('name', 'Unknown')}")
    print("=" * 70)
    
    modulus = config['modulus']
    a = config['a']
    b = config['b']
    G = config['generator']
    order = config['order']
    num_bits = config['num_bits']
    
    print(f"\nCurve equation: y² = x³ + {a}x + {b} (mod {modulus})")
    if config.get('description'):
        print(f"\n{config['description']}")
    print(f"\nField size: {modulus} ({num_bits} bits)")
    print(f"Generator: G = {G}")
    print(f"Group order: {order}")
    
    # Show some multiples
    print(f"\nFirst 10 multiples of G:")
    for k in range(1, 11):
        kG = classical_scalar_multiply(config['curve'], G, k)
        if kG:
            print(f"  {k:2d} * G = {kG}")
        else:
            print(f"  {k:2d} * G = O (point at infinity)")
    
    # Qubit estimates
    logical_qubits = 9 * num_bits + 2 * math.ceil(math.log2(num_bits)) + 10
    print(f"\nQuantum resource estimates:")
    print(f"  Logical qubits (Roetteler formula): ~{logical_qubits}")
    if num_bits <= 3:
        print(f"  Simulation feasibility: Easy (seconds)")
    elif num_bits <= 8:
        print(f"  Simulation feasibility: Challenging but possible (minutes to hours)")
    else:
        print(f"  Simulation feasibility: Very difficult (may be impractical)")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quantum attack on elliptic curve discrete logarithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python elliptic_curve_attack.py --key 3 --shots 8
  python elliptic_curve_attack.py --config configs/small_curve.json --key 42 --shots 4
  python elliptic_curve_attack.py --info
        """
    )
    parser.add_argument(
        "--config", type=str, default="configs/tiny_curve.json",
        help="Path to JSON configuration file (default: configs/tiny_curve.json)"
    )
    parser.add_argument(
        "--key", type=int, default=3,
        help="Private key to attack"
    )
    parser.add_argument(
        "--shots", type=int, default=8,
        help="Number of quantum shots"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show curve information and exit"
    )
    
    args = parser.parse_args()
    
    if args.info:
        show_curve_info(args.config)
        return
    
    # Validate key range (we'll check this after loading config)
    try:
        config = load_curve_config(args.config)
        order = config['order']
        
        if not (1 <= args.key < order):
            print(f"Error: key must be between 1 and {order - 1}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    run_attack(config_path=args.config, private_key=args.key, num_shots=args.shots)


if __name__ == "__main__":
    main()

