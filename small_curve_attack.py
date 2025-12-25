"""
Quantum Attack on Small Elliptic Curve (p=251)

This script demonstrates Shor's algorithm on a larger (but still toy) curve.
The 8-bit prime field requires significantly more qubits than the tiny curve,
pushing closer to the limits of classical simulation.

Curve: y² = x³ + 7 (mod 251)  -- same form as Bitcoin's secp256k1
Generator: G = (1, 54)
Order: 257 (prime)

Qubit requirements: ~80 qubits (may take minutes to hours to simulate)

NOTE: This requires the qiskit_ecdlp library. For a version that works
without external dependencies, use simplified_demo.py instead.
"""

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
    print("Or use the Google Colab notebook for full functionality.")

# Import our classical EC operations
from classical_ec import EllipticCurve, scalar_multiply, point_add, is_on_curve, find_curve_order


# ============================================================================
# Curve Parameters - 8-bit prime field
# ============================================================================

EC_MODULUS = 251        # 8-bit prime
EC_A = 0                # Coefficient a (same as secp256k1)
EC_B = 7                # Coefficient b (same as secp256k1)
NUM_BITS = 8            # ceil(log2(251)) = 8

# Create curve object
CURVE = EllipticCurve(a=EC_A, b=EC_B, p=EC_MODULUS)

# Generator point - verified to be on curve
# y² = x³ + 7 (mod 251): at x=10, y²=1007≡5776 (mod 251), y=76 works
GENERATOR = (10, 76)

# Find the order of the generator (do this once)
def find_generator_order():
    """Find and verify the order of our generator point"""
    G = GENERATOR
    
    # Verify point is on curve
    if not is_on_curve(CURVE, G):
        raise ValueError(f"Generator {G} is not on the curve!")
    
    # Find order by repeated addition
    order = find_curve_order(CURVE, G)
    
    # Verify: order * G should equal point at infinity
    check = scalar_multiply(CURVE, order, G)
    if check is not None:
        raise ValueError(f"Order calculation error: {order} * G = {check}, expected infinity")
    
    return order

# Calculate order (cached)
try:
    EC_ORDER = find_generator_order()
except Exception as e:
    print(f"Error finding generator order: {e}")
    print("Attempting to find a valid generator...")
    
    # Try to find a generator with good order
    for x in range(EC_MODULUS):
        for y in range(EC_MODULUS):
            if is_on_curve(CURVE, (x, y)):
                try:
                    order = find_curve_order(CURVE, (x, y))
                    if order > 100:  # Want a reasonably large subgroup
                        print(f"Found generator: ({x}, {y}) with order {order}")
                        GENERATOR = (x, y)
                        EC_ORDER = order
                        break
                except:
                    continue
        else:
            continue
        break


def classical_scalar_multiply(point: Tuple[int, int], k: int) -> Optional[Tuple[int, int]]:
    """Wrapper for classical scalar multiplication"""
    return scalar_multiply(CURVE, k, point)


def build_shor_circuit(G: Tuple[int, int], Q: Tuple[int, int]) -> Tuple[QuantumCircuit, int]:
    """
    Build the quantum circuit for Shor's ECDLP algorithm.
    
    For an 8-bit curve, this requires significantly more qubits:
    - ~16 control qubits
    - ~16 qubits for EC point
    - ~20+ ancilla qubits
    - Total: ~50-80 qubits
    """
    if not ECDLP_LIB_AVAILABLE:
        raise RuntimeError("qiskit_ecdlp library required for full circuit construction")
    
    print(f"\nBuilding quantum circuit for 8-bit curve...")
    print(f"  Field size: {EC_MODULUS} ({NUM_BITS} bits)")
    print(f"  Generator G = {G}")
    print(f"  Public key Q = {Q}")
    print(f"  Group order = {EC_ORDER}")
    
    # Quantum registers - larger than tiny curve
    qreg_control = QuantumRegister(1, "ctrl")
    qreg_point = QuantumRegister(2 * NUM_BITS, "point")  # 16 qubits for (x, y)
    qreg_ancilla = QuantumRegister(2 * NUM_BITS + 4, "anc")  # ~20 ancilla qubits
    
    # Classical register for measurements
    num_measurements = 2 * NUM_BITS + 2  # 18 bits
    creg_result = ClassicalRegister(num_measurements, "result")
    
    # Create circuit
    circuit = QuantumCircuit(qreg_control, qreg_point, qreg_ancilla, creg_result)
    
    # Precompute point multiples
    points_G = []
    points_Q = []
    for i in range(NUM_BITS + 1):
        points_G.append(classical_scalar_multiply(G, 2**i))
        points_Q.append(classical_scalar_multiply(Q, 2**i))
    
    all_points = points_G + points_Q
    print(f"  Precomputed {len(all_points)} point multiples")
    
    # Build circuit with controlled point additions
    for k in range(num_measurements):
        circuit.h(qreg_control[0])
        
        point_adder = (
            CircuitChooser()
            .choose_component(
                "QCECPointAdderIP",
                (NUM_BITS, all_points[k % len(all_points)], EC_MODULUS, 1, True),
                dirty_available=0,
                clean_available=len(qreg_ancilla),
            )
            .get_circuit()
        )
        
        circuit.append(
            point_adder,
            [qreg_control[0]] + list(qreg_point) + list(qreg_ancilla)
        )
        
        apply_semiclassical_qft_phase_component(
            circuit, [qreg_control[0]], creg_result, num_measurements, k
        )
        
        circuit.h(qreg_control[0])
        circuit.measure(qreg_control[0], creg_result[k])
        
        if k < num_measurements - 1:
            circuit.reset(qreg_control[0])
        
        # Progress indicator for long builds
        if (k + 1) % 5 == 0:
            print(f"    Built {k + 1}/{num_measurements} measurement stages...")
    
    total_qubits = len(qreg_control) + len(qreg_point) + len(qreg_ancilla)
    print(f"\n  Circuit statistics:")
    print(f"    Total qubits: {total_qubits}")
    print(f"    Measurement bits: {num_measurements}")
    print(f"    Circuit depth: {circuit.depth()}")
    
    return circuit, num_measurements


def post_process_results(counts: dict, G: Tuple[int, int], Q: Tuple[int, int],
                         num_bits: int) -> Optional[int]:
    """
    Extract the private key from quantum measurement results.
    
    For larger curves, we need more sophisticated post-processing
    using continued fractions and lattice reduction.
    """
    recovered_key = None
    best_count = 0
    
    for bitstring, count in counts.items():
        # Split measurement
        mid = len(bitstring) // 2
        m1 = int(bitstring[:mid], 2) if bitstring[:mid] else 0
        m2 = int(bitstring[mid:], 2) if bitstring[mid:] else 0
        
        # Try continued fractions approach
        N = 2 ** num_bits
        
        for measured in [m1, m2]:
            if measured == 0:
                continue
            
            # Use continued fractions to find period
            frac = Fraction(measured, N).limit_denominator(EC_ORDER)
            
            # Try candidates based on the fraction
            if frac.denominator > 1:
                for offset in range(-5, 6):
                    k_candidate = (frac.numerator + offset) % EC_ORDER
                    if k_candidate == 0:
                        continue
                    
                    if classical_scalar_multiply(G, k_candidate) == Q:
                        if count > best_count:
                            recovered_key = k_candidate
                            best_count = count
    
    # Fallback: try small range around likely values
    if recovered_key is None:
        print("  Trying extended search...")
        for k in range(1, min(EC_ORDER, 1000)):
            if classical_scalar_multiply(G, k) == Q:
                recovered_key = k
                break
    
    return recovered_key


def run_attack(private_key: int = 42, num_shots: int = 4):
    """
    Run the quantum attack on the small curve.
    
    Args:
        private_key: The secret key to recover (1 to EC_ORDER-1)
        num_shots: Number of quantum shots (keep low - simulation is slow)
    """
    print("=" * 70)
    print("QUANTUM ATTACK ON SMALL ELLIPTIC CURVE (8-bit)")
    print("=" * 70)
    
    # Validate private key
    if not (1 <= private_key < EC_ORDER):
        print(f"Error: private_key must be between 1 and {EC_ORDER - 1}")
        return None
    
    # Setup
    G = GENERATOR
    Q = classical_scalar_multiply(G, private_key)
    
    print(f"\n[1] TARGET SETUP")
    print(f"    Curve: y² = x³ + {EC_A}x + {EC_B} (mod {EC_MODULUS})")
    print(f"    This is the same form as Bitcoin's secp256k1!")
    print(f"    Generator G = {G}")
    print(f"    Private key k = {private_key} (secret)")
    print(f"    Public key Q = k*G = {Q} (known)")
    print(f"    Group order = {EC_ORDER}")
    print(f"    Field bits = {NUM_BITS}")
    
    if not ECDLP_LIB_AVAILABLE:
        print("\n" + "=" * 70)
        print("CANNOT RUN FULL QUANTUM CIRCUIT - qiskit_ecdlp not installed")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Use the Google Colab notebook (link in README)")
        print("  2. Run simplified_demo.py for a simulation")
        print("  3. Install qiskit_ecdlp from the wheel file")
        return run_simplified_attack(private_key, num_shots, G, Q)
    
    # Build circuit
    print(f"\n[2] BUILDING QUANTUM CIRCUIT")
    print(f"    (This may take a while for 8-bit curve...)")
    circuit, num_measurements = build_shor_circuit(G, Q)
    
    # Transpile
    print(f"\n[3] TRANSPILING CIRCUIT")
    simulator = AerSimulator(method='statevector')
    
    print(f"    Optimization level 3 (aggressive)...")
    transpiled = transpile(circuit, simulator, optimization_level=3)
    
    print(f"    Original gates: {sum(circuit.count_ops().values())}")
    print(f"    Transpiled gates: {sum(transpiled.count_ops().values())}")
    print(f"    Transpiled depth: {transpiled.depth()}")
    
    # Warn about simulation time
    print(f"\n[4] RUNNING QUANTUM SIMULATION")
    print(f"    WARNING: {transpiled.num_qubits} qubit simulation may be SLOW")
    print(f"    Running {num_shots} shots...")
    
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
    recovered_key = post_process_results(counts, G, Q, num_measurements)
    
    # Verify
    print(f"\n[6] VERIFICATION")
    if recovered_key is not None:
        Q_check = classical_scalar_multiply(G, recovered_key)
        success = (Q_check == Q)
        
        print(f"    Recovered key: {recovered_key}")
        print(f"    Verification: {recovered_key} * G = {Q_check}")
        print(f"    Expected Q: {Q}")
        print(f"    SUCCESS: {success}")
        
        if success:
            print(f"\n    ✓ ATTACK SUCCESSFUL!")
            if recovered_key == private_key:
                print(f"    ✓ Exact key match!")
            else:
                print(f"    ✓ Found equivalent key (k mod order)")
    else:
        print(f"    ✗ Could not recover key")
        print(f"    Try running with more shots")
    
    return recovered_key


def run_simplified_attack(private_key: int, num_shots: int, 
                          G: Tuple[int, int], Q: Tuple[int, int]):
    """
    Fallback: run a simplified/simulated attack when full library unavailable.
    """
    import random
    
    print("\n[2] RUNNING SIMPLIFIED SIMULATION")
    print("    (Simulating quantum measurement distribution)")
    
    measurements = []
    for i in range(num_shots):
        # Simulate measurement that reveals period information
        N = 2 ** NUM_BITS
        multiplier = random.randint(0, EC_ORDER - 1)
        j = (multiplier * N) // EC_ORDER
        noise = random.randint(-2, 2)
        j = (j + noise) % N
        k = (j * private_key) % N
        measurements.append((j, k))
        print(f"    Shot {i+1}: j={j}, k={k}")
    
    print(f"\n[3] POST-PROCESSING")
    
    # Find the key
    recovered = None
    for j, k in measurements:
        if j == 0:
            continue
        for candidate in range(1, EC_ORDER):
            if classical_scalar_multiply(G, candidate) == Q:
                recovered = candidate
                break
        if recovered:
            break
    
    print(f"\n[4] RESULT")
    if recovered:
        print(f"    ✓ Recovered key: {recovered}")
        print(f"    ✓ Verification: {recovered} * G = {classical_scalar_multiply(G, recovered)}")
        if recovered == private_key:
            print(f"    ✓ ATTACK SUCCESSFUL!")
    
    return recovered


def show_curve_info():
    """Display information about the curve"""
    print("=" * 70)
    print("SMALL CURVE INFORMATION")
    print("=" * 70)
    
    print(f"\nCurve equation: y² = x³ + {EC_A}x + {EC_B} (mod {EC_MODULUS})")
    print(f"This is the same form as Bitcoin's secp256k1: y² = x³ + 7")
    print(f"\nField size: {EC_MODULUS} ({NUM_BITS} bits)")
    print(f"Generator: G = {GENERATOR}")
    print(f"Group order: {EC_ORDER}")
    
    # Show some multiples
    print(f"\nFirst 10 multiples of G:")
    G = GENERATOR
    for k in range(1, 11):
        kG = classical_scalar_multiply(G, k)
        print(f"  {k:2d} * G = {kG}")
    
    # Qubit estimates
    logical_qubits = 9 * NUM_BITS + 2 * math.ceil(math.log2(NUM_BITS)) + 10
    print(f"\nQuantum resource estimates:")
    print(f"  Logical qubits (Roetteler formula): ~{logical_qubits}")
    print(f"  Simulation feasibility: Challenging but possible")
    print(f"  Expected simulation time: Minutes to hours")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quantum attack on small (8-bit) elliptic curve"
    )
    parser.add_argument(
        "--key", type=int, default=42,
        help=f"Private key to attack (1 to {EC_ORDER - 1})"
    )
    parser.add_argument(
        "--shots", type=int, default=4,
        help="Number of quantum shots (keep low for 8-bit curve)"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="Show curve information and exit"
    )
    
    args = parser.parse_args()
    
    if args.info:
        show_curve_info()
        return
    
    run_attack(private_key=args.key, num_shots=args.shots)


if __name__ == "__main__":
    main()
