"""
Resource Estimates for Breaking Bitcoin's secp256k1 Curve

This script extrapolates from our toy demonstrations to show what
would be required to attack real Bitcoin keys.

Key findings from research literature:
- Roetteler et al. (2017): 9n + 2⌈log₂(n)⌉ + 10 qubits for n-bit curve
- Circuit depth: O(n³ log n) Toffoli gates
- With error correction: ~2000x physical qubit overhead

For Bitcoin's 256-bit curve, this means:
- ~2,330 logical qubits minimum
- ~4.6 million physical qubits (with surface codes)
- Runtime: hours to days depending on assumptions
"""

import math
from dataclasses import dataclass


@dataclass
class CurveParams:
    """Parameters for an elliptic curve"""
    name: str
    bits: int  # Size of prime field in bits
    description: str


# Define curves to analyze
CURVES = [
    CurveParams("Tiny Demo", 3, "Our p=5 demonstration curve"),
    CurveParams("Small Demo", 8, "8-bit curve for extended demo"),
    CurveParams("Medium Demo", 16, "16-bit curve (pushes simulators)"),
    CurveParams("secp192r1", 192, "NIST P-192 curve"),
    CurveParams("secp256k1", 256, "Bitcoin/Ethereum curve"),
    CurveParams("secp384r1", 384, "NIST P-384 curve"),
    CurveParams("secp521r1", 521, "NIST P-521 curve"),
]


def estimate_logical_qubits(n: int) -> int:
    """
    Estimate logical qubits for Shor's ECDLP algorithm.
    
    Formula from Roetteler et al. (2017):
    qubits ≤ 9n + 2⌈log₂(n)⌉ + 10
    """
    return 9 * n + 2 * math.ceil(math.log2(n)) + 10


def estimate_toffoli_gates(n: int) -> int:
    """
    Estimate Toffoli gate count.
    
    Formula: 448n³ log₂(n) + 4090n³
    """
    log_n = math.log2(n)
    return int(448 * (n ** 3) * log_n + 4090 * (n ** 3))


def estimate_physical_qubits(logical: int, code_distance: int = 27) -> int:
    """
    Estimate physical qubits with surface code error correction.
    
    Surface codes require ~2d² physical qubits per logical qubit,
    where d is the code distance (error correction strength).
    
    For cryptographic reliability, d ≈ 27 is commonly assumed.
    """
    physical_per_logical = 2 * (code_distance ** 2)
    return logical * physical_per_logical


def estimate_runtime_hours(toffoli_gates: int, gate_time_ns: float = 100) -> float:
    """
    Estimate runtime assuming given gate execution time.
    
    Optimistic: 100ns per Toffoli (superconducting, well-optimized)
    Pessimistic: 1μs per Toffoli (trapped ion, higher fidelity)
    
    Note: This ignores parallelism and error correction overhead.
    """
    total_ns = toffoli_gates * gate_time_ns
    return total_ns / (1e9 * 3600)  # Convert to hours


def format_large_number(n: int) -> str:
    """Format large numbers with SI prefixes"""
    if n >= 1e15:
        return f"{n/1e15:.1f}P"
    elif n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return str(n)


def analyze_curve(curve: CurveParams) -> dict:
    """Compute all resource estimates for a curve"""
    n = curve.bits
    
    logical_qubits = estimate_logical_qubits(n)
    toffoli_gates = estimate_toffoli_gates(n)
    physical_qubits = estimate_physical_qubits(logical_qubits)
    runtime_optimistic = estimate_runtime_hours(toffoli_gates, 100)
    runtime_pessimistic = estimate_runtime_hours(toffoli_gates, 1000)
    
    return {
        "name": curve.name,
        "bits": n,
        "description": curve.description,
        "logical_qubits": logical_qubits,
        "toffoli_gates": toffoli_gates,
        "physical_qubits": physical_qubits,
        "runtime_optimistic_hours": runtime_optimistic,
        "runtime_pessimistic_hours": runtime_pessimistic,
    }


def print_comparison_table():
    """Print a comparison table of all curves"""
    print("\n" + "=" * 100)
    print("QUANTUM RESOURCE ESTIMATES FOR BREAKING ELLIPTIC CURVES")
    print("=" * 100)
    
    print("\nBased on Roetteler et al. (2017) resource estimates")
    print("Physical qubits assume surface code with distance 27")
    print()
    
    # Header
    print(f"{'Curve':<15} {'Bits':>6} {'Logical':>10} {'Physical':>12} "
          f"{'Toffolis':>12} {'Time (opt)':>12} {'Time (pess)':>12}")
    print("-" * 100)
    
    for curve in CURVES:
        est = analyze_curve(curve)
        
        time_opt = est["runtime_optimistic_hours"]
        time_pess = est["runtime_pessimistic_hours"]
        
        # Format times
        if time_opt < 1:
            time_opt_str = f"{time_opt * 60:.1f} min"
        elif time_opt < 24:
            time_opt_str = f"{time_opt:.1f} hours"
        elif time_opt < 8760:  # 365 days
            time_opt_str = f"{time_opt / 24:.1f} days"
        else:
            time_opt_str = f"{time_opt / 8760:.1f} years"
        
        if time_pess < 1:
            time_pess_str = f"{time_pess * 60:.1f} min"
        elif time_pess < 24:
            time_pess_str = f"{time_pess:.1f} hours"
        elif time_pess < 8760:
            time_pess_str = f"{time_pess / 24:.1f} days"
        else:
            time_pess_str = f"{time_pess / 8760:.1f} years"
        
        print(f"{est['name']:<15} {est['bits']:>6} "
              f"{format_large_number(est['logical_qubits']):>10} "
              f"{format_large_number(est['physical_qubits']):>12} "
              f"{format_large_number(est['toffoli_gates']):>12} "
              f"{time_opt_str:>12} {time_pess_str:>12}")


def print_bitcoin_deep_dive():
    """Detailed analysis of Bitcoin's secp256k1"""
    print("\n" + "=" * 80)
    print("DEEP DIVE: ATTACKING BITCOIN's secp256k1")
    print("=" * 80)
    
    est = analyze_curve(CurveParams("secp256k1", 256, "Bitcoin"))
    
    print(f"""
Curve Parameters:
  Name: secp256k1 (used by Bitcoin, Ethereum, and many others)
  Field size: 256 bits
  Security level: 128 bits (classical)
  
Quantum Resource Requirements:
  Logical qubits:  {est['logical_qubits']:,}
  Physical qubits: {est['physical_qubits']:,} (with surface codes, d=27)
  Toffoli gates:   {est['toffoli_gates']:,}
  
Estimated Runtime:
  Optimistic (100ns/gate): {est['runtime_optimistic_hours']:.1f} hours
  Pessimistic (1μs/gate):  {est['runtime_pessimistic_hours']:.1f} hours

Current Quantum Computer Capabilities (as of 2025):
  IBM Quantum:     ~1,000 physical qubits
  Google Willow:   105 physical qubits (with error correction demo)
  IonQ:            ~32 algorithmic qubits
  
Gap to Close:
  Physical qubits needed: {est['physical_qubits']:,}
  Currently available:    ~1,000
  Factor:                 {est['physical_qubits'] / 1000:,.0f}x more qubits needed
    """)
    
    print("""
Bitcoin Vulnerability Timeline Estimates:
  
  Pessimistic (for attackers): 2045-2055
    - Assumes slow progress in error correction
    - Requires major breakthroughs in qubit stability
    
  Moderate: 2035-2045  
    - Assumes continued exponential progress
    - Still requires significant engineering advances
    
  Optimistic (for attackers): 2030-2035
    - Assumes rapid breakthroughs
    - Some experts consider this plausible
    
Key Factors:
  1. Error rates must drop from ~0.1% to ~0.001%
  2. Qubit coherence times must increase 100-1000x
  3. Cryogenic and control systems must scale
  4. Logical qubit overhead might be reduced by algorithmic improvements
    """)


def print_attack_window():
    """Explain the attack window for Bitcoin transactions"""
    print("\n" + "=" * 80)
    print("THE BITCOIN ATTACK WINDOW")
    print("=" * 80)
    
    print("""
Bitcoin's Exposure Model:

1. UNEXPOSED ADDRESSES (relatively safe)
   - Addresses that have never sent a transaction
   - Public key is hashed (RIPEMD160(SHA256(pubkey)))
   - Attacker would need to break hashing AND ECDSA
   - These are NOT directly vulnerable to Shor's algorithm
   
2. EXPOSED ADDRESSES (vulnerable)  
   - Addresses that have sent at least one transaction
   - Public key is visible on the blockchain
   - ~20-25% of all Bitcoin is in exposed addresses
   - Includes Satoshi's original coins (~1M BTC)

The Attack Scenario:

  Step 1: Scan blockchain for transactions with exposed public keys
  Step 2: Run Shor's algorithm to recover private key
  Step 3: Create transaction spending the funds
  Step 4: Broadcast before legitimate owner
  
  Time available: Depends on mempool congestion (minutes to hours)

Mempool Attack (More Dangerous):

  When you broadcast a transaction, your public key is exposed in the
  mempool BEFORE the transaction confirms (~10 minutes average).
  
  A quantum attacker could:
  1. Monitor mempool for transactions
  2. Extract public key from pending transaction
  3. Run Shor's algorithm (must complete in ~10 minutes)
  4. Create competing transaction sending funds to attacker
  5. Pay higher fee to get priority
  
  This requires:
  - Shor's algorithm completing in <10 minutes
  - With our estimates: needs 100ns gates AND parallelism
    """)


def main():
    """Main entry point"""
    print_comparison_table()
    print_bitcoin_deep_dive()
    print_attack_window()
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The quantum threat to Bitcoin is real but not imminent. Key takeaways:

1. Current quantum computers are ~4,000x too small
2. Even with enough qubits, error rates are ~100x too high  
3. Bitcoin has time to upgrade (BIP-360 proposes solutions)
4. The bigger risk may be perception/market panic before actual threat

What you can do:
- Support BIP-360 and quantum-resistant Bitcoin development
- Avoid address reuse (limits exposure)
- Consider moving funds to fresh addresses periodically
- Stay informed about quantum computing progress
    """)


if __name__ == "__main__":
    main()
