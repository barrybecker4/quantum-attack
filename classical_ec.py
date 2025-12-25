"""
Classical Elliptic Curve Operations over Prime Fields

This module implements elliptic curve arithmetic for curves of the form:
    y² = x³ + ax + b (mod p)

These classical operations are used to:
1. Generate public keys from private keys
2. Verify that quantum-recovered keys are correct
3. Precompute point multiples for the quantum circuit
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class EllipticCurve:
    """An elliptic curve y² = x³ + ax + b over GF(p)"""
    a: int  # Coefficient a
    b: int  # Coefficient b
    p: int  # Prime modulus
    
    def __post_init__(self):
        # Verify curve is non-singular: 4a³ + 27b² ≠ 0 (mod p)
        discriminant = (4 * pow(self.a, 3, self.p) + 27 * pow(self.b, 2, self.p)) % self.p
        if discriminant == 0:
            raise ValueError(f"Singular curve: 4a³ + 27b² = 0 (mod {self.p})")
    
    @property
    def num_bits(self) -> int:
        """Number of bits needed to represent field elements"""
        return math.ceil(math.log2(self.p)) if self.p > 1 else 1


# Type alias for points: (x, y) or None for point at infinity
Point = Optional[Tuple[int, int]]


def mod_inverse(a: int, p: int) -> int:
    """Compute modular multiplicative inverse using extended Euclidean algorithm"""
    if a == 0:
        raise ValueError("Cannot compute inverse of 0")
    
    # Extended Euclidean Algorithm
    old_r, r = a % p, p
    old_s, s = 1, 0
    
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    
    if old_r != 1:
        raise ValueError(f"{a} has no inverse mod {p}")
    
    return old_s % p


def point_add(curve: EllipticCurve, P: Point, Q: Point) -> Point:
    """
    Add two points on an elliptic curve.
    
    Uses the standard addition formulas:
    - If P = O (infinity), return Q
    - If Q = O (infinity), return P
    - If P = -Q, return O
    - Otherwise, compute slope and new point
    """
    # Handle identity cases
    if P is None:
        return Q
    if Q is None:
        return P
    
    x1, y1 = P
    x2, y2 = Q
    p = curve.p
    
    # Check if P = -Q (points are inverses)
    if x1 == x2 and (y1 + y2) % p == 0:
        return None  # Point at infinity
    
    # Compute slope (lambda)
    if x1 == x2 and y1 == y2:
        # Point doubling: λ = (3x₁² + a) / (2y₁)
        numerator = (3 * x1 * x1 + curve.a) % p
        denominator = (2 * y1) % p
    else:
        # Point addition: λ = (y₂ - y₁) / (x₂ - x₁)
        numerator = (y2 - y1) % p
        denominator = (x2 - x1) % p
    
    lam = (numerator * mod_inverse(denominator, p)) % p
    
    # Compute new point
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    
    return (x3, y3)


def point_double(curve: EllipticCurve, P: Point) -> Point:
    """Double a point (P + P)"""
    return point_add(curve, P, P)


def scalar_multiply(curve: EllipticCurve, k: int, P: Point) -> Point:
    """
    Compute k * P using double-and-add algorithm.
    
    This is the operation that's easy to compute classically but
    hard to reverse (finding k given P and k*P is the ECDLP).
    """
    if k == 0 or P is None:
        return None
    
    if k < 0:
        # Negate the point and use positive scalar
        if P is not None:
            P = (P[0], (-P[1]) % curve.p)
        k = -k
    
    result = None  # Start with point at infinity
    addend = P
    
    while k > 0:
        if k & 1:  # If current bit is 1
            result = point_add(curve, result, addend)
        addend = point_double(curve, addend)
        k >>= 1
    
    return result


def find_curve_order(curve: EllipticCurve, G: Point) -> int:
    """
    Find the order of point G (smallest n such that n*G = O).
    
    For small curves only - uses naive iteration.
    For cryptographic curves, use Schoof's algorithm.
    """
    if G is None:
        return 1
    
    current = G
    order = 1
    
    # Maximum possible order is p + 1 + 2√p (Hasse's theorem)
    max_order = curve.p + 1 + 2 * int(math.sqrt(curve.p)) + 1
    
    while current is not None and order <= max_order:
        current = point_add(curve, current, G)
        order += 1
    
    if current is None:
        return order
    else:
        raise ValueError("Could not find order (point may not be on curve)")


def is_on_curve(curve: EllipticCurve, P: Point) -> bool:
    """Verify that a point lies on the curve"""
    if P is None:
        return True  # Point at infinity is always on curve
    
    x, y = P
    left = (y * y) % curve.p
    right = (x * x * x + curve.a * x + curve.b) % curve.p
    return left == right


def find_all_points(curve: EllipticCurve) -> list[Point]:
    """
    Find all points on the curve (for small curves only).
    
    Returns list including point at infinity.
    """
    points = [None]  # Start with point at infinity
    
    for x in range(curve.p):
        # Compute y² = x³ + ax + b
        y_squared = (x * x * x + curve.a * x + curve.b) % curve.p
        
        # Find square roots (if they exist)
        for y in range(curve.p):
            if (y * y) % curve.p == y_squared:
                points.append((x, y))
    
    return points


# ============================================================================
# Predefined Test Curves
# ============================================================================

# Tiny curve for quick testing (~15 qubits needed for Shor's)
TINY_CURVE = EllipticCurve(a=3, b=2, p=5)
TINY_GENERATOR = (2, 1)  # A generator point of order 5

# Small curve for more realistic demo (~80 qubits)
# y² = x³ + 7 (mod 251) - similar structure to secp256k1
SMALL_CURVE = EllipticCurve(a=0, b=7, p=251)
SMALL_GENERATOR = (1, 54)  # Verify this is on curve and find its order

# Medium curve - pushes simulator limits (~120 qubits)
MEDIUM_CURVE = EllipticCurve(a=0, b=7, p=65521)  # Largest 16-bit prime


def demo():
    """Demonstrate classical EC operations"""
    print("=" * 60)
    print("Classical Elliptic Curve Demo")
    print("=" * 60)
    
    curve = TINY_CURVE
    G = TINY_GENERATOR
    
    print(f"\nCurve: y² = x³ + {curve.a}x + {curve.b} (mod {curve.p})")
    print(f"Generator G = {G}")
    print(f"Bits needed: {curve.num_bits}")
    
    # Verify generator is on curve
    print(f"\nG on curve: {is_on_curve(curve, G)}")
    
    # Find all points
    all_points = find_all_points(curve)
    print(f"Total points on curve: {len(all_points)}")
    print(f"Points: {all_points}")
    
    # Compute scalar multiples
    print("\nScalar multiples of G:")
    for k in range(1, 8):
        kG = scalar_multiply(curve, k, G)
        print(f"  {k} * G = {kG}")
    
    # Simulate key generation
    private_key = 3
    public_key = scalar_multiply(curve, private_key, G)
    print(f"\nKey generation:")
    print(f"  Private key k = {private_key}")
    print(f"  Public key Q = k * G = {public_key}")
    
    # The ECDLP: Given G and Q, find k
    print(f"\nThe ECDLP challenge:")
    print(f"  Given: G = {G}, Q = {public_key}")
    print(f"  Find: k such that k * G = Q")
    print(f"  (Classical brute force would try k = 1, 2, 3, ...)")
    print(f"  (Quantum Shor's algorithm finds k in polynomial time)")


if __name__ == "__main__":
    demo()
