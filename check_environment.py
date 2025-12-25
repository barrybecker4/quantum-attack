#!/usr/bin/env python3
"""
Diagnostic script to check Python environment and dependencies.
Run this from PyCharm to see what interpreter it's using.
"""

import sys
import os

print("=" * 70)
print("PYTHON ENVIRONMENT DIAGNOSTICS")
print("=" * 70)

print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")  # First few entries

print("\n" + "=" * 70)
print("DEPENDENCY CHECKS")
print("=" * 70)

# Check qiskit
try:
    import qiskit
    print(f"✓ qiskit: {qiskit.__version__}")
except ImportError as e:
    print(f"✗ qiskit: NOT INSTALLED ({e})")

# Check qiskit-aer
try:
    import qiskit_aer
    print(f"✓ qiskit-aer: {qiskit_aer.__version__}")
except ImportError as e:
    print(f"✗ qiskit-aer: NOT INSTALLED ({e})")

# Check qiskit_ecdlp
try:
    from qiskit_ecdlp.api.CircuitChooser import CircuitChooser
    print("✓ qiskit_ecdlp: INSTALLED")
except ImportError as e:
    print(f"✗ qiskit_ecdlp: NOT INSTALLED ({e})")

# Check sympy (critical dependency)
try:
    import sympy
    print(f"✓ sympy: {sympy.__version__}")
except ImportError as e:
    print(f"✗ sympy: NOT INSTALLED ({e})")
    print("  This is required by qiskit_ecdlp!")

# Check numpy
try:
    import numpy
    print(f"✓ numpy: {numpy.__version__}")
except ImportError as e:
    print(f"✗ numpy: NOT INSTALLED ({e})")

print("\n" + "=" * 70)
print("TESTING qiskit_ecdlp FUNCTIONALITY")
print("=" * 70)

try:
    from qiskit_ecdlp.api.CircuitChooser import CircuitChooser
    try:
        import sympy
        # Try to instantiate CircuitChooser (this is where sympy is needed)
        chooser = CircuitChooser()
        print("✓ CircuitChooser can be instantiated")
    except Exception as e:
        print(f"✗ CircuitChooser instantiation failed: {e}")
        print("  This is likely due to missing dependencies (e.g., sympy)")
except Exception as e:
    print(f"✗ Cannot test CircuitChooser: {e}")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

# Check if we're in a virtual environment
if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
    print("✓ Running in a virtual environment")
    print(f"  Virtual env: {sys.prefix}")
    print("\nTo install missing packages in this environment:")
    print(f"  {sys.executable} -m pip install sympy")
else:
    print("⚠ Running in system Python (not a virtual environment)")
    print(f"  Python: {sys.executable}")
    print("\nTo install missing packages:")
    print(f"  {sys.executable} -m pip install sympy")

print("\n" + "=" * 70)

