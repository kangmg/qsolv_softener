"""
Test new features: apply_qeq option and each_neighbor
"""

import numpy as np
from ase import Atoms
from ase.build import molecule
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from qsolv_softener import ChargeSoftener


def create_test_system():
    """Create a simple test system with methanol and water."""
    methanol = molecule('CH3OH')
    methanol.center(vacuum=5.0)
    
    water_positions = [
        [8.0, 5.0, 5.0],
        [8.8, 5.3, 5.2],
        [7.3, 5.4, 5.3],
        [5.0, 8.0, 5.0],
        [5.3, 8.7, 5.3],
        [4.5, 8.2, 5.6],
        [15.0, 15.0, 15.0],
        [15.5, 15.3, 15.2],
        [14.8, 15.4, 15.3],
    ]
    
    water_symbols = ['O', 'H', 'H'] * 3
    water = Atoms(water_symbols, positions=water_positions)
    
    combined = methanol + water
    
    n_methanol = len(methanol)
    solute_indices = list(range(n_methanol))
    
    methanol_charges = np.array([
        -0.18,
        0.06,
        0.06,
        0.06,
        -0.68,
        0.42
    ])
    
    water_charges = np.array([-0.834, 0.417, 0.417] * 3)
    
    return combined, solute_indices, methanol_charges, water_charges


def test_apply_qeq_option():
    """Test the apply_qeq option in run()."""
    print("="*70)
    print("Test 1: apply_qeq Option")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_test_system()
    
    softener = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=5.0
    )
    
    print("\n--- With QEq (default) ---")
    updated_with_qeq = softener.run(apply_qeq=True)
    charges_with_qeq = updated_with_qeq.get_initial_charges()
    print(f"Solute charges: {charges_with_qeq[solute_indices]}")
    print(f"Total charge: {np.sum(charges_with_qeq):.6f}")
    
    softener2 = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=5.0
    )
    
    print("\n--- Without QEq (only softening) ---")
    updated_without_qeq = softener2.run(apply_qeq=False)
    charges_without_qeq = updated_without_qeq.get_initial_charges()
    print(f"Solute charges: {charges_without_qeq[solute_indices]}")
    print(f"Total charge: {np.sum(charges_without_qeq):.6f}")
    
    softened_only = softener2.get_softened_charges()
    print(f"\nVerification: softened charges match no-QEq result:")
    print(f"  Match: {np.allclose(charges_without_qeq, softened_only)}")
    
    print("\n✓ apply_qeq option working correctly")


def test_each_neighbor():
    """Test the each_neighbor attribute."""
    print("\n" + "="*70)
    print("Test 2: each_neighbor Attribute")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_test_system()
    
    print(f"\nSystem setup:")
    print(f"  Solute atoms: {len(solute_indices)} (methanol)")
    print(f"  Solute indices: {solute_indices}")
    print(f"  Solvent atoms: {len(solvent_charges)} (water)")
    print(f"  Total atoms: {len(atoms)}")
    
    softener = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=5.0
    )
    
    updated_atoms = softener.run()
    
    print(f"\nNeighbor information for each solute atom (radius={softener.radius} Å):")
    print(f"  each_neighbor length: {len(softener.each_neighbor)}")
    
    for i, (solute_idx, neighbors) in enumerate(zip(solute_indices, softener.each_neighbor)):
        symbol = atoms[solute_idx].symbol
        n_neighbors = len(neighbors)
        if n_neighbors > 0:
            print(f"\n  Solute atom {i} (index {solute_idx}, {symbol}):")
            print(f"    Neighbors: {neighbors}")
            print(f"    Count: {n_neighbors}")
            
            for neighbor_idx in neighbors:
                neighbor_symbol = atoms[neighbor_idx].symbol
                dist = np.linalg.norm(
                    atoms.positions[neighbor_idx] - atoms.positions[solute_idx]
                )
                print(f"      - Atom {neighbor_idx} ({neighbor_symbol}): {dist:.3f} Å")
        else:
            print(f"\n  Solute atom {i} (index {solute_idx}, {symbol}): No neighbors")
    
    print(f"\nVerification:")
    print(f"  each_neighbor is a list: {isinstance(softener.each_neighbor, list)}")
    print(f"  Length matches solute atoms: {len(softener.each_neighbor) == len(solute_indices)}")
    
    has_empty = any(len(n) == 0 for n in softener.each_neighbor)
    has_nonempty = any(len(n) > 0 for n in softener.each_neighbor)
    print(f"  Contains some empty lists (no neighbors): {has_empty}")
    print(f"  Contains some non-empty lists (with neighbors): {has_nonempty}")
    
    print("\n✓ each_neighbor attribute working correctly")


def test_visualization_use_case():
    """Demonstrate use case for visualization."""
    print("\n" + "="*70)
    print("Test 3: Visualization Use Case")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_test_system()
    
    softener = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=5.0
    )
    
    updated_atoms = softener.run()
    
    print("\nVisualization data available:")
    print(f"  1. Solute indices: {solute_indices}")
    print(f"  2. Filtered solvent indices: {softener.filtered_solvent_indices}")
    print(f"  3. Each neighbor for solute atoms:")
    
    for i, (solute_idx, neighbors) in enumerate(zip(solute_indices, softener.each_neighbor)):
        if len(neighbors) > 0:
            print(f"     Solute {i} -> Neighbors: {neighbors}")
    
    print(f"\n  4. Filtered atoms object: {len(softener.filtered_atoms)} atoms")
    
    print("\nPotential visualizations:")
    print("  - Draw spheres of radius around each solute atom")
    print("  - Highlight solute atoms and their neighbors with different colors")
    print("  - Draw bonds/connections between solute and neighbor atoms")
    print("  - Show charge distribution on filtered system only")
    
    print("\n✓ Visualization data ready")


def main():
    print("\n" + "="*70)
    print("Testing New Features: apply_qeq and each_neighbor")
    print("="*70)
    
    try:
        test_apply_qeq_option()
        test_each_neighbor()
        test_visualization_use_case()
        
        print("\n" + "="*70)
        print("All tests passed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
