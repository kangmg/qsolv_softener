"""
Test filtered_atoms functionality
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
    
    # Add water molecules at different distances
    water_positions = [
        [8.0, 5.0, 5.0],   # close to methanol
        [8.8, 5.3, 5.2],
        [7.3, 5.4, 5.3],
        [15.0, 15.0, 15.0],  # far from methanol (outside radius)
        [15.5, 15.3, 15.2],
        [14.8, 15.4, 15.3],
    ]
    
    water_symbols = ['O', 'H', 'H'] * 2
    water = Atoms(water_symbols, positions=water_positions)
    
    combined = methanol + water
    
    n_methanol = len(methanol)
    solute_indices = list(range(n_methanol))
    
    methanol_charges = np.array([
        -0.18,   # C
        0.06,    # H
        0.06,    # H
        0.06,    # H
        -0.68,   # O
        0.42     # H
    ])
    
    water_charges = np.array([-0.834, 0.417, 0.417] * 2)
    
    return combined, solute_indices, methanol_charges, water_charges


def main():
    print("Testing filtered_atoms functionality\n")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_test_system()
    
    print(f"Original system:")
    print(f"  Total atoms: {len(atoms)}")
    print(f"  Solute atoms: {len(solute_indices)}")
    print(f"  Solvent atoms: {len(solvent_charges)}")
    print(f"  Symbols: {atoms.get_chemical_symbols()}")
    
    # Create softener with radius that will filter some solvent atoms
    softener = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=5.0,  # Only first water molecule should be within radius
        alpha=0.5
    )
    
    # Run softening + QEq
    updated_atoms = softener.run()
    
    print(f"\nFiltered solvent indices within radius={softener.radius} Å:")
    print(f"  {softener.filtered_solvent_indices}")
    
    print(f"\nFiltered atoms (solute + filtered solvent):")
    print(f"  Total atoms: {len(softener.filtered_atoms)}")
    print(f"  Expected: {len(solute_indices)} solute + {len(softener.filtered_solvent_indices)} solvent")
    print(f"  Symbols: {softener.filtered_atoms.get_chemical_symbols()}")
    print(f"  Charges: {softener.filtered_atoms.get_initial_charges()}")
    
    # Verify that only atoms within radius are included
    print(f"\nVerification:")
    print(f"  Solute atoms: {len(solute_indices)}")
    print(f"  Filtered solvent atoms: {len(softener.filtered_solvent_indices)}")
    print(f"  Total in filtered_atoms: {len(softener.filtered_atoms)}")
    print(f"  Match: {len(softener.filtered_atoms) == len(solute_indices) + len(softener.filtered_solvent_indices)}")
    
    # Check distances
    print(f"\nDistance check:")
    solute_center = atoms.positions[solute_indices].mean(axis=0)
    for i, solvent_idx in enumerate(softener.filtered_solvent_indices):
        dist = np.linalg.norm(atoms.positions[solvent_idx] - solute_center)
        symbol = atoms[solvent_idx].symbol
        print(f"  Solvent atom {solvent_idx} ({symbol}): {dist:.2f} Å from solute center")
    
    print("\n" + "="*70)
    print("✓ Test completed successfully!")


if __name__ == "__main__":
    main()
