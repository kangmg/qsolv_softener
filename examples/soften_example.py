"""
Example demonstrating charge softening with methanol solute in water solvent.

This example creates a toy system with a methanol molecule (CH3OH) as the solute
surrounded by water molecules as the solvent. It demonstrates:
1. Setting up the system with initial charges
2. Applying charge softening with different weight functions
3. Verifying charge conservation after QEq
"""

import numpy as np
from ase import Atoms
from ase.build import molecule
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from qsolv_softener import ChargeSoftener, DEFAULT_QEQ_PARAMS


def create_methanol_water_system():
    """
    Create a toy system with methanol solute in water solvent.
    
    Returns:
        Tuple of (atoms, solute_indices, solute_charges, solvent_charges)
    """
    methanol = molecule('CH3OH')
    methanol.center(vacuum=5.0)
    
    water_positions = [
        [8.0, 5.0, 5.0],
        [8.8, 5.3, 5.2],
        [7.3, 5.4, 5.3],
        [5.0, 8.0, 5.0],
        [5.3, 8.7, 5.3],
        [4.5, 8.2, 5.6],
        [2.0, 5.0, 5.0],
        [2.5, 5.7, 5.2],
        [1.2, 5.3, 5.3],
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


def print_charge_summary(label, charges, solute_indices):
    """Print summary of charges."""
    print(f"\n{label}:")
    print(f"  Total charge: {np.sum(charges):.6f}")
    print(f"  Solute charges: {charges[solute_indices]}")
    print(f"  Solute total: {np.sum(charges[solute_indices]):.6f}")
    print(f"  Charge range: [{np.min(charges):.4f}, {np.max(charges):.4f}]")


def example_basic_softening():
    """Example 1: Basic softening with default parameters."""
    print("="*70)
    print("Example 1: Basic Charge Softening (Default Inverse Distance)")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_methanol_water_system()
    
    print(f"\nSystem: {len(solute_indices)} solute atoms (methanol), "
          f"{len(solvent_charges)} solvent atoms (water)")
    
    initial_charges = np.zeros(len(atoms))
    initial_charges[solute_indices] = solute_charges
    solvent_indices = [i for i in range(len(atoms)) if i not in solute_indices]
    initial_charges[solvent_indices] = solvent_charges
    
    print_charge_summary("Initial charges", initial_charges, solute_indices)
    
    softener = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=5.0,
        weight_func='inverse',
        alpha=0.5
    )
    
    softened = softener.get_softened_charges()
    print_charge_summary("After softening (before QEq)", softened, solute_indices)
    
    updated_atoms = softener.run()
    final_charges = updated_atoms.get_initial_charges()
    print_charge_summary("Final charges (after QEq)", final_charges, solute_indices)
    
    initial_total = np.sum(initial_charges)
    final_total = np.sum(final_charges)
    print(f"\nCharge conservation check:")
    print(f"  Initial total: {initial_total:.8f}")
    print(f"  Final total:   {final_total:.8f}")
    print(f"  Difference:    {abs(final_total - initial_total):.2e}")
    
    assert abs(final_total - initial_total) < 1e-6, "Charge conservation failed!"
    print("  ✓ Charge conserved within tolerance")


def example_gaussian_weighting():
    """Example 2: Softening with Gaussian weight function."""
    print("\n" + "="*70)
    print("Example 2: Gaussian Weight Function")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_methanol_water_system()
    
    initial_charges = np.zeros(len(atoms))
    initial_charges[solute_indices] = solute_charges
    solvent_indices = [i for i in range(len(atoms)) if i not in solute_indices]
    initial_charges[solvent_indices] = solvent_charges
    
    print(f"\nUsing Gaussian weight function with σ = 1.5 Å")
    
    softener = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=6.0,
        weight_func='gaussian',
        weight_params={'sigma': 1.5},
        alpha=0.6
    )
    
    updated_atoms = softener.run()
    final_charges = updated_atoms.get_initial_charges()
    
    print_charge_summary("Final charges", final_charges, solute_indices)
    
    initial_total = np.sum(initial_charges)
    final_total = np.sum(final_charges)
    print(f"\nCharge conservation: {abs(final_total - initial_total):.2e}")
    assert abs(final_total - initial_total) < 1e-6, "Charge conservation failed!"
    print("✓ Charge conserved within tolerance")


def example_parameter_comparison():
    """Example 3: Compare different alpha values."""
    print("\n" + "="*70)
    print("Example 3: Comparing Different Alpha Values")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_methanol_water_system()
    
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("\nAlpha parameter controls mixing:")
    print("  α = 0.0 → Full softening (100% solvent influence)")
    print("  α = 1.0 → No softening (100% original charge)")
    
    for alpha in alphas:
        softener = ChargeSoftener(
            atoms=atoms,
            solute_indices=solute_indices,
            solute_charges=solute_charges,
            solvent_charges=solvent_charges,
            radius=5.0,
            alpha=alpha
        )
        
        updated_atoms = softener.run()
        final_charges = updated_atoms.get_initial_charges()
        solute_final = final_charges[solute_indices]
        
        change = np.abs(solute_final - solute_charges)
        avg_change = np.mean(change)
        
        print(f"\n  α = {alpha:.2f}: Average charge change = {avg_change:.4f}")
        print(f"    Solute charges: {solute_final}")


def example_custom_qeq_params():
    """Example 4: Using custom QEq parameters."""
    print("\n" + "="*70)
    print("Example 4: Custom QEq Parameters")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_methanol_water_system()
    
    custom_qeq = {
        'C': {'chi': 2.6, 'eta': 10.5},
        'O': {'chi': 3.4, 'eta': 11.5},
        'H': {'chi': 2.3, 'eta': 13.5}
    }
    
    print("\nUsing custom QEq parameters:")
    for element, params in custom_qeq.items():
        print(f"  {element}: χ = {params['chi']}, η = {params['eta']}")
    
    softener = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=5.0,
        qeq_params=custom_qeq
    )
    
    updated_atoms = softener.run()
    final_charges = updated_atoms.get_initial_charges()
    
    print_charge_summary("Final charges", final_charges, solute_indices)
    
    initial_total = np.sum(solute_charges) + np.sum(solvent_charges)
    final_total = np.sum(final_charges)
    print(f"\nCharge conservation: {abs(final_total - initial_total):.2e}")
    assert abs(final_total - initial_total) < 1e-6, "Charge conservation failed!"
    print("✓ Charge conserved within tolerance")


def example_filtered_atoms():
    """Example 5: Accessing filtered atoms (solute + solvent within radius)."""
    print("\n" + "="*70)
    print("Example 5: Filtered Atoms (Solute + Solvent within Radius)")
    print("="*70)
    
    atoms, solute_indices, solute_charges, solvent_charges = create_methanol_water_system()
    
    print(f"\nOriginal system:")
    print(f"  Total atoms: {len(atoms)}")
    print(f"  Solute atoms: {len(solute_indices)}")
    print(f"  Solvent atoms: {len(solvent_charges)}")
    
    softener = ChargeSoftener(
        atoms=atoms,
        solute_indices=solute_indices,
        solute_charges=solute_charges,
        solvent_charges=solvent_charges,
        radius=5.0
    )
    
    updated_atoms = softener.run()
    
    print(f"\nFiltered system (solute + solvent within {softener.radius} Å):")
    print(f"  Total atoms: {len(softener.filtered_atoms)}")
    print(f"  Solute atoms: {len(solute_indices)}")
    print(f"  Filtered solvent atoms: {len(softener.filtered_solvent_indices)}")
    print(f"  Filtered solvent indices: {softener.filtered_solvent_indices}")
    print(f"  Chemical symbols: {softener.filtered_atoms.get_chemical_symbols()}")
    
    print(f"\nFiltered atoms object can be used for further analysis or calculations:")
    print(f"  - Reduced system size for expensive computations")
    print(f"  - Only atoms that contributed to charge softening")
    print(f"  - Charges already equilibrated with QEq")
    
    print(f"\n✓ Filtered atoms created successfully")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("qsolv_softener: Charge Softening Examples")
    print("="*70)
    
    try:
        example_basic_softening()
        example_gaussian_weighting()
        example_parameter_comparison()
        example_custom_qeq_params()
        example_filtered_atoms()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
