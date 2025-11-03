# qsolv_softener

Charge softening for solute atoms based on solvent environment with QEq (Charge Equilibration) integration.

## Overview

`qsolv_softener` is a Python package that softens partial charges on solute atoms by incorporating information from neighboring solvent atoms within a specified radius. After softening, it applies Charge Equilibration (QEq) to maintain the total system charge while allowing charges to re-equilibrate based on electronegativity and hardness parameters.

## Features

- **Distance-weighted charge softening**: Multiple weight functions (inverse, Gaussian, inverse square)
- **QEq integration**: Maintains total charge conservation (within 1e-6 tolerance)
- **Flexible parameterization**: Customizable radius, mixing parameter (α), and weight function parameters
- **ASE integration**: Works seamlessly with ASE Atoms objects
- **Efficient neighbor search**: Uses KDTree for O(log N) neighbor queries
- **Filtered atoms**: Automatically creates a reduced system containing only solute + solvent atoms within the specified radius

## Installation

Using `uv` (recommended):

```bash
uv pip install -e .
```

Or with pip:

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
from ase import Atoms
from qsolv_softener import ChargeSoftener

# Create your system (solute + solvent)
atoms = Atoms(...)  # Your ASE Atoms object

# Define solute indices and charges
solute_indices = [0, 1, 2]  # Indices of solute atoms
solute_charges = np.array([0.5, -0.3, 0.2])
solvent_charges = np.array([-0.8, 0.4, -0.8, 0.4, ...])

# Create softener and run
softener = ChargeSoftener(
    atoms=atoms,
    solute_indices=solute_indices,
    solute_charges=solute_charges,
    solvent_charges=solvent_charges,
    radius=5.0,  # Search radius in Angstroms
    alpha=0.5     # Mixing parameter
)

updated_atoms = softener.run()  # Default: apply_qeq=True
final_charges = updated_atoms.get_initial_charges()

# Access filtered atoms (solute + solvent within radius)
print(f"Filtered system: {len(softener.filtered_atoms)} atoms")
print(f"Filtered solvent indices: {softener.filtered_solvent_indices}")

# Access neighbor information for each solute atom
for i, neighbors in enumerate(softener.each_neighbor):
    print(f"Solute atom {i} has {len(neighbors)} neighbors: {neighbors}")

# Run without QEq (only softening)
updated_atoms_no_qeq = softener.run(apply_qeq=False)
```

## Weight Functions

Three weight functions are supported:

1. **Inverse Distance** (default): `w(r) = 1/r`
   ```python
   softener = ChargeSoftener(..., weight_func='inverse')
   ```

2. **Gaussian**: `w(r) = exp(-r²/(2σ²))`
   ```python
   softener = ChargeSoftener(
       ..., 
       weight_func='gaussian',
       weight_params={'sigma': 1.5}
   )
   ```

3. **Inverse Square**: `w(r) = 1/r²`
   ```python
   softener = ChargeSoftener(..., weight_func='inverse_square')
   ```

## Parameters

- `atoms`: ASE Atoms object with solute + solvent
- `solute_indices`: List or array of solute atom indices
- `solute_charges`: Array of initial solute charges
- `solvent_charges`: Array of initial solvent charges
- `radius`: Search radius for neighbors (default: 5.0 Å)
- `weight_func`: Weight function name (default: 'inverse')
- `weight_params`: Dict of parameters for weight function (default: {})
- `alpha`: Mixing parameter, 0 = full softening, 1 = no softening (default: 0.5)
- `qeq_params`: Custom QEq parameters (default: uses built-in values)

## Examples

See `examples/soften_example.py` for comprehensive examples including:
- Basic softening with default parameters
- Gaussian weight function usage
- Comparing different α values
- Custom QEq parameters

Run the example:

```bash
cd examples
python soften_example.py
```

Or with uv:

```bash
uv run examples/soften_example.py
```

## QEq Parameters

Default QEq parameters are provided for common elements (H, C, N, O, F, P, S, Cl, Br, I). You can override these:

```python
custom_qeq = {
    'C': {'chi': 2.6, 'eta': 10.5},
    'O': {'chi': 3.4, 'eta': 11.5},
    'H': {'chi': 2.3, 'eta': 13.5}
}

softener = ChargeSoftener(..., qeq_params=custom_qeq)
```

## Filtered Atoms

After running the softening workflow, the `ChargeSoftener` object provides access to a filtered system containing only the solute atoms and the solvent atoms that were within the specified radius:

```python
softener = ChargeSoftener(...)
updated_atoms = softener.run()

# Access filtered solvent indices (within radius)
print(softener.filtered_solvent_indices)

# Access filtered atoms object (solute + filtered solvent)
filtered_system = softener.filtered_atoms
print(f"Reduced system: {len(filtered_system)} atoms")

# The filtered_atoms object is a regular ASE Atoms object
# with charges already equilibrated - useful for:
# - Further QM calculations on a reduced system
# - Analysis of the local environment
# - Saving only relevant atoms for visualization
```

This is particularly useful for large systems where you only want to perform expensive calculations on the atoms that actually contributed to the charge softening.

## Optional QEq Equilibration

The `run()` method accepts an `apply_qeq` parameter to control whether QEq equilibration is applied:

```python
softener = ChargeSoftener(...)

# Default: apply QEq after softening
updated_atoms = softener.run(apply_qeq=True)

# Skip QEq: only apply softening
softened_only = softener.run(apply_qeq=False)
```

When `apply_qeq=False`, the method returns atoms with only softened charges (without re-equilibration). This can be useful for:
- Analyzing the effect of softening separately from QEq
- Faster calculations when charge conservation is not required
- Custom post-processing workflows

## Neighbor Information for Visualization

The `each_neighbor` attribute provides a list of neighbor indices for each solute atom in order, useful for visualization and analysis:

```python
softener = ChargeSoftener(...)
updated_atoms = softener.run()

# each_neighbor[i] contains neighbor indices for solute atom i
for i, (solute_idx, neighbors) in enumerate(zip(softener.solute_indices, softener.each_neighbor)):
    if len(neighbors) > 0:
        print(f"Solute atom {i} (index {solute_idx}): {len(neighbors)} neighbors")
        print(f"  Neighbor indices: {neighbors}")
    else:
        print(f"Solute atom {i} (index {solute_idx}): No neighbors within radius")

# Use for visualization:
# - Draw spheres around solute atoms showing their influence radius
# - Highlight neighbor atoms with different colors
# - Draw connections between solute and neighbor atoms
# - Analyze local environment around each solute atom
```

Each element in `each_neighbor` is a list of solvent atom indices within the radius for that solute atom. Empty lists indicate no neighbors were found.

## Charge Conservation

The package guarantees charge conservation within 1e-6 tolerance. The total charge before and after the softening + QEq workflow is preserved.

## Requirements

- Python >= 3.8
- ASE >= 3.22.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.
