# qsolv_softener Implementation Plan

## Overview of Softening Algorithm

The `qsolv_softener` package implements a two-stage charge modification algorithm:

1. **Charge Softening Stage**: For each solute atom, we compute a distance-weighted average of neighboring solvent charges within a specified radius. The solute charge is then updated using a mixing parameter α:
   ```
   new_q_solute = α * original_q_solute + (1 - α) * weighted_avg_q_solvent
   ```
   Solvent charges remain **unchanged** during this stage, acting as a fixed background field. During this stage, the algorithm also collects all solvent atom indices that fall within the radius of any solute atom.

2. **Charge Equilibration (QEq) Stage**: After softening, we apply QEq to the entire system (solute + solvent) to re-equilibrate charges while preserving the total system charge. This uses electronegativity (χ) and hardness (η) parameters per element.

3. **Filtered Atoms Creation**: After QEq, the algorithm creates a `filtered_atoms` object containing only the solute atoms and the solvent atoms that were within the specified radius. This provides a reduced system for further analysis or calculations.

## Weight Function Details

The package supports multiple weight functions for computing distance-weighted averages:

### Supported Weight Functions

1. **Inverse Distance** (`weight_func='inverse'`)
   - Formula: `w(r) = 1 / r`
   - Parameters: None (or optional `{'min_distance': 0.1}` to avoid division by zero)
   - Use case: Standard distance weighting, emphasizes nearby atoms
   
2. **Gaussian** (`weight_func='gaussian'`)
   - Formula: `w(r) = exp(-r² / (2σ²))`
   - Parameters: `{'sigma': float}` (default: σ = 1.0 Å)
   - Use case: Smooth falloff, adjustable width of influence region

3. **Inverse Square** (`weight_func='inverse_square'`)
   - Formula: `w(r) = 1 / r²`
   - Parameters: None (or optional `{'min_distance': 0.1}`)
   - Use case: Stronger distance dependence than inverse

### Parameter Specification

Weight function parameters are passed as a dictionary to the `ChargeSoftener` constructor:

```python
# Default inverse distance
softener = ChargeSoftener(..., weight_func='inverse')

# Gaussian with custom sigma
softener = ChargeSoftener(..., weight_func='gaussian', weight_params={'sigma': 2.0})

# Inverse with minimum distance cutoff
softener = ChargeSoftener(..., weight_func='inverse', weight_params={'min_distance': 0.2})
```

## Data Flow Diagram

```
                    Input Data
                        |
    +-------------------+-------------------+
    |                   |                   |
Atoms Object    Solute Indices    Charge Arrays
 (positions)    (list/array)      (solute_q, solvent_q)
    |                   |                   |
    +-------------------+-------------------+
                        |
                        v
              +---------+---------+
              | Initialize System |
              |  - Set charges    |
              |  - Build kdtree   |
              +---------+---------+
                        |
                        v
              +---------+---------+
              | Softening Loop    |
              | For each solute:  |
              |  - Find neighbors |
              |  - Collect indices|
              |  - Compute weights|
              |  - Mix charges    |
              +---------+---------+
                        |
                        v
              +---------+---------+
              |   QEq Solver      |
              | - Build matrices  |
              | - Solve system    |
              | - Update charges  |
              +---------+---------+
                        |
                        v
              +---------+---------+
              | Create Filtered   |
              | Atoms Object      |
              | (solute+filtered) |
              +---------+---------+
                        |
                        v
              +---------+---------+
              | Updated Atoms     |
              | (new charges)     |
              | + filtered_atoms  |
              +-------------------+
```

## Usage Examples

### Basic Usage (Default Inverse Distance)

```python
from ase import Atoms
from qsolv_softener import ChargeSoftener
import numpy as np

# Create system
atoms = Atoms(...)  # Combined solute + solvent
solute_indices = [0, 1, 2]  # First 3 atoms are solute
solute_charges = np.array([0.5, -0.3, 0.2])
solvent_charges = np.array([-0.8, 0.4, -0.8, 0.4, ...])  # Water molecules

# Default: inverse distance, alpha=0.5, radius=5.0 Å
softener = ChargeSoftener(
    atoms=atoms,
    solute_indices=solute_indices,
    solute_charges=solute_charges,
    solvent_charges=solvent_charges,
    radius=5.0
)

updated_atoms = softener.run()
print(f"New charges: {updated_atoms.get_initial_charges()}")

# Access filtered atoms (solute + solvent within radius)
print(f"Filtered system: {len(softener.filtered_atoms)} atoms")
print(f"Filtered solvent indices: {softener.filtered_solvent_indices}")
```

### Using Gaussian Weight Function

```python
softener = ChargeSoftener(
    atoms=atoms,
    solute_indices=solute_indices,
    solute_charges=solute_charges,
    solvent_charges=solvent_charges,
    radius=6.0,
    weight_func='gaussian',
    weight_params={'sigma': 1.5},  # σ = 1.5 Å
    alpha=0.6  # More weight on original charge
)

updated_atoms = softener.run()
```

### Custom QEq Parameters

```python
# Override default QEq parameters for specific elements
custom_qeq = {
    'C': {'chi': 2.5, 'eta': 10.0},
    'O': {'chi': 3.5, 'eta': 12.0},
    'H': {'chi': 2.2, 'eta': 13.0},
    'N': {'chi': 3.0, 'eta': 11.0}
}

softener = ChargeSoftener(
    atoms=atoms,
    solute_indices=solute_indices,
    solute_charges=solute_charges,
    solvent_charges=solvent_charges,
    radius=5.0,
    qeq_params=custom_qeq
)

updated_atoms = softener.run()
```

### Combining All Features

```python
softener = ChargeSoftener(
    atoms=atoms,
    solute_indices=solute_indices,
    solute_charges=solute_charges,
    solvent_charges=solvent_charges,
    radius=6.5,
    weight_func='gaussian',
    weight_params={'sigma': 2.0},
    alpha=0.7,
    qeq_params=custom_qeq
)

updated_atoms = softener.run()

# Verify charge conservation
initial_charge = np.sum(solute_charges) + np.sum(solvent_charges)
final_charge = np.sum(updated_atoms.get_initial_charges())
assert abs(initial_charge - final_charge) < 1e-6, "Charge not conserved!"
```

## QEq Parameter Handling

### Default QEq Parameters

The package provides default electronegativity (χ) and hardness (η) values for common elements:

```python
DEFAULT_QEQ_PARAMS = {
    'H': {'chi': 2.2, 'eta': 13.0},
    'C': {'chi': 2.5, 'eta': 10.0},
    'N': {'chi': 3.0, 'eta': 11.0},
    'O': {'chi': 3.5, 'eta': 12.0},
    'F': {'chi': 4.0, 'eta': 14.0},
    'P': {'chi': 2.1, 'eta': 8.5},
    'S': {'chi': 2.3, 'eta': 9.0},
    'Cl': {'chi': 3.0, 'eta': 11.0}
}
```

### QEq Algorithm

The QEq method solves the following linear system:

```
| 2η₁   0    0   ... 1 | | Δq₁ |   | -χ₁ |
| 0    2η₂   0   ... 1 | | Δq₂ |   | -χ₂ |
| 0     0   2η₃  ... 1 | | Δq₃ | = | -χ₃ |
| ...  ...  ...  ... 1 | | ... |   | ... |
| 1     1    1   ... 0 | | λ   |   | Q_tot|
```

Where:
- ηᵢ = hardness of atom i
- χᵢ = electronegativity of atom i
- Δqᵢ = charge adjustment for atom i
- λ = Lagrange multiplier (electrochemical potential)
- Q_tot = total system charge (constraint)

The final charge is: q_final = q_softened + Δq

## Testing Strategy

### Unit Tests Structure

```bash
tests/
  ├── __init__.py
  ├── test_weight_functions.py    # Test each weight function
  ├── test_softening.py            # Test charge softening logic
  ├── test_qeq.py                  # Test QEq solver
  └── test_integration.py          # Full workflow tests
```

### Running Tests with uv

```bash
# Install package in development mode
uv pip install -e .

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=qsolv_softener --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_integration.py -v
```

### Test Cases

1. **Weight Function Tests**:
   - Verify inverse distance: w(1.0) = 1.0, w(2.0) = 0.5
   - Verify Gaussian: w(0) = 1.0, w(σ) = exp(-0.5)
   - Test minimum distance handling

2. **Softening Tests**:
   - Single solute atom with uniform solvent: softened charge should equal solvent average
   - No neighbors case: charge should remain original
   - Alpha parameter: α=0 → full softening, α=1 → no softening

3. **QEq Tests**:
   - Charge conservation: sum(q_final) == sum(q_initial) within 1e-6
   - Single atom: should return to initial charge
   - Two atoms with same element: charges should equilibrate

4. **Integration Tests**:
   - Full workflow with toy system (methanol in water)
   - Verify all constraints: charge conservation, solvent unchanged during softening

## Potential Edge Cases

### 1. No Neighbors Found

**Scenario**: A solute atom has no solvent neighbors within the radius.

**Handling**: 
- During softening, keep the original solute charge unchanged
- Log a warning if verbose mode is enabled
- Implementation:
  ```python
  if len(neighbors) == 0:
      softened_charges[i] = original_charges[i]  # No change
      continue
  ```

### 2. Zero Distance

**Scenario**: Solute and solvent atoms overlap (r ≈ 0).

**Handling**:
- For inverse distance weights, use `min_distance` parameter (default 0.1 Å)
- Implementation: `r_effective = max(r, min_distance)`
- For Gaussian, no issue (always finite)

### 3. Charge Sum Mismatch After QEq

**Scenario**: Numerical errors accumulate in QEq solver.

**Handling**:
- Check charge conservation: |Q_final - Q_initial| < tolerance
- Raise error if tolerance exceeded (1e-6)
- Consider iterative refinement if needed

### 4. Missing QEq Parameters

**Scenario**: Atoms object contains elements not in qeq_params dict.

**Handling**:
- Raise informative error with element symbol
- Suggest adding parameters to the dict
- Implementation:
  ```python
  if element not in qeq_params:
      raise ValueError(f"QEq parameters not found for element '{element}'")
  ```

### 5. Numerical Instability in QEq

**Scenario**: QEq matrix is singular or ill-conditioned.

**Handling**:
- Use `numpy.linalg.lstsq` instead of `solve` for robustness
- Check condition number of matrix
- Fallback: skip QEq and return softened charges with warning

### 6. Empty Solute or Solvent

**Scenario**: solute_indices is empty or solvent has no atoms.

**Handling**:
- Validate inputs in `__init__`
- Raise `ValueError` with clear message
- Implementation:
  ```python
  if len(solute_indices) == 0:
      raise ValueError("solute_indices cannot be empty")
  if len(solvent_charges) == 0:
      raise ValueError("solvent_charges cannot be empty")
  ```

## Performance Considerations

1. **Neighbor List**: Use SciPy's KDTree for efficient spatial queries (O(log N) per query)
2. **QEq Solver**: Use NumPy's optimized linear algebra (LAPACK backend)
3. **Memory**: Avoid creating unnecessary copies of large arrays
4. **Scaling**: Algorithm is O(N_solute * k) for softening + O(N_total³) for QEq, where k is average neighbors

## Dependencies and Version Requirements

```toml
[project]
dependencies = [
    "ase>=3.22.0",
    "numpy>=1.20.0",
    "scipy>=1.7.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0"
]
```

## Future Enhancements

1. **Adaptive Radius**: Automatically determine optimal radius per solute atom
2. **GPU Acceleration**: Use CuPy for large systems
3. **Iterative QEq**: Support iterative solvers (conjugate gradient) for very large systems
4. **Custom Weight Functions**: Allow user-defined weight functions via callback
5. **Visualization**: Plot charge distribution before/after softening
