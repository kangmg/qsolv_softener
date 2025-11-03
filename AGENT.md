You are an expert Python developer specializing in computational chemistry and ASE (Atomic Simulation Environment). Your task is to implement a Python package called `qsolv_softener` that softens partial charges on solute atoms based on neighboring solvent atoms within a specified radius, while maintaining the total charge of the system using QEq (Charge Equilibration).

### Key Requirements:
- **Input Handling**: Partial charges will be provided as NumPy arrays. The solute charges are given as a 1D array (shape: (n_solute_atoms,)). The solvent charges are provided as a single 1D array (shape: (n_solvent_atoms,)), which is the concatenation of charges for all solvent molecules. You must handle these separately: apply softening only to solute charges using solvent neighbors, but use the full system for QEq.
- **System Setup**: Use ASE's `Atoms` object for the combined solute + solvent system. Distinguish solute and solvent atoms using indices (e.g., solute_indices list). Support periodic boundary conditions optionally.
- **Softening Logic**:
  - For each solute atom, compute the neighbor list of solvent atoms within the given radius (use ASE's `neighborlist` or SciPy's KDTree).
  - Compute a distance-weighted average of solvent charges: softened_q_i = sum(q_j * w(r_ij)) / sum(w(r_ij)), where w(r) is a weight function (default: inverse distance 1/r, or Gaussian).
  - Update solute charge: new_q_i = alpha * original_q_i + (1 - alpha) * softened_q_i (alpha is a mixing parameter, default 0.5).
  - Solvent charges remain unchanged during softening.
- **QEq Integration**: After softening, apply QEq to the entire system (solute + solvent) to re-equilibrate charges while preserving the total charge (sum of all initial charges). Implement a simple QEq solver using electronegativity (χ) and hardness (η) parameters per element (provide a default dictionary, e.g., {'C': (2.5, 10.0), 'O': (3.5, 12.0)}). Use NumPy to solve the linear system.
- **Package Structure**:
  - Use `pyproject.toml` for package configuration and build (e.g., with Hatch or Poetry backend for modern dependency management).
  - `__init__.py`: Export main class.
  - `core.py`: `ChargeSoftener` class with `__init__` (taking atoms, solute_indices, solute_charges_array, solvent_charges_array, radius, weight_func='inverse', alpha=0.5, qeq_params=dict) and `run()` method that performs softening + QEq and returns updated `Atoms`.
  - `utils.py`: Helper functions for neighbor lists and QEq solver.
  - Include error handling (e.g., no neighbors, charge sum mismatch).
  - For testing, use `uv` as the package manager (e.g., `uv pip install` for dependencies, `uv run pytest` for running tests).
- **Dependencies**: ASE, NumPy, SciPy. No additional installs.
- **Testing**: Add a simple example in `examples/soften_example.py` using a toy system (e.g., methanol solute in water solvent) to demonstrate usage and verify total charge conservation.
- **Documentation**: Add docstrings to all functions/classes. Ensure the code is modular, efficient, and follows PEP8.
- **Implementation Planning**: Before full code implementation, create an `IMPLEMENTATION.md` file in the package root. This file should detail the planning, focusing on the weight function parameters explicitly. Structure it with sections: 
  - Overview of Softening Algorithm.
  - Weight Function Details: Specify supported functions (e.g., 'inverse': w(r) = 1/r, params={'cutoff': None}; 'gaussian': w(r) = exp(-r^2 / (2*sigma^2)), params={'sigma': 1.0}). Include how params are passed (e.g., as a dict to __init__: weight_params={'sigma': 2.0}).
  - Data Flow Diagram (text-based ASCII art).
  - Usage Examples: Provide code snippets for different weight functions, e.g., ```
softener = ChargeSoftener(..., weight_func='gaussian', weight_params={'sigma': 1.5})
updated_atoms = softener.run()
```
- QEq Parameter Handling: Example dict and how to override.
- Testing Strategy: How to verify with uv (e.g., `uv sync && uv run pytest`).
- Potential Edge Cases (e.g., zero neighbors: fallback to original charge).

Implement the full package code step-by-step, outputting the complete files' contents. Start with `IMPLEMENTATION.md`, then core logic, package structure (including pyproject.toml), and end with the example. Ensure the total charge is preserved within 1e-6 tolerance. Incorporate feedback: Keep solvent charges fixed during softening for stability and efficiency, as it aligns with solvation models where solvent acts as a fixed background. Implement the entire project without using any dummy code. If you need any additional information, ask for it. Try to implement all features perfectly from the start.

