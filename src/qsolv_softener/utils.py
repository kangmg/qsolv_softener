"""
Utility functions for charge softening and QEq equilibration.
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Dict, Callable, Tuple, List


DEFAULT_QEQ_PARAMS = {
    'H': {'chi': 4.528, 'eta': 13.8904},
    'He': {'chi': 9.66, 'eta': 29.8400},
    'Li': {'chi': 3.006, 'eta': 4.7720},
    'Be': {'chi': 4.877, 'eta': 8.8860},
    'B': {'chi': 5.11, 'eta': 9.5000},
    'C': {'chi': 5.343, 'eta': 10.1260},
    'N': {'chi': 6.899, 'eta': 11.7600},
    'O': {'chi': 8.741, 'eta': 13.3640},
    'F': {'chi': 10.874, 'eta': 14.9480},
    'Ne': {'chi': 11.04, 'eta': 21.1000},
    'Na': {'chi': 2.843, 'eta': 4.5920},
    'Mg': {'chi': 3.951, 'eta': 7.3860},
    'Al': {'chi': 4.06, 'eta': 7.1800},
    'Si': {'chi': 4.168, 'eta': 6.9740},
    'P': {'chi': 5.463, 'eta': 8.0000},
    'S': {'chi': 6.928, 'eta': 8.9720},
    'Cl': {'chi': 8.564, 'eta': 9.8920},
    'Ar': {'chi': 9.465, 'eta': 12.7100},
    'K': {'chi': 2.421, 'eta': 3.8400},
    'Ca': {'chi': 3.231, 'eta': 5.7600},
    'Sc': {'chi': 3.395, 'eta': 6.1600},
    'Ti': {'chi': 3.47, 'eta': 6.7600},
    'V': {'chi': 3.65, 'eta': 6.8200},
    'Cr': {'chi': 3.415, 'eta': 7.7300},
    'Mn': {'chi': 3.325, 'eta': 8.2100},
    'Fe': {'chi': 3.76, 'eta': 8.2800},
    'Co': {'chi': 4.105, 'eta': 8.3500},
    'Ni': {'chi': 4.465, 'eta': 8.4100},
    'Cu': {'chi': 4.2, 'eta': 8.4400},
    'Zn': {'chi': 5.106, 'eta': 8.5700},
    'Ga': {'chi': 3.641, 'eta': 6.3200},
    'Ge': {'chi': 4.051, 'eta': 6.8760},
    'As': {'chi': 5.188, 'eta': 7.6180},
    'Se': {'chi': 6.428, 'eta': 8.2620},
    'Br': {'chi': 7.79, 'eta': 8.8500},
    'Kr': {'chi': 8.505, 'eta': 11.4300},
    'Rb': {'chi': 2.331, 'eta': 3.6920},
    'Sr': {'chi': 3.024, 'eta': 4.8800},
    'Y': {'chi': 3.83, 'eta': 5.6200},
    'Zr': {'chi': 3.4, 'eta': 7.1000},
    'Nb': {'chi': 3.55, 'eta': 6.7600},
    'Mo': {'chi': 3.465, 'eta': 7.5100},
    'Tc': {'chi': 3.29, 'eta': 7.9800},
    'Ru': {'chi': 3.575, 'eta': 8.0300},
    'Rh': {'chi': 3.975, 'eta': 8.0100},
    'Pd': {'chi': 4.32, 'eta': 8.0000},
    'Ag': {'chi': 4.436, 'eta': 6.2680},
    'Cd': {'chi': 5.034, 'eta': 7.9140},
    'In': {'chi': 3.506, 'eta': 5.7920},
    'Sn': {'chi': 3.987, 'eta': 6.2480},
    'Sb': {'chi': 4.899, 'eta': 6.6840},
    'Te': {'chi': 5.816, 'eta': 7.0520},
    'I': {'chi': 6.822, 'eta': 7.5240},
    'Xe': {'chi': 7.595, 'eta': 9.9500},
    'Cs': {'chi': 2.183, 'eta': 3.4220},
    'Ba': {'chi': 2.814, 'eta': 4.7920},
    'La': {'chi': 2.8355, 'eta': 5.4830},
    'Ce': {'chi': 2.774, 'eta': 5.3840},
    'Pr': {'chi': 2.858, 'eta': 5.1280},
    'Nd': {'chi': 2.8685, 'eta': 5.2410},
    'Pm': {'chi': 2.881, 'eta': 5.3460},
    'Sm': {'chi': 2.9115, 'eta': 5.4390},
    'Eu': {'chi': 2.8785, 'eta': 5.5750},
    'Gd': {'chi': 3.1665, 'eta': 5.9490},
    'Tb': {'chi': 3.018, 'eta': 5.6680},
    'Dy': {'chi': 3.0555, 'eta': 5.7430},
    'Ho': {'chi': 3.127, 'eta': 5.7820},
    'Er': {'chi': 3.1865, 'eta': 5.8290},
    'Tm': {'chi': 3.2514, 'eta': 5.8658},
    'Yb': {'chi': 3.2889, 'eta': 5.9300},
    'Lu': {'chi': 2.9629, 'eta': 4.9258},
    'Hf': {'chi': 3.7, 'eta': 6.8000},
    'Ta': {'chi': 5.1, 'eta': 5.7000},
    'W': {'chi': 4.63, 'eta': 6.6200},
    'Re': {'chi': 3.96, 'eta': 7.8400},
    'Os': {'chi': 5.14, 'eta': 7.2600},
    'Ir': {'chi': 5.0, 'eta': 8.0000},
    'Pt': {'chi': 4.79, 'eta': 8.8600},
    'Au': {'chi': 4.894, 'eta': 5.1720},
    'Hg': {'chi': 6.27, 'eta': 8.3200},
    'Tl': {'chi': 3.2, 'eta': 5.8000},
    'Pb': {'chi': 3.9, 'eta': 7.0600},
    'Bi': {'chi': 4.69, 'eta': 7.4800},
    'Po': {'chi': 4.21, 'eta': 8.4200},
    'At': {'chi': 4.75, 'eta': 9.5000},
    'Rn': {'chi': 5.37, 'eta': 10.7400},
    'Fr': {'chi': 2.0, 'eta': 4.0000},
    'Ra': {'chi': 2.843, 'eta': 4.8680},
    'Ac': {'chi': 2.835, 'eta': 5.6700},
    'Th': {'chi': 3.175, 'eta': 5.8100},
    'Pa': {'chi': 2.985, 'eta': 5.8100},
    'U': {'chi': 3.341, 'eta': 5.7060},
    'Np': {'chi': 3.549, 'eta': 5.4340},
    'Pu': {'chi': 3.243, 'eta': 5.6380},
    'Am': {'chi': 2.9895, 'eta': 6.0070},
    'Cm': {'chi': 2.8315, 'eta': 6.3790},
    'Bk': {'chi': 3.1935, 'eta': 6.0710},
    'Cf': {'chi': 3.197, 'eta': 6.2020},
    'Es': {'chi': 3.333, 'eta': 6.1780},
    'Fm': {'chi': 3.4, 'eta': 6.2000},
    'Md': {'chi': 3.47, 'eta': 6.2200},
    'No': {'chi': 3.475, 'eta': 6.3500},
    'Lr': {'chi': 3.5, 'eta': 6.4000}
}


def get_weight_function(weight_func: str, weight_params: Dict = None) -> Callable:
    """
    Get the weight function for distance-weighted averaging.
    
    Args:
        weight_func: Name of weight function ('inverse', 'gaussian', 'inverse_square')
        weight_params: Parameters for the weight function
    
    Returns:
        Callable that takes distance array and returns weight array
    """
    if weight_params is None:
        weight_params = {}
    
    if weight_func == 'inverse':
        min_distance = weight_params.get('min_distance', 0.1)
        def weight_fn(distances):
            safe_distances = np.maximum(distances, min_distance)
            return 1.0 / safe_distances
        return weight_fn
    
    elif weight_func == 'gaussian':
        sigma = weight_params.get('sigma', 1.0)
        def weight_fn(distances):
            return np.exp(-distances**2 / (2 * sigma**2))
        return weight_fn
    
    elif weight_func == 'inverse_square':
        min_distance = weight_params.get('min_distance', 0.1)
        def weight_fn(distances):
            safe_distances = np.maximum(distances, min_distance)
            return 1.0 / (safe_distances**2)
        return weight_fn
    
    else:
        raise ValueError(
            f"Unknown weight function: {weight_func}. "
            f"Supported: 'inverse', 'gaussian', 'inverse_square'"
        )


def compute_neighbor_list(
    solute_positions: np.ndarray,
    solvent_positions: np.ndarray,
    radius: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Compute neighbor list for solute atoms within radius of solvent atoms.
    
    Args:
        solute_positions: Array of shape (n_solute, 3) with solute positions
        solvent_positions: Array of shape (n_solvent, 3) with solvent positions
        radius: Search radius in Angstroms
    
    Returns:
        Tuple of (neighbor_indices, neighbor_distances) where each element is a list
        containing arrays for each solute atom
    """
    kdtree = KDTree(solvent_positions)
    
    neighbor_indices = []
    neighbor_distances = []
    
    for solute_pos in solute_positions:
        indices = kdtree.query_ball_point(solute_pos, radius)
        
        if len(indices) > 0:
            distances = np.linalg.norm(
                solvent_positions[indices] - solute_pos,
                axis=1
            )
            neighbor_indices.append(np.array(indices))
            neighbor_distances.append(distances)
        else:
            neighbor_indices.append(np.array([], dtype=int))
            neighbor_distances.append(np.array([]))
    
    return neighbor_indices, neighbor_distances


def apply_qeq(
    atoms_obj,
    current_charges: np.ndarray,
    qeq_params: Dict[str, Dict[str, float]],
    total_charge: float,
    tolerance: float = 1e-6
) -> np.ndarray:
    """
    Apply Charge Equilibration (QEq) to equilibrate charges while preserving total charge.
    
    Args:
        atoms_obj: ASE Atoms object
        current_charges: Current charges array (after softening)
        qeq_params: Dictionary mapping element symbols to {'chi': float, 'eta': float}
        total_charge: Target total charge to preserve
        tolerance: Tolerance for charge conservation check
    
    Returns:
        Updated charges array after QEq equilibration
    
    Raises:
        ValueError: If QEq parameters missing for any element or charge not conserved
    """
    n_atoms = len(atoms_obj)
    symbols = atoms_obj.get_chemical_symbols()
    
    for symbol in symbols:
        if symbol not in qeq_params:
            raise ValueError(
                f"QEq parameters not found for element '{symbol}'. "
                f"Please add to qeq_params dictionary."
            )
    
    chi_values = np.array([qeq_params[symbol]['chi'] for symbol in symbols])
    eta_values = np.array([qeq_params[symbol]['eta'] for symbol in symbols])
    
    A = np.zeros((n_atoms + 1, n_atoms + 1))
    b = np.zeros(n_atoms + 1)
    
    for i in range(n_atoms):
        A[i, i] = 2.0 * eta_values[i]
        A[i, n_atoms] = 1.0
        A[n_atoms, i] = 1.0
        b[i] = -chi_values[i] - 2.0 * eta_values[i] * current_charges[i]
    
    A[n_atoms, n_atoms] = 0.0
    b[n_atoms] = total_charge - np.sum(current_charges)
    
    try:
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(A, b, rcond=None)[0]
    
    charge_adjustments = solution[:n_atoms]
    final_charges = current_charges + charge_adjustments
    
    charge_sum = np.sum(final_charges)
    if abs(charge_sum - total_charge) > tolerance:
        raise ValueError(
            f"Charge conservation failed: "
            f"expected {total_charge:.6f}, got {charge_sum:.6f}, "
            f"difference {abs(charge_sum - total_charge):.6e}"
        )
    
    return final_charges


def validate_inputs(
    atoms_obj,
    solute_indices: np.ndarray,
    solute_charges: np.ndarray,
    solvent_charges: np.ndarray
):
    """
    Validate input parameters for ChargeSoftener.
    
    Args:
        atoms_obj: ASE Atoms object
        solute_indices: Array of solute atom indices
        solute_charges: Array of solute charges
        solvent_charges: Array of solvent charges
    
    Raises:
        ValueError: If inputs are invalid
    """
    n_atoms = len(atoms_obj)
    n_solute = len(solute_indices)
    n_solvent = n_atoms - n_solute
    
    if n_solute == 0:
        raise ValueError("solute_indices cannot be empty")
    
    if len(solvent_charges) == 0:
        raise ValueError("solvent_charges cannot be empty")
    
    if len(solute_charges) != n_solute:
        raise ValueError(
            f"Length mismatch: solute_charges ({len(solute_charges)}) "
            f"!= number of solute atoms ({n_solute})"
        )
    
    if len(solvent_charges) != n_solvent:
        raise ValueError(
            f"Length mismatch: solvent_charges ({len(solvent_charges)}) "
            f"!= number of solvent atoms ({n_solvent})"
        )
    
    if np.max(solute_indices) >= n_atoms:
        raise ValueError(
            f"solute_indices contains invalid index: max={np.max(solute_indices)}, "
            f"n_atoms={n_atoms}"
        )
    
    if np.min(solute_indices) < 0:
        raise ValueError(
            f"solute_indices contains negative index: min={np.min(solute_indices)}"
        )
    
    if len(np.unique(solute_indices)) != len(solute_indices):
        raise ValueError("solute_indices contains duplicate indices")
