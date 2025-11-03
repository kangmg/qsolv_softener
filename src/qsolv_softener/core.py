"""
Core charge softening functionality.
"""

import numpy as np
from ase import Atoms
from typing import Dict, Optional, List, Union
from .utils import (
    get_weight_function,
    compute_neighbor_list,
    apply_qeq,
    validate_inputs,
    DEFAULT_QEQ_PARAMS
)


class ChargeSoftener:
    """
    Soften partial charges on solute atoms based on neighboring solvent atoms
    and apply QEq to maintain total charge.
    
    Attributes:
        atoms: ASE Atoms object containing the full system
        solute_indices: Array of indices identifying solute atoms
        solute_charges: Array of initial solute charges
        solvent_charges: Array of initial solvent charges
        radius: Search radius for neighbor detection (Angstroms)
        weight_func: Name of weight function ('inverse', 'gaussian', 'inverse_square')
        weight_params: Parameters for the weight function
        alpha: Mixing parameter (0=full softening, 1=no softening)
        qeq_params: Dictionary of QEq parameters per element
        filtered_solvent_indices: Indices of solvent atoms within radius (set after run)
        filtered_atoms: Atoms object with solute + filtered solvent (set after run)
    """
    
    def __init__(
        self,
        atoms: Atoms,
        solute_indices: Union[List[int], np.ndarray],
        solute_charges: np.ndarray,
        solvent_charges: np.ndarray,
        radius: float = 5.0,
        weight_func: str = 'inverse',
        weight_params: Optional[Dict] = None,
        alpha: float = 0.5,
        qeq_params: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize ChargeSoftener.
        
        Args:
            atoms: ASE Atoms object with solute + solvent system
            solute_indices: Indices of solute atoms in the Atoms object
            solute_charges: Initial charges for solute atoms (1D array)
            solvent_charges: Initial charges for solvent atoms (1D array)
            radius: Search radius for neighbor detection (default: 5.0 Å)
            weight_func: Weight function name (default: 'inverse')
            weight_params: Parameters for weight function (default: None)
            alpha: Mixing parameter between 0 and 1 (default: 0.5)
            qeq_params: QEq parameters dict (default: None, uses DEFAULT_QEQ_PARAMS)
        
        Raises:
            ValueError: If input validation fails
        """
        self.atoms = atoms.copy()
        self.solute_indices = np.array(solute_indices, dtype=int)
        self.solute_charges = np.array(solute_charges, dtype=float)
        self.solvent_charges = np.array(solvent_charges, dtype=float)
        self.radius = float(radius)
        self.weight_func = weight_func
        self.weight_params = weight_params if weight_params is not None else {}
        self.alpha = float(alpha)
        self.qeq_params = qeq_params if qeq_params is not None else DEFAULT_QEQ_PARAMS.copy()
        
        validate_inputs(
            self.atoms,
            self.solute_indices,
            self.solute_charges,
            self.solvent_charges
        )
        
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        
        if self.radius <= 0:
            raise ValueError(f"radius must be positive, got {self.radius}")
        
        n_atoms = len(self.atoms)
        self.solvent_indices = np.array([
            i for i in range(n_atoms) if i not in self.solute_indices
        ])
        
        all_charges = np.zeros(n_atoms)
        all_charges[self.solute_indices] = self.solute_charges
        all_charges[self.solvent_indices] = self.solvent_charges
        self.initial_charges = all_charges
        self.total_charge = np.sum(all_charges)
        
        self.filtered_solvent_indices = None
        self.filtered_atoms = None
    
    def _soften_charges(self) -> np.ndarray:
        """
        Apply charge softening to solute atoms based on solvent neighbors.
        Also collects filtered solvent indices within radius.
        
        Returns:
            Array of softened charges for all atoms (solvent charges unchanged)
        """
        softened_charges = self.initial_charges.copy()
        
        weight_fn = get_weight_function(self.weight_func, self.weight_params)
        
        solute_positions = self.atoms.positions[self.solute_indices]
        solvent_positions = self.atoms.positions[self.solvent_indices]
        
        neighbor_indices, neighbor_distances = compute_neighbor_list(
            solute_positions,
            solvent_positions,
            self.radius
        )
        
        filtered_solvent_set = set()
        
        for i, solute_idx in enumerate(self.solute_indices):
            if len(neighbor_indices[i]) == 0:
                continue
            
            neighbor_solvent_indices = self.solvent_indices[neighbor_indices[i]]
            filtered_solvent_set.update(neighbor_solvent_indices)
            
            neighbor_charges = self.initial_charges[neighbor_solvent_indices]
            distances = neighbor_distances[i]
            
            weights = weight_fn(distances)
            weighted_avg_charge = np.sum(neighbor_charges * weights) / np.sum(weights)
            
            original_charge = self.initial_charges[solute_idx]
            softened_charge = self.alpha * original_charge + (1.0 - self.alpha) * weighted_avg_charge
            softened_charges[solute_idx] = softened_charge
        
        self.filtered_solvent_indices = np.array(sorted(filtered_solvent_set))
        
        return softened_charges
    
    def run(self) -> Atoms:
        """
        Execute the charge softening and QEq equilibration workflow.
        Creates filtered_atoms containing only solute + solvent atoms within radius.
        
        Returns:
            Updated ASE Atoms object with new charges set via set_initial_charges()
        
        Raises:
            ValueError: If charge conservation fails or QEq parameters missing
        """
        softened_charges = self._soften_charges()
        
        final_charges = apply_qeq(
            self.atoms,
            softened_charges,
            self.qeq_params,
            self.total_charge
        )
        
        updated_atoms = self.atoms.copy()
        updated_atoms.set_initial_charges(final_charges)
        
        self._create_filtered_atoms(final_charges)
        
        return updated_atoms
    
    def _create_filtered_atoms(self, charges: np.ndarray):
        """
        Create filtered Atoms object with solute + solvent atoms within radius.
        
        Args:
            charges: Final charges array to apply to filtered atoms
        """
        combined_indices = np.concatenate([self.solute_indices, self.filtered_solvent_indices])
        combined_indices = np.sort(combined_indices)
        
        self.filtered_atoms = self.atoms[combined_indices].copy()
        self.filtered_atoms.set_initial_charges(charges[combined_indices])
    
    def get_softened_charges(self) -> np.ndarray:
        """
        Get charges after softening but before QEq (for analysis/debugging).
        
        Returns:
            Array of charges after softening step only
        """
        return self._soften_charges()
