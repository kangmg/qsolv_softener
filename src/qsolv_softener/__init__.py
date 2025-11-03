"""
qsolv_softener: Charge softening for solute atoms with QEq equilibration.

This package provides tools to soften partial charges on solute atoms based on
neighboring solvent atoms within a specified radius, while maintaining the total
charge of the system using QEq (Charge Equilibration).
"""

from .core import ChargeSoftener
from .utils import DEFAULT_QEQ_PARAMS

__version__ = "0.1.0"
__all__ = ["ChargeSoftener", "DEFAULT_QEQ_PARAMS"]
