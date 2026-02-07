import jax
import jax.numpy as jnp
from .loader import load_mechanism
from .thermo import compute_mixture_props
from .kinetics import compute_wdot
from .constants import ONE_ATM

class Solution:
    """A user-friendly wrapper for chemical state and kinetics.
    
    This class is intended for ease of use and state management. 
    Numerical performance is achieved by calling the pure functions 
    directly with MechData.
    """
    
    def __init__(self, yaml_file: str):
        self.mech = load_mechanism(yaml_file)
        self.n_species = self.mech.n_species
        self.n_reactions = self.mech.n_reactions
        self.species_names = self.mech.species_names
        
        # Default state: 300K, 1 atm, pure N2 if available
        self._T = 300.0
        self._P = ONE_ATM
        self._Y = jnp.zeros(self.n_species)
        if 'N2' in self.species_names:
            self._Y = self._Y.at[self.species_names.index('N2')].set(1.0)
        else:
            self._Y = self._Y.at[0].set(1.0)

    @property
    def T(self): return self._T
    @T.setter
    def T(self, value): self._T = float(value)
    
    @property
    def P(self): return self._P
    @P.setter
    def P(self, value): self._P = float(value)
    
    def _parse_composition(self, value):
        if isinstance(value, str):
            res = jnp.zeros(self.n_species)
            parts = [p.strip() for p in value.split(',')]
            for p in parts:
                spec, val = p.split(':')
                if spec.strip() in self.species_names:
                    res = res.at[self.species_names.index(spec.strip())].set(float(val))
                else:
                    raise ValueError(f"Unknown species: {spec.strip()}")
            return res
        return jnp.array(value)

    @property
    def Y(self): return self._Y
    @Y.setter
    def Y(self, value):
        Y = self._parse_composition(value)
        self._Y = Y / jnp.sum(Y)
        
    @property
    def X(self):
        y_mw = self._Y / self.mech.mol_weights
        return y_mw / jnp.sum(y_mw)
    @X.setter
    def X(self, value):
        X = self._parse_composition(value)
        X = X / jnp.sum(X)
        y_unnorm = X * self.mech.mol_weights
        self._Y = y_unnorm / jnp.sum(y_unnorm)

    @property
    def TP(self): return self.T, self.P
    @TP.setter
    def TP(self, value):
        self.T, self.P = value

    @property
    def TPY(self): return self.T, self.P, self.Y
    @TPY.setter
    def TPY(self, value):
        self.T, self.P, self.Y = value

    @property
    def TPX(self): return self.T, self.P, self.X
    @TPX.setter
    def TPX(self, value):
        self.T, self.P, self.X = value

    def set_TPY(self, T, P, Y):
        self.TPY = T, P, Y
        
    def set_TPX(self, T, P, X):
        self.TPX = T, P, X

    # Thermodynamics
    @property
    def cp_mass(self):
        cp, _, _ = compute_mixture_props(self.T, self.P, self.Y, self.mech)
        return float(cp)
        
    @property
    def enthalpy_mass(self):
        _, h, _ = compute_mixture_props(self.T, self.P, self.Y, self.mech)
        return float(h)
        
    @property
    def density(self):
        _, _, rho = compute_mixture_props(self.T, self.P, self.Y, self.mech)
        return float(rho)

    # Kinetics
    def net_production_rates(self):
        wdot, _, _, _ = compute_wdot(self.T, self.P, self.Y, self.mech)
        return np.array(wdot)
        
    def equilibrate(self, mode='TP'):
        from .equilibrate import equilibrate as eq_func
        return eq_func(self, mode=mode)

import numpy as np # for array conversion in methods if needed
