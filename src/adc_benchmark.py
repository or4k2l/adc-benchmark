#!/usr/bin/env python3
"""
Adaptive Dissipation Control: Comprehensive Benchmark Suite
============================================================
Automated comparison of error mitigation strategies for superconducting qubits.

Strategies tested:
- Baseline (fixed low γ)
- Dynamical Decoupling (XY8)
- Adaptive Dissipation Control (ADC)
- Hybrid (ADC + DD)

Features:
- Temperature-dependent optimization
- Statistical validation with bootstrap CI
- Automatic parameter tuning
- Zero gate overhead for ADC

Install: pip install -r requirements.txt
Runtime: ~25 minutes for full benchmark (depends on machine & qutip backend)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import qutip as qt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADAPTIVE DISSIPATION CONTROL - COMPREHENSIVE BENCHMARK")
print("="*80)
print(f"Start: {datetime.now().strftime('%H:%M:%S')}\n")

# ============================================================================
# HARDWARE MODEL
# ============================================================================

class QuantumCircuit:
    """Superconducting qubit circuit with realistic noise"""
    
    def __init__(self, N=4, T=50, seed=42):
        self.N = N
        self.T = T
        self.seed = seed
        self.ops = self._ops()
        
        # Noise parameters (temperature-scaled)
        self.thermal_base = 0.01 * (T / 50)
        self.crosstalk_base = 0.02
        self.flux = 0.005
        self.rabi = 0.5
        self.gamma_phi_base = 0.005 * (T / 50)
    
    def _ops(self):
        """Build Pauli operator basis"""
        N = self.N
        sx = [qt.tensor([qt.sigmax() if i==j else qt.qeye(2) for j in range(N)]) for i in range(N)]
        sy = [qt.tensor([qt.sigmay() if i==j else qt.qeye(2) for j in range(N)]) for i in range(N)]
        sz = [qt.tensor([qt.sigmaz() if i==j else qt.qeye(2) for j in range(N)]) for i in range(N)]
        sm = [(sx[i]-1j*sy[i])/2 for i in range(N)]
        sp = [(sx[i]+1j*sy[i])/2 for i in range(N)]
        return {'sx':sx, 'sy':sy, 'sz':sz, 'sm':sm, 'sp':sp}
    
    def H(self, t=0, drive=False):
        """
        Build Hamiltonian with realistic noise
        
        Args:
            t: Time point (for time-dependent noise)
            drive: Enable Rabi drive during gate operations
        """
        np.random.seed(self.seed + int(t*100))
        sx, sz = self.ops['sx'], self.ops['sz']
        
        # Ideal Hamiltonian
        H = sum([d*sz[i]/2 for i,d in enumerate([0.05,-0.03,0.02,-0.04][:self.N])])
        H += sum([sx[i]*sx[i+1] for i in range(self.N-1)])
        
        # Rabi drive (active during compute phase)
        if drive:
            H += self.rabi*sum(sx)
        
        # Noise amplification during gates
        thermal = self.thermal_base if not drive else self.thermal_base * 8.0
        crosstalk = self.crosstalk_base if not drive else self.crosstalk_base * 5.0
        
        # Thermal noise
        H += sum([thermal*np.random.randn()*sz[i] for i in range(self.N)])
        
        # Crosstalk
        for i in range(self.N):
            for j in range(i+2, self.N):
                H += crosstalk*np.random.randn()*(sx[i]*sx[j])
        
        # 1/f flux noise
        w = 0.1+t*0.01
        H += sum([self.flux*np.sin(w*(i+1))/np.sqrt(w+0.1)*sz[i] for i in range(self.N)])
        
        return H
    
    def diss(self, γ_relax, gamma_phi=None):
        """
        Create Lindblad dissipators
        
        Args:
            γ_relax: T1 relaxation rate
            gamma_phi: T2 dephasing rate (optional)
        """
        if gamma_phi is None:
            gamma_phi = self.gamma_phi_base
        
        sm, sp, sz = self.ops['sm'], self.ops['sp'], self.ops['sz']
        
        # Thermal occupation
        nth = 1/(np.exp(240/self.T)-1) if 240/self.T<20 else 0
        
        # T1 relaxation
        c = [np.sqrt(γ_relax*(1+nth))*sm[i] for i in range(self.N)]
        if nth > 1e-6:
            c += [np.sqrt(γ_relax*nth)*sp[i] for i in range(self.N)]
        
        # T2 dephasing
        c += [np.sqrt(gamma_phi / 2)*sz[i] for i in range(self.N)]
        
        return c
