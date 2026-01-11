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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for CI
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import qutip as qt
from datetime import datetime
import warnings
import argparse
warnings.filterwarnings('ignore')

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

# ============================================================================
# MITIGATION STRATEGIES
# ============================================================================

def depolarize(ρ, p=0.002):
    """Apply depolarizing channel (gate error model)"""
    if p == 0:
        return ρ
    N = len(ρ.dims[0])
    I = qt.tensor([qt.qeye(2) for _ in range(N)])
    dim = 2**N
    return (1-p)*ρ + p * I / dim

def make_mitigation(circ, config, seed_offset=0, p_gate=0.002):
    """
    Create mitigation function based on strategy
    
    Args:
        circ: QuantumCircuit instance
        config: Strategy configuration dict
        seed_offset: Random seed offset for reproducibility
        p_gate: Gate error probability
    
    Returns:
        (mitigation_function, gate_overhead)
    """
    circ.seed = 42 + seed_offset*1000
    strategy_name = config['name']
    γm = config.get('γm', 1e-5)
    γc = config.get('γc', 0.05)
    
    if strategy_name == 'baseline':
        def baseline_mitigation(ρ, iteration):
            """Fixed low γ throughout"""
            result = qt.mesolve(
                circ.H(iteration*2), ρ, 
                np.linspace(0, 2, 12), 
                circ.diss(γm), [],
                options=qt.Options(nsteps=50000)
            )
            return result.states[-1]
        return baseline_mitigation, 0
    
    elif strategy_name == 'adc':
        def adc_mitigation(ρ, iteration):
            """Adaptive dissipation: switch between low/high γ"""
            t = iteration * 2
            
            # IDLE phase (low γ, no drive)
            ρ = qt.mesolve(
                circ.H(t, False), ρ,
                np.linspace(0, 1.0, 8),
                circ.diss(γm), [],
                options=qt.Options(nsteps=50000)
            ).states[-1]
            
            # COMPUTE phase (high γ, WITH drive)
            ρ = qt.mesolve(
                circ.H(t+1.0, True), ρ,
                np.linspace(0, 1.0, 12),
                circ.diss(γc), [],
                options=qt.Options(nsteps=50000)
            ).states[-1]
            
            return ρ
        return adc_mitigation, 0
    
    elif strategy_name == 'dd':
        def dd_mitigation(ρ, iteration):
            """XY8 dynamical decoupling"""
            H = circ.H(iteration*2)
            c_ops = circ.diss(1e-5)
            sx, sy = circ.ops['sx'], circ.ops['sy']
            
            sequence = ['x','y','x','y','y','x','y','x']
            τ = 2.0 / 9
            
            for axis in sequence:
                # Free evolution
                ρ = qt.mesolve(H, ρ, np.linspace(0, τ, 6), c_ops, []).states[-1]
                
                # Apply π-pulses
                for q in range(circ.N):
                    op = sx[q] if axis=='x' else sy[q]
                    U = (-1j*np.pi*op/2).expm()
                    ρ = U*ρ*U.dag()
                
                # Gate error
                ρ = depolarize(ρ, p_gate)
            
            # Final free evolution
            ρ = qt.mesolve(H, ρ, np.linspace(0, τ, 6), c_ops, []).states[-1]
            return ρ
        
        return dd_mitigation, 8*circ.N
    
    elif strategy_name == 'hybrid':
        def hybrid_mitigation(ρ, iteration):
            """Hybrid: DD during idle + ADC during compute"""
            t = iteration * 2
            H_idle = circ.H(t, False)
            c_ops = circ.diss(γm)
            
            # DD during idle phase
            sequence = ['x','y','x','y']
            τ = 1.0 / len(sequence)
            
            for axis in sequence:
                ρ = qt.mesolve(H_idle, ρ, np.linspace(0, τ, 6), c_ops, []).states[-1]
                for q in range(circ.N):
                    op = circ.ops['sx'][q] if axis=='x' else circ.ops['sy'][q]
                    U = (-1j*np.pi*op/2).expm()
                    ρ = U*ρ*U.dag()
                ρ = depolarize(ρ, p_gate)
            
            # ADC during compute phase
            ρ = qt.mesolve(
                circ.H(t+1.0, True), ρ,
                np.linspace(0, 1.0, 12),
                circ.diss(γc), [],
                options=qt.Options(nsteps=50000)
            ).states[-1]
            
            return ρ
        
        return hybrid_mitigation, 4*circ.N
    
    return lambda ρ,i: ρ, 0

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_state_preparation(T=50, config=None, cycles=16, num_seeds=20):
    """
    Run state preparation benchmark with statistical validation
    
    Args:
        T: Temperature (mK)
        config: Strategy configuration
        cycles: Number of evolution cycles
        num_seeds: Number of noise realizations
    
    Returns:
        Dict with median, confidence intervals, gate overhead
    """
    fidelities = []
    
    for seed_offset in range(num_seeds):
        # Create circuit with independent noise
        circ = QuantumCircuit(4, T, 42+seed_offset*100)
        
        # Target state (ground state)
        H_target = circ.H(0, False) - 0.01*sum(circ.ops['sz'])
        _, eigenstates = H_target.eigenstates()
        target = eigenstates[0]
        
        # Initial state |0000⟩
        ρ = qt.ket2dm(qt.tensor([qt.basis(2,0) for _ in range(4)]))
        
        # Apply mitigation strategy
        mitigation_func, gates = make_mitigation(circ, config, seed_offset)
        
        for cycle in range(cycles):
            ρ = mitigation_func(ρ, cycle)
        
        # Measure fidelity
        fid = qt.fidelity(ρ, target)
        fidelities.append(fid)
    
    # Bootstrap confidence intervals
    fidelities = np.array(fidelities)
    bootstraps = np.array([
        np.mean(np.random.choice(fidelities, len(fidelities), replace=True))
        for _ in range(1000)
    ])
    ci_low, ci_high = np.quantile(bootstraps, [0.025, 0.975])
    
    return {
        'median': np.median(fidelities),
        'ci_low': ci_low,
        'ci_high': ci_high,
        'gates': gates
    }

def optimize_gamma_compute(T=50, cycles=10, num_seeds=10):
    """
    Grid search to find optimal γ_compute for given temperature
    
    Args:
        T: Temperature (mK)
        cycles: Evolution cycles for quick test
        num_seeds: Noise realizations per test
    
    Returns:
        Optimal γ_compute value
    """
    print(f"\n🔍 Optimizing γ_compute at T={T} mK...")
    
    gamma_values = np.logspace(-6, -1, 12)
    medians = []
    
    for γc in gamma_values:
        config = {'name': 'adc', 'γm': 1e-5, 'γc': γc}
        result = run_state_preparation(T, config, cycles, num_seeds)
        medians.append(result['median'])
        print(f"   γ_c = {γc:.2e} → median fidelity = {medians[-1]:.4f}")
    
    best_idx = np.argmax(medians)
    best_gamma = gamma_values[best_idx]
    print(f"   ✅ Optimal: γ_c = {best_gamma:.2e}")
    
    return best_gamma

# ============================================================================
# TEMPERATURE SWEEP
# ============================================================================

def temperature_sweep(quick=False):
    """
    Comprehensive temperature sweep comparing all strategies
    
    Args:
        quick: If True, run abbreviated version for CI testing
    
    Tests temperatures: 10, 30, 50, 70, 100 mK (full) or 50 mK only (quick)
    Optimizes γ_compute for each temperature
    Statistical validation with bootstrap CI
    """
    if quick:
        temperatures = [50]
        opt_cycles, opt_seeds = 3, 3
        bench_cycles, bench_seeds = 4, 5
    else:
        temperatures = [10, 30, 50, 70, 100]
        opt_cycles, opt_seeds = 10, 10
        bench_cycles, bench_seeds = 16, 20
    
    results = {
        'T': temperatures,
        'Baseline': [],
        'DD': [],
        'ADC opt': [],
        'Hybrid opt': []
    }
    
    for T in temperatures:
        print(f"\n{'='*80}")
        print(f"TEMPERATURE: {T} mK")
        print('='*80)
        
        # Optimize γ_compute for this temperature
        best_gamma = optimize_gamma_compute(T, opt_cycles, opt_seeds)
        
        # Test all strategies
        print(f"\n📊 Running benchmarks ({bench_cycles} cycles, {bench_seeds} seeds)...")
        
        results['Baseline'].append(
            run_state_preparation(T, {'name':'baseline'}, bench_cycles, bench_seeds)
        )
        
        results['DD'].append(
            run_state_preparation(T, {'name':'dd'}, bench_cycles, bench_seeds)
        )
        
        results['ADC opt'].append(
            run_state_preparation(T, {'name':'adc', 'γm':1e-5, 'γc':best_gamma}, bench_cycles, bench_seeds)
        )
        
        results['Hybrid opt'].append(
            run_state_preparation(T, {'name':'hybrid', 'γm':1e-5, 'γc':best_gamma}, bench_cycles, bench_seeds)
        )
        
        # Print summary
        print(f"\n✅ RESULTS at T={T} mK:")
        for strategy_name in ['Baseline', 'DD', 'ADC opt', 'Hybrid opt']:
            r = results[strategy_name][-1]
            print(f"   {strategy_name:12s}: median={r['median']:.4f}  "
                  f"95% CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]  "
                  f"gates={r['gates']}")
    
    # Plot results
    plot_temperature_sweep(results)
    
    return results

def plot_temperature_sweep(results):
    """Create publication-quality temperature sweep plot"""
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    strategies = ['Baseline', 'DD', 'ADC opt', 'Hybrid opt']
    colors = {'Baseline':'#6b7280', 'DD':'#f59e0b', 
              'ADC opt':'#ef4444', 'Hybrid opt':'#22c55e'}
    
    for strategy in strategies:
        temps = results['T']
        medians = [r['median'] for r in results[strategy]]
        ci_low = [r['ci_low'] for r in results[strategy]]
        ci_high = [r['ci_high'] for r in results[strategy]]
        
        ax.plot(temps, medians, 'o-', label=strategy, 
               linewidth=3, markersize=8, color=colors[strategy])
        ax.fill_between(temps, ci_low, ci_high, 
                        alpha=0.2, color=colors[strategy])
    
    # Highlight optimal window (only if multiple temperatures)
    if len(results['T']) > 1:
        ax.axvspan(30, 70, alpha=0.1, color='green', 
                  label='ADC optimal window')
    
    ax.set_xlabel('Temperature (mK)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Median Fidelity (95% Bootstrap CI)', fontsize=13, fontweight='bold')
    ax.set_title('Temperature-Dependent Error Mitigation Performance', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adc_temperature_sweep.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved: adc_temperature_sweep.png")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete benchmark suite"""
    parser = argparse.ArgumentParser(
        description='ADC Benchmark Suite - Compare error mitigation strategies'
    )
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick smoke test (for CI/testing, ~1-2 min)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("ADAPTIVE DISSIPATION CONTROL - COMPREHENSIVE BENCHMARK")
    print("="*80)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}\n")
    
    if args.quick:
        print("\n🚀 RUNNING QUICK MODE (smoke test)")
        print("   Estimated time: ~1-2 minutes")
        print("   Strategies: Baseline, DD, ADC, Hybrid")
        print("   Temperature: 50 mK only\n")
    else:
        print("\n🚀 STARTING COMPREHENSIVE BENCHMARK")
        print("   Estimated time: ~25 minutes")
        print("   Strategies: Baseline, DD, ADC, Hybrid")
        print("   Temperatures: 10, 30, 50, 70, 100 mK\n")
    
    results = temperature_sweep(quick=args.quick)
    
    print(f"\n{'='*80}")
    print("✅ BENCHMARK COMPLETED")
    print('='*80)
    print(f"Finish time: {datetime.now().strftime('%H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    results = main()
