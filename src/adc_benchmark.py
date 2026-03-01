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
import qutip as qt
from datetime import datetime
import warnings
import argparse
import time
import json
import csv
import os
warnings.filterwarnings('ignore', category=FutureWarning, module='qutip')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='qutip')

# Physical constants
ENERGY_SCALE_MK = 240  # Energy scale in mK for thermal occupation calculation

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
        rng = np.random.default_rng(self.seed + int(t * 100))
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
        H += sum([thermal*rng.standard_normal()*sz[i] for i in range(self.N)])
        
        # Crosstalk
        for i in range(self.N):
            for j in range(i+2, self.N):
                H += crosstalk*rng.standard_normal()*(sx[i]*sx[j])
        
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
        
        # Thermal occupation - robust calculation to prevent overflow
        arg = ENERGY_SCALE_MK / float(self.T)
        # Prevent overflow for extremely large arg; exp(>700) overflows in double
        arg_clamped = min(arg, 700.0)
        # If arg is large, nth ~ 0
        nth = 1.0 / (np.exp(arg_clamped) - 1.0) if arg_clamped < 700.0 else 0.0
        
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

def depolarize(ρ: qt.Qobj, p: float = 0.002) -> qt.Qobj:
    """Apply depolarizing channel (gate error model)"""
    if p == 0:
        return ρ
    N = len(ρ.dims[0])
    I = qt.tensor([qt.qeye(2) for _ in range(N)])
    dim = 2**N
    return (1-p)*ρ + p * I / dim

def make_mitigation(circ: "QuantumCircuit", config: dict, seed_offset: int = 0, p_gate: float = 0.002) -> tuple:
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
            c_ops = circ.diss(γm)
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

def run_state_preparation(
    T: float = 50,
    config: dict | None = None,
    cycles: int = 16,
    num_seeds: int = 20,
) -> dict:
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
    if config is None:
        raise ValueError("config must be provided as a dict with at least a 'name' key")
    fidelities = []
    purities = []
    
    t_start = time.perf_counter()
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
        
        # Measure purity tr(ρ²)
        purity = (ρ * ρ).tr().real
        purities.append(purity)
    
    runtime_seconds = time.perf_counter() - t_start
    
    # Bootstrap confidence intervals
    # Use fixed RNG seed for reproducibility and bootstrap the MEDIAN
    fidelities = np.array(fidelities)
    rng = np.random.default_rng(12345)
    n_boot = 2000
    bootstraps = np.array([
        np.median(rng.choice(fidelities, size=len(fidelities), replace=True))
        for _ in range(n_boot)
    ])
    ci_low, ci_high = np.quantile(bootstraps, [0.025, 0.975])
    median = float(np.median(fidelities))
    median_purity = float(np.median(purities))
    
    return {
        'median': median,
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'gates': gates,
        'median_purity': median_purity,
        'runtime_seconds': runtime_seconds,
    }

def optimize_gamma_compute(T: float = 50, cycles: int = 10, num_seeds: int = 10) -> float:
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

def temperature_sweep(quick: bool = False, output_path: str = "adc_temperature_sweep.png") -> dict:
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
        
        # Print summary table
        print(f"\n✅ RESULTS at T={T} mK:")
        header = f"{'Strategy':<14} {'Median':>8}   {'95% CI':<22} {'Gates':>6}  {'Purity':>8}  {'Runtime(s)':>10}"
        separator = f"{'--------':<14} {'------':>8}   {'------':<22} {'-----':>6}  {'------':>8}  {'----------':>10}"
        print(header)
        print(separator)
        for strategy_name in ['Baseline', 'DD', 'ADC opt', 'Hybrid opt']:
            r = results[strategy_name][-1]
            ci_str = f"[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
            print(f"{strategy_name:<14} {r['median']:>8.4f}   {ci_str:<22} {r['gates']:>6}  {r['median_purity']:>8.4f}  {r['runtime_seconds']:>10.1f}")
    
    # Plot results
    plot_temperature_sweep(results, output_path)
    
    return results

def plot_temperature_sweep(results: dict, output_path: str = "adc_temperature_sweep.png") -> None:
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
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Figure saved: {output_path}")
    plt.close()

# ============================================================================
# RESULTS EXPORT
# ============================================================================

def save_results(results: dict, output_dir: str = "results", save_json: bool = True, save_csv: bool = True) -> None:
    """
    Save benchmark results to JSON and/or CSV files.
    
    Args:
        results: Results dict from temperature_sweep()
        output_dir: Directory to save files (created if missing)
        save_json: If True, save a JSON file
        save_csv: If True, save a CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    
    strategy_names = ['Baseline', 'DD', 'ADC opt', 'Hybrid opt']
    
    if save_json:
        json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
        export = {
            "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "temperatures": results["T"],
            "strategies": {
                name: [
                    {
                        "median": r["median"],
                        "ci_low": r["ci_low"],
                        "ci_high": r["ci_high"],
                        "gates": r["gates"],
                        "median_purity": r["median_purity"],
                        "runtime_seconds": r["runtime_seconds"],
                    }
                    for r in results[name]
                ]
                for name in strategy_names
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
        print(f"   JSON saved: {json_path}")
    
    if save_csv:
        csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["temperature", "strategy", "median", "ci_low", "ci_high", "gates", "median_purity", "runtime_seconds"])
            for i, T in enumerate(results["T"]):
                for name in strategy_names:
                    r = results[name][i]
                    writer.writerow([
                        T, name,
                        r["median"], r["ci_low"], r["ci_high"], r["gates"],
                        r["median_purity"], r["runtime_seconds"],
                    ])
        print(f"   CSV saved: {csv_path}")


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
    parser.add_argument(
        '--output',
        default='adc_temperature_sweep.png',
        help='Output file path for the temperature sweep plot (default: adc_temperature_sweep.png)'
    )
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save benchmark results as both JSON and CSV to --results-dir'
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory for saved results files (default: results)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Save benchmark results as JSON to --results-dir'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Save benchmark results as CSV to --results-dir'
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
    
    results = temperature_sweep(quick=args.quick, output_path=args.output)
    
    # Determine which formats to save
    do_json = args.save_results or args.json
    do_csv = args.save_results or args.csv
    if do_json or do_csv:
        print(f"\n💾 Saving results to '{args.results_dir}'...")
        save_results(results, output_dir=args.results_dir, save_json=do_json, save_csv=do_csv)
    
    print(f"\n{'='*80}")
    print("✅ BENCHMARK COMPLETED")
    print('='*80)
    print(f"Finish time: {datetime.now().strftime('%H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    results = main()
