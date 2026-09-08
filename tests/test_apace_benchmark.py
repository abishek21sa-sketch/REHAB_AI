from rehab_ai.apace.benchmark import run_policy_benchmark

def test_policy_benchmark_is_reproducible_and_safe():
    a=run_policy_benchmark(patients=6,horizon=2,seed=9)
    b=run_policy_benchmark(patients=6,horizon=2,seed=9)
    assert a==b
    rows={r.policy:r for r in a.results}
    assert rows['apace'].unsafe_rate==0
    assert rows['oracle'].mean_regret_to_oracle==0
    assert rows['apace'].mean_regret_to_oracle>=0

def test_apace_uncertainty_not_worse_than_nonexploratory_mpc_reference():
    suite=run_policy_benchmark(patients=8,horizon=2,seed=12)
    rows={r.policy:r for r in suite.results}
    assert rows['apace'].mean_final_uncertainty <= rows['standard_mpc'].mean_final_uncertainty + 1e-9
