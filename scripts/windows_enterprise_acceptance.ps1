$ErrorActionPreference = "Stop"
Write-Host "REHAB AI Enterprise Acceptance"
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root
if (-not (Test-Path ".venv\Scripts\python.exe")) { py -3 -m venv .venv }
$python = ".venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -e ".[dev]"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$g1=@("tests/test_ai.py","tests/test_apace_algorithm.py","tests/test_apace_benchmark.py","tests/test_apace_studio.py","tests/test_audit.py","tests/test_biomechanics.py","tests/test_biomechanics_dynamics.py","tests/test_biomechanics_energetics.py","tests/test_data_validation.py","tests/test_feature_engines.py","tests/test_fusion.py","tests/test_gait_event_fusion.py","tests/test_governance.py","tests/test_hdf5_archive.py","tests/test_ingestion.py","tests/test_integrated_adaptive_system.py","tests/test_ml_model.py","tests/test_multiobjective.py","tests/test_optimization.py","tests/test_phenotypes.py","tests/test_pipeline.py")
$g2=@("tests/test_policy_benchmark_api.py","tests/test_policy_studio.py","tests/test_portfolio_validation.py","tests/test_rehab_math_identity.py","tests/test_safe_envelope_ml.py","tests/test_safety_certificate.py","tests/test_safety_certificate_api.py","tests/test_scheduling.py","tests/test_scheduling_oracle.py","tests/test_signal_quality.py","tests/test_simulation.py","tests/test_starlette_web.py","tests/test_stochastic_mpc.py","tests/test_temporal_recovery_ml.py","tests/test_trajectory_ai.py","tests/test_treatment_response_ml.py","tests/test_twin.py","tests/test_twin_forecast.py","tests/test_v095_platform.py","tests/test_v1_full_workflow.py","tests/test_v1_motion_acceptance.py")
& $python -m pytest -q @g1
if ($LASTEXITCODE -ne 0) { throw "REHAB regression group 1 failed" }
& $python -m pytest -q @g2
if ($LASTEXITCODE -ne 0) { throw "REHAB regression group 2 failed" }
& $python scripts/portfolio_validation.py
if ($LASTEXITCODE -ne 0) { throw "REHAB portfolio validation failed" }
& $python scripts/enterprise_operability.py
if ($LASTEXITCODE -ne 0) { throw "REHAB enterprise operability failed" }
Write-Host "REHAB_AI_ENTERPRISE_ACCEPTANCE=PASS"
