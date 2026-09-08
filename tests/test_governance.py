from pathlib import Path


def test_required_governance_assets_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        "README.md",
        "Dockerfile",
        "docker-compose.yml",
        ".github/workflows/ci.yml",
        "docs/clinical/CLINICAL_SAFETY.md",
        "docs/architecture/ARCHITECTURE.md",
        "docs/validation/VALIDATION_STRATEGY.md",
    ]
    for rel in required:
        assert (root / rel).exists(), rel
