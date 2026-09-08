from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.services.pipeline import RehabDecisionPipeline


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        pipeline = RehabDecisionPipeline(database_path=str(Path(tmp) / "demo.db"))
        result = pipeline.run(generate_synthetic_session())
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
