from __future__ import annotations
import json
from rehab_ai.data.synthetic import generate_synthetic_session
from rehab_ai.services.policy_studio import run_policy_studio

if __name__ == "__main__":
    result = run_policy_studio(generate_synthetic_session(), horizon_weeks=4)
    print(json.dumps(result.to_dict(), indent=2))
