from rehab_ai.scheduling.therapy_schedule import TherapyDemand, optimize_weekly_schedule


def test_scheduler_respects_capacity_and_priority():
    demands = [
        TherapyDemand(patient_id="HIGH", priority=.95, sessions_required=3),
        TherapyDemand(patient_id="LOW", priority=.35, sessions_required=3),
    ]
    result = optimize_weekly_schedule(demands, therapists=1, slots_per_day=1, days=5)
    assert len(result.sessions) == 5
    assert sum(1 for s in result.sessions if s.patient_id == "HIGH") == 3
    assert result.unscheduled["LOW"] == 1
    assert result.utilization == 1.0
