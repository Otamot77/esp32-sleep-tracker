from sleep_demo.demo import run_demo


def test_demo_runs_end_to_end() -> None:
    result = run_demo(verbose=False)

    assert result.decoded_ppg_ok
    assert 55.0 <= result.heart_rate_bpm <= 65.0
    assert result.rmssd_ms > 0.0
    assert abs(result.respiratory_rate_bpm - 15.0) <= 1.0
    assert 0.4 <= result.ratio_of_ratios <= 1.2
    assert result.total_sleep_minutes > 300.0
    assert result.deep_minutes > 0.0
    assert result.rem_minutes > 0.0
    assert 0.0 <= result.recovery_score <= 100.0
