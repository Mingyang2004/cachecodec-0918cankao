from script.train.SFT_train import LossConvergenceTracker


def test_loss_convergence_tracker_detects_stable_window():
    tracker = LossConvergenceTracker(window_size=3, patience=2, min_delta=1e-3)
    assert not tracker.update(1.0)
    assert not tracker.update(0.9998)
    assert not tracker.update(0.9999)
    assert tracker.update(1.0001)


def test_loss_convergence_tracker_resets_on_improvement():
    tracker = LossConvergenceTracker(window_size=3, patience=2, min_delta=1e-3)
    for value in (1.0, 1.0001, 1.0002):
        tracker.update(value)
    assert not tracker.update(0.9)
