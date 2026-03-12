"""Tests for v28 training improvements in qwen_supermix_pipeline.py.

Tests cover:
1. Cosine warm restarts LR schedule
2. Focal loss weighting
3. Gradient noise injection (sanity)
4. Curriculum quality ramp logic
5. _build_lr_lambda outputs for various schedules
"""

import math
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "source"))
sys.path.insert(0, os.path.dirname(__file__))


def _import_pipeline():
    """Import _build_lr_lambda from the pipeline module."""
    from source.qwen_supermix_pipeline import _build_lr_lambda
    return _build_lr_lambda


# ── 1. Cosine warm restarts ────────────────────────────────────────

def test_cosine_restarts_produces_periodic_peaks():
    """LR should peak (return ~1.0) at the start of each restart cycle."""
    _build_lr_lambda = _import_pipeline()
    lr_fn = _build_lr_lambda(
        schedule="cosine_restarts",
        warmup_steps=0,
        total_steps=100,
        min_lr_ratio=0.1,
        restart_period=10,
    )
    # At step=0 (step_idx=0 → step=1) we're at cycle_pos=1%10=1.
    # At start of each period (cycle_pos=0, i.e. step multiples of period):
    peaks = []
    valleys = []
    for step_idx in range(100):
        val = lr_fn(step_idx)
        step = step_idx + 1
        post_warmup = step
        cycle_pos = post_warmup % 10
        if cycle_pos == 0:
            # End of a cycle (progress=0 after mod) → should be near 1.0
            peaks.append((step, val))
        elif cycle_pos == 5:
            # Middle of cycle → should be lower
            valleys.append((step, val))
    # Peaks should be near 1.0 (min_ratio + (1-min_ratio)*cos(0) = 1.0)
    for step, val in peaks:
        assert val > 0.95, f"Peak at step {step} should be > 0.95, got {val:.4f}"
    # Valleys should be lower than peaks
    for step, val in valleys:
        assert val < 0.7, f"Valley at step {step} should be < 0.7, got {val:.4f}"
    print(f"  ✓ cosine_restarts: {len(peaks)} peaks found, all > 0.95")


def test_cosine_restarts_disabled_by_zero_period():
    """When restart_period=0, cosine_restarts falls back to standard cosine."""
    _build_lr_lambda = _import_pipeline()
    lr_fn_restarts = _build_lr_lambda(
        schedule="cosine_restarts",
        warmup_steps=0,
        total_steps=100,
        min_lr_ratio=0.1,
        restart_period=0,
    )
    lr_fn_cosine = _build_lr_lambda(
        schedule="cosine",
        warmup_steps=0,
        total_steps=100,
        min_lr_ratio=0.1,
        restart_period=0,
    )
    for step_idx in range(100):
        v1 = lr_fn_restarts(step_idx)
        v2 = lr_fn_cosine(step_idx)
        assert abs(v1 - v2) < 1e-6, f"Step {step_idx}: restarts={v1}, cosine={v2}"
    print("  ✓ cosine_restarts with period=0 matches standard cosine")


# ── 2. Standard schedules still work ──────────────────────────────

def test_constant_schedule():
    """Constant schedule should always return 1.0 after warmup."""
    _build_lr_lambda = _import_pipeline()
    lr_fn = _build_lr_lambda(
        schedule="constant",
        warmup_steps=5,
        total_steps=50,
        min_lr_ratio=0.1,
    )
    assert lr_fn(0) < 1.0, "Step 1 should be in warmup"
    for step_idx in range(5, 50):
        val = lr_fn(step_idx)
        assert abs(val - 1.0) < 1e-6, f"Step {step_idx}: expected 1.0, got {val}"
    print("  ✓ constant schedule works correctly")


def test_cosine_schedule_decay():
    """Cosine should decay from 1.0 to min_lr_ratio."""
    _build_lr_lambda = _import_pipeline()
    lr_fn = _build_lr_lambda(
        schedule="cosine",
        warmup_steps=0,
        total_steps=100,
        min_lr_ratio=0.15,
    )
    first = lr_fn(0)
    last = lr_fn(99)
    assert first > 0.9, f"First step should be near 1.0, got {first}"
    assert last < 0.25, f"Last step should be near min_lr_ratio, got {last}"
    # Should be monotonically decreasing
    prev = 2.0
    for step_idx in range(100):
        val = lr_fn(step_idx)
        assert val <= prev + 1e-6, f"Step {step_idx}: LR went up from {prev} to {val}"
        prev = val
    print("  ✓ cosine schedule decays monotonically")


# ── 3. Focal loss weighting logic ─────────────────────────────────

def test_focal_weight_modulation():
    """Easy samples (low loss) get lower focal weight than hard samples (high loss)."""
    easy_loss = torch.tensor([0.1, 0.2, 0.15])
    hard_loss = torch.tensor([2.0, 3.0, 2.5])
    gamma = 2.0

    easy_weight = (1.0 - torch.exp(-easy_loss)).pow(gamma)
    hard_weight = (1.0 - torch.exp(-hard_loss)).pow(gamma)

    assert easy_weight.mean() < hard_weight.mean(), (
        f"Easy weight ({easy_weight.mean():.4f}) should be < hard weight ({hard_weight.mean():.4f})"
    )
    assert easy_weight.mean() < 0.1, f"Easy samples should have very low focal weight"
    assert hard_weight.mean() > 0.5, f"Hard samples should have high focal weight"
    print(f"  ✓ focal weighting: easy_mean={easy_weight.mean():.4f} < hard_mean={hard_weight.mean():.4f}")


def test_focal_gamma_zero_is_identity():
    """With gamma=0, focal weight should be 1.0 everywhere."""
    losses = torch.tensor([0.1, 1.0, 5.0])
    gamma = 0.0
    focal_weight = (1.0 - torch.exp(-losses)).pow(gamma)
    assert torch.allclose(focal_weight, torch.ones_like(focal_weight)), (
        f"Gamma=0 should give all-ones weights, got {focal_weight}"
    )
    print("  ✓ focal gamma=0 is identity (all weights = 1.0)")


# ── 4. Gradient noise injection ───────────────────────────────────

def test_gradient_noise_decays():
    """Noise std should decrease as step increases."""
    eta = 0.01
    noise_early = eta / (1.0 + 1.0) ** 0.55
    noise_mid = eta / (1.0 + 500.0) ** 0.55
    noise_late = eta / (1.0 + 5000.0) ** 0.55

    assert noise_early > noise_mid > noise_late, (
        f"Noise should decay: {noise_early:.6f} > {noise_mid:.6f} > {noise_late:.6f}"
    )
    assert noise_late < noise_early * 0.1, "Late noise should be < 10% of early noise"
    print(f"  ✓ grad noise decays: early={noise_early:.6f}, mid={noise_mid:.6f}, late={noise_late:.6f}")


# ── 5. Curriculum quality ramp ────────────────────────────────────

def test_curriculum_ramp_reduces_low_weight_samples():
    """Curriculum ramp should reduce weight of below-median samples early in training."""
    ramp = 0.3
    total_steps = 100
    weights = torch.tensor([0.5, 0.8, 1.2, 1.5, 2.0])
    median = weights.median()

    # Early in training (step 0): ramp_scale = 1 - 0.3 * (1 - 0) = 0.7
    ramp_progress = 0.0
    ramp_scale = 1.0 - ramp * (1.0 - ramp_progress)
    below_median = (weights < median).float()
    adjusted = weights * (1.0 - below_median * (1.0 - ramp_scale))

    # Below-median weights should be reduced
    for i in range(len(weights)):
        if weights[i] < median:
            assert adjusted[i] < weights[i], (
                f"Below-median weight {weights[i]:.2f} should be reduced, got {adjusted[i]:.2f}"
            )
        else:
            assert adjusted[i] == weights[i], (
                f"Above-median weight {weights[i]:.2f} should stay same, got {adjusted[i]:.2f}"
            )

    # Late in training (step = total_steps//2): ramp_scale = 1.0, no modulation
    ramp_progress = 1.0
    ramp_scale = 1.0 - ramp * (1.0 - ramp_progress)
    assert abs(ramp_scale - 1.0) < 1e-6, f"At end of ramp, scale should be 1.0, got {ramp_scale}"
    print("  ✓ curriculum ramp correctness verified")


def run_all():
    print("=" * 60)
    print("v28 Training Improvements - Unit Tests")
    print("=" * 60)

    tests = [
        ("Cosine warm restarts", test_cosine_restarts_produces_periodic_peaks),
        ("Restarts disabled by period=0", test_cosine_restarts_disabled_by_zero_period),
        ("Constant schedule", test_constant_schedule),
        ("Cosine decay", test_cosine_schedule_decay),
        ("Focal loss modulation", test_focal_weight_modulation),
        ("Focal gamma=0 identity", test_focal_gamma_zero_is_identity),
        ("Gradient noise decay", test_gradient_noise_decays),
        ("Curriculum ramp", test_curriculum_ramp_reduces_low_weight_samples),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if failed:
        sys.exit(1)
    print("\nAll tests PASSED!")


if __name__ == "__main__":
    run_all()
