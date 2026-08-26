"""A one-card service must take the emptiest visible card, not demand the floor from all of them.

On a packed node each Track B story gets its own generation card plus the node's shared services
card, so the small crop server can reach it. The shared card is nearly full by design, so the
all-must-clear check refused every story with ``CUDA_VISIBLE_DEVICES=2,0 does not satisfy
min_free_mib=44000`` even though card 2 was empty.
"""

from __future__ import annotations

import memstrata.lib.gpu as gpu


def _fake_free(monkeypatch, mapping: dict[int, int]) -> None:
    monkeypatch.setattr(gpu, "gpu_free_memory_mib", lambda: mapping)


def test_the_empty_card_is_chosen_over_the_busy_services_card(monkeypatch) -> None:
    _fake_free(monkeypatch, {0: 14123, 2: 45216})
    assert gpu.freest_visible_device("2,0", min_free_mib=44000) == "2"


def test_the_emptiest_wins_when_several_clear_the_floor(monkeypatch) -> None:
    _fake_free(monkeypatch, {3: 60000, 5: 80000})
    assert gpu.freest_visible_device("3,5", min_free_mib=44000) == "5"


def test_no_choice_is_made_when_nothing_clears_the_floor(monkeypatch) -> None:
    _fake_free(monkeypatch, {0: 14123, 2: 20000})
    assert gpu.freest_visible_device("2,0", min_free_mib=44000) is None


def test_a_single_visible_card_is_left_to_the_strict_check(monkeypatch) -> None:
    """With one card there is nothing to choose, so the caller's own guard should speak."""
    _fake_free(monkeypatch, {2: 80000})
    assert gpu.freest_visible_device("2", min_free_mib=44000) is None


def test_uuid_syntax_is_declined_rather_than_guessed(monkeypatch) -> None:
    _fake_free(monkeypatch, {0: 80000})
    assert gpu.freest_visible_device("GPU-abc,GPU-def", min_free_mib=1000) is None


def test_missing_nvidia_smi_is_declined(monkeypatch) -> None:
    monkeypatch.setattr(gpu, "gpu_free_memory_mib", lambda: None)
    assert gpu.freest_visible_device("2,0", min_free_mib=44000) is None
