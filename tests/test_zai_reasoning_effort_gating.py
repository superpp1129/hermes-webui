"""Native GLM controls: 5.2 high/max with off; 5.3 low/high/max without off.
Toggle-only models expose None, while mandatory-thinking models do not.
Unknown future models and third-party routes must not inherit guessed levels.
"""

import pytest

import api.config as cfg


# ── GLM-5.2: full ladder preserved (the only model that supports reasoning_effort) ─

def test_glm52_offers_only_native_controls():
    assert cfg.resolve_model_reasoning_efforts("glm-5.2", provider_id="zai") == ["none", "high", "max"]


def test_glm_5_2_preserves_none_sentinel(monkeypatch):
    monkeypatch.setattr(cfg, "cfg", {"providers": {"zai": {"reasoning_efforts": ["none", "low", "high", "max"]}}})
    assert cfg.resolve_model_reasoning_efforts("glm-5.2", provider_id="zai") == ["none", "high", "max"]


@pytest.mark.parametrize(
    "model_id",
    ["glm-5.3", "glm-5.2-air", "glm-6-pro", "glm-6", "glm-5.2.1"],
)
def test_glm_versions_do_not_share_a_universal_ladder(model_id):
    expected = (["low", "high", "max"] if model_id == "glm-5.3" else
                ["none", "high", "max"] if model_id.startswith("glm-5.2") else [])
    assert cfg.resolve_model_reasoning_efforts(model_id, provider_id="zai") == expected


@pytest.mark.parametrize(
    "model_id",
    [
        "glm-5.1",
        "glm-5",
        "glm-5-turbo",
        "glm-4.5",
        "glm-4.5-flash",
    ],
)
def test_pre_5_2_glm_drops_reasoning_effort_ladder(model_id):
    assert cfg.resolve_model_reasoning_efforts(model_id, provider_id="zai") == ["none"]


def test_glm_4_7_forced_thinking_drops_entire_ladder():
    """Bug 3: GLM-4.7 uses forced thinking and cannot be disabled."""
    efforts = cfg.resolve_model_reasoning_efforts("glm-4.7", provider_id="zai")
    assert efforts == [], (
        "glm-4.7 uses forced thinking per Z.AI docs — no reasoning_effort "
        f"and no 'none'; got {efforts!r}"
    )


def test_glm_4_7_air_forced_thinking_drops_entire_ladder():
    efforts = cfg.resolve_model_reasoning_efforts("glm-4.7-air", provider_id="zai")
    assert efforts == []


# ── Three-tier classification: effort / thinking / forced ────────────────────────
# Per docs.z.ai, the three Z.AI capability tiers are distinct:
#   - GLM-5.2+:           effort ladder (max..minimal) AND thinking toggle
#   - GLM-4.5–5.1:        thinking toggle ONLY (no effort ladder)
#   - GLM-4.7:            forced thinking (neither toggle nor ladder)
# Returning [] for the effort ladder must NOT also hide the thinking on/off
# control for the middle tier — otherwise GLM-4.5/4.6/5.0/5.1 users lose the
# working thinking toggle they had before (#6219 round-2 review).


@pytest.mark.parametrize(
    "model_id,expected",
    [
        # GLM-5.2+ → effort ladder + thinking toggle
        ("glm-5.2", "effort"),
        ("glm-5.2-air", "effort"),
        ("glm-5.3", "effort"),
        ("glm-6", "effort"),
        # GLM-4.5 up to (but not including) 5.2 → thinking toggle only
        ("glm-5.1", "thinking"),
        ("glm-5", "thinking"),
        ("glm-5-turbo", "thinking"),
        ("glm-4.5", "thinking"),
        ("glm-4.5-flash", "thinking"),
        ("glm-4.5-air", "thinking"),
        ("glm-4.6", "thinking"),
        # GLM-4.7 family → forced thinking (no toggle, no ladder)
        ("glm-4.7", "forced"),
        ("glm-4.7-air", "forced"),
        # Non-zai / non-glm → defer (None)
        ("gpt-5", None),
        ("claude-sonnet-4.6", None),
    ],
)
def test_zai_glm_classification_three_tiers(model_id, expected):
    cls = cfg._zai_glm_classification(model_id, provider_id="zai")
    assert cls == expected, (
        f"{model_id} via zai should classify as {expected!r}, got {cls!r}"
    )


def test_classification_none_for_non_zai_provider():
    """Aggregators must defer (None) — they route through their own routers."""
    assert cfg._zai_glm_classification("glm-5.2", provider_id="openrouter") is None
    assert cfg._zai_glm_classification("glm-5.2", provider_id="custom:newapi") is None


def test_classification_none_for_non_glm_on_zai():
    """A non-GLM model routed through zai must defer (None)."""
    assert cfg._zai_glm_classification("gpt-5", provider_id="zai") is None
    assert cfg._zai_glm_classification("claude-sonnet-4.6", provider_id="zai") is None


# ── supports_thinking_toggle: chip visibility contract ──────────────────────────


def _reasoning_status(model_id, provider_id="zai"):
    """Helper: call get_reasoning_status with a stub config so it does not depend
    on the active profile's config.yaml."""
    import unittest.mock as mock
    with mock.patch("api.config._load_yaml_config_file", return_value={}):
        return cfg.get_reasoning_status(
            model_id=model_id, provider_id=provider_id
        )


def test_glm_5_2_status_offers_effort_ladder_and_toggle():
    st = _reasoning_status("glm-5.2")
    assert st["supported_efforts"] == ["none", *cfg.VALID_REASONING_EFFORTS]
    assert st["supports_reasoning_effort"] is True
    assert st["supports_thinking_toggle"] is True
    assert st["reasoning_selection_mode"] == "manual_full_range"


@pytest.mark.parametrize(
    "model_id",
    ["glm-4.6", "glm-4.5", "glm-4.5-flash", "glm-5", "glm-5.1", "glm-5-turbo"],
)
def test_sub_5_2_glm_status_keeps_thinking_toggle_with_full_manual_ladder(model_id):
    st = _reasoning_status(model_id)
    assert st["supported_efforts"] == ["none", *cfg.VALID_REASONING_EFFORTS]
    assert st["supports_reasoning_effort"] is True
    assert st["supports_thinking_toggle"] is True
    assert st["reasoning_selection_mode"] == "manual_full_range"


def test_glm_4_7_status_offers_manual_controls():
    st = _reasoning_status("glm-4.7")
    assert st["supported_efforts"] == ["none", *cfg.VALID_REASONING_EFFORTS]
    assert st["supports_reasoning_effort"] is True
    assert st["supports_thinking_toggle"] is True
    assert st["reasoning_selection_mode"] == "manual_full_range"


@pytest.mark.parametrize("alias", ["glm", "z-ai", "z.ai", "zhipu"])
def test_thinking_toggle_aliases_resolve_through_same_gate(alias):
    for model in ("glm-4.6", "glm-4.7"):
        status = _reasoning_status(model, provider_id=alias)
        assert status["supports_thinking_toggle"] is True
        assert status["supported_efforts"] == ["none", *cfg.VALID_REASONING_EFFORTS]


def test_manual_flags_are_controls_not_native_capability_claims():
    for model in ("gpt-5", "gpt-4o"):
        st = _reasoning_status(model, provider_id="openai")
        assert st["supports_reasoning_effort"] is True
        assert st["supports_thinking_toggle"] is True
        assert st["reasoning_capabilities_known"] is False


@pytest.mark.parametrize("alias", ["glm", "z-ai", "z.ai", "zhipu"])
def test_zai_aliases_resolve_through_same_gate(alias):
    assert cfg.resolve_model_reasoning_efforts("glm-5.2", provider_id=alias) == ["none", "high", "max"]
    assert cfg.resolve_model_reasoning_efforts("glm-5.1", provider_id=alias) == ["none"]
    assert cfg.resolve_model_reasoning_efforts("glm-4.7", provider_id=alias) == []


@pytest.mark.parametrize(
    "model_id, provider_id",
    [
        ("glm-5.1:free", "kilocode"),
        ("glm-6-pro", "custom:newapi"),
        ("glm-4.5-flash", "openrouter"),
    ],
)
def test_aggregator_requires_route_specific_capabilities(model_id, provider_id, monkeypatch):
    import hermes_cli.models as models
    monkeypatch.setattr(models, "openrouter_model_reasoning_capabilities", lambda *a, **k: None)
    assert cfg.resolve_model_reasoning_efforts(model_id, provider_id=provider_id) == []


def test_non_glm_on_zai_is_not_assumed_supported():
    assert cfg.resolve_model_reasoning_efforts("gpt-5", provider_id="zai") == []


@pytest.mark.parametrize(
    "model_id",
    ["glm-5.1", "glm-5", "glm-5-turbo", "glm-4.5", "glm-4.5-flash", "glm-4.7"],
)
def test_manual_levels_preserved_for_pre_5_2_glm(model_id):
    for level in ("none", *cfg.VALID_REASONING_EFFORTS):
        assert cfg.coerce_reasoning_effort_for_model(level, model_id, provider_id="zai") == level


def test_coerce_preserves_levels_for_glm_5_2():
    for level in ("none", *cfg.VALID_REASONING_EFFORTS):
        assert cfg.coerce_reasoning_effort_for_model(level, "glm-5.2", provider_id="zai") == level


@pytest.mark.parametrize("alias", ["glm", "z-ai", "z.ai", "zhipu"])
def test_coerce_zai_aliases_resolve_through_same_gate(alias):
    assert cfg.coerce_reasoning_effort_for_model("high", "glm-5.1", provider_id=alias) == "high"


def test_unknown_custom_route_preserves_manual_selection():
    assert cfg.coerce_reasoning_effort_for_model("high", "unknown-model", provider_id="custom:myrouter") == "high"


@pytest.mark.parametrize("model_id", ["glm-4.7", "glm-4.7-air"])
def test_manual_none_selection_reaches_adapter_for_forced_glm(model_id):
    assert cfg.coerce_reasoning_effort_for_model("none", model_id, provider_id="zai") == "none"


@pytest.mark.parametrize("model_id", ["glm-5.2", "glm-4.6", "glm-5.1"])
def test_coerce_preserves_none_for_non_forced_glm(model_id):
    """Regression guard: non-forced GLM models MUST still accept 'none'."""
    coerced = cfg.coerce_reasoning_effort_for_model(
        "none", model_id, provider_id="zai"
    )
    assert coerced == "none", (
        f"{model_id} is not forced-thinking — 'none' must be preserved; "
        f"got {coerced!r}"
    )


def test_resolve_does_not_reattach_none_for_forced_glm(monkeypatch):
    monkeypatch.setattr(cfg, "cfg", {"providers": {"zai": {"reasoning_efforts": ["none", "high", "max"]}}})
    for model in ("glm-4.7", "glm-4.7-air", "glm-5.3"):
        assert "none" not in cfg.resolve_model_reasoning_efforts(model, provider_id="zai")


def test_resolve_reattaches_none_for_thinking_tier(monkeypatch):
    monkeypatch.setattr(cfg, "cfg", {"providers": {"zai": {"reasoning_efforts": ["none", "high"]}}})
    assert cfg.resolve_model_reasoning_efforts("glm-4.6", provider_id="zai") == ["none"]


def test_manual_status_retains_none_even_for_forced_glm():
    from unittest import mock
    with mock.patch("api.config._load_yaml_config_file", return_value={"agent": {"reasoning_effort": "none"}}):
        st = cfg.get_reasoning_status(model_id="glm-4.7", provider_id="zai")
    assert st["reasoning_effort"] == "none"
    assert st["supports_thinking_toggle"] is True
    assert st["reasoning_selection_mode"] == "manual_full_range"


def test_set_reasoning_effort_accepts_empty_as_clear():
    """Gap #1 backend: empty effort clears the override (Default/On path)."""
    import unittest.mock as mock
    saved = {}

    def fake_save(path, data):
        saved.update(data.get("agent", {}) or {})

    with mock.patch("api.config._save_yaml_config_file", side_effect=fake_save), \
         mock.patch("api.config.reload_config"):
        result = cfg.set_reasoning_effort("", model_id="glm-4.6", provider_id="zai")
    # Empty effort must remove the key entirely (not write empty string).
    assert "reasoning_effort" not in saved, (
        f"empty effort must clear agent.reasoning_effort; saved agent cfg={saved}"
    )
    assert result["reasoning_effort"] == ""


def test_set_reasoning_effort_still_rejects_invalid():
    """Regression guard: invalid effort values must still raise ValueError."""
    import unittest.mock as mock
    with mock.patch("api.config._save_yaml_config_file"), \
         mock.patch("api.config.reload_config"):
        with pytest.raises(ValueError):
            cfg.set_reasoning_effort("banana", model_id="glm-4.6", provider_id="zai")


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
def test_set_valid_manual_vocabulary_needs_no_capability_lookup(effort):
    from unittest import mock
    with mock.patch("api.config._save_yaml_config_file") as save, mock.patch("api.config.reload_config"):
        cfg.set_reasoning_effort(effort, model_id="glm-4.6", provider_id="zai")
        assert save.call_args.args[1]["agent"]["reasoning_effort"] == effort
