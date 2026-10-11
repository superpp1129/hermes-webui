"""Tests for model-aware reasoning effort chip visibility."""

from api import config as cfg


GPT_5_6_MODELS = (
    "gpt-5.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
)

OPENAI_FAMILY_PROVIDERS = (
    "openai",
    "openai-api",
    "openai-codex",
    "azure",
    "azure-openai",
    "azure-foundry",
)


def test_cursor_acp_models_do_not_support_reasoning_effort_levels():
    assert cfg.resolve_model_reasoning_efforts(
        "cursor/composer-2.5",
        provider_id="cursor-acp",
    ) == []


def test_openai_codex_gpt5_supports_reasoning_effort_levels():
    efforts = cfg.resolve_model_reasoning_efforts(
        "gpt-5.5",
        provider_id="openai-codex",
    )
    assert "medium" in efforts
    assert "high" in efforts
    assert "xhigh" in efforts
    assert "max" not in efforts


def test_openai_codex_prefixed_gpt5_supports_reasoning_effort_levels():
    efforts = cfg.resolve_model_reasoning_efforts(
        "@openai-codex:gpt-5.5",
        provider_id="openai-codex",
    )
    assert "medium" in efforts
    assert "high" in efforts
    assert "xhigh" in efforts
    assert "max" not in efforts


def test_manual_codex_override_is_not_relabelled():
    assert cfg.coerce_reasoning_effort_for_model("max", "gpt-5.5", provider_id="openai-codex") == "max"


def test_openai_family_gpt56_models_expose_and_preserve_max():
    for provider in OPENAI_FAMILY_PROVIDERS:
        for model in GPT_5_6_MODELS:
            efforts = cfg.resolve_model_reasoning_efforts(
                f"@{provider}:{model}",
                provider_id=provider,
            )
            assert "max" in efforts, f"{model} on {provider} must expose max"
            assert cfg.coerce_reasoning_effort_for_model(
                "max",
                f"@{provider}:{model}",
                provider_id=provider,
            ) == "max", f"{model} on {provider} must preserve max"


def test_manual_o_series_override_is_preserved():
    for effort in ("xhigh", "max"):
        assert cfg.coerce_reasoning_effort_for_model(effort, "o3-mini", provider_id="openai-codex") == effort


def test_coerce_never_escalates_above_configured_effort():
    # A supported lower effort is returned verbatim; coercion only degrades.
    assert cfg.coerce_reasoning_effort_for_model(
        "low",
        "gpt-5.5",
        provider_id="openai-codex",
    ) == "low"


def test_unknown_model_does_not_guess_effort_support():
    for effort in ("high", "max", "none", ""):
        assert cfg.coerce_reasoning_effort_for_model(effort, "unknown-model", provider_id="custom") == effort


def test_copilot_does_not_infer_controls_from_name():
    capability = cfg._reasoning_capability("gpt-5.5", "github-copilot")
    assert capability["efforts"] == []
    assert capability["known"] is False


def test_openrouter_uses_exact_catalog(monkeypatch):
    import hermes_cli.models as models
    monkeypatch.setattr(models, "openrouter_model_reasoning_capabilities",
                        lambda *a, **k: {"supports_reasoning": True, "supported_efforts": ["low", "high"], "mandatory": True})
    assert cfg.resolve_model_reasoning_efforts("anthropic/claude-sonnet-4.5", provider_id="openrouter") == ["low", "high"]


def test_non_reasoning_http_models_hide_reasoning_effort_levels():
    assert cfg.resolve_model_reasoning_efforts(
        "meta-llama/llama-3.1-8b-instruct",
        provider_id="openrouter",
    ) == []


def test_provider_config_reasoning_efforts_return_filtered_deduped(monkeypatch):
    original = cfg.cfg.get("providers")
    monkeypatch.setitem(
        cfg.cfg,
        "providers",
        {
            "wandb": {
                "reasoning_efforts": [
                    " none ",
                    "HIGH",
                    "bogus",
                    "high",
                    "xhigh",
                ]
            }
        },
    )
    try:
        assert cfg.resolve_model_reasoning_efforts(
            "zai-org/GLM-5.2",
            provider_id="wandb",
        ) == ["none", "high", "xhigh"]
    finally:
        if original is None:
            cfg.cfg.pop("providers", None)
        else:
            monkeypatch.setitem(cfg.cfg, "providers", original)


def test_provider_config_all_invalid_falls_through(monkeypatch):
    original = cfg.cfg.get("providers")
    monkeypatch.setitem(
        cfg.cfg,
        "providers",
        {"wandb": {"reasoning_efforts": ["bogus", "typo"]}},
    )
    try:
        result = cfg.resolve_model_reasoning_efforts(
            "zai-org/GLM-5.2",
            provider_id="wandb",
        )
        assert result == []
        assert "bogus" not in result
        assert "typo" not in result
    finally:
        if original is None:
            cfg.cfg.pop("providers", None)
        else:
            monkeypatch.setitem(cfg.cfg, "providers", original)


def test_named_custom_provider_config_reasoning_efforts(monkeypatch):
    original = cfg.cfg.get("custom_providers")
    monkeypatch.setitem(
        cfg.cfg,
        "custom_providers",
        [{"name": "llm-proxy", "reasoning_efforts": ["none", "high", "xhigh"]}],
    )
    try:
        assert cfg.resolve_model_reasoning_efforts(
            "some-model",
            provider_id="custom:llm-proxy",
        ) == ["none", "high", "xhigh"]
    finally:
        if original is None:
            cfg.cfg.pop("custom_providers", None)
        else:
            monkeypatch.setitem(cfg.cfg, "custom_providers", original)


def test_named_custom_provider_model_reasoning_efforts_take_precedence(monkeypatch):
    original = cfg.cfg.get("custom_providers")
    monkeypatch.setitem(
        cfg.cfg,
        "custom_providers",
        [
            {
                "name": "llm-proxy",
                "reasoning_efforts": ["high"],
                "models": {
                    "Inkling": {
                        "reasoning_efforts": ["none", "low", "medium", "high", "xhigh"]
                    }
                },
            }
        ],
    )
    try:
        assert cfg.resolve_model_reasoning_efforts(
            "inkling",
            provider_id="custom:llm-proxy",
        ) == ["none", "low", "medium", "high", "xhigh"]
        assert cfg.resolve_model_reasoning_efforts(
            "another-model",
            provider_id="custom:llm-proxy",
        ) == ["high"]
    finally:
        if original is None:
            cfg.cfg.pop("custom_providers", None)
        else:
            monkeypatch.setitem(cfg.cfg, "custom_providers", original)


def test_model_reasoning_efforts_fall_back_to_provider_when_invalid(monkeypatch):
    original = cfg.cfg.get("providers")
    monkeypatch.setitem(
        cfg.cfg,
        "providers",
        {
            "wandb": {
                "reasoning_efforts": ["none", "high"],
                "models": {"inkling": {"reasoning_efforts": ["bogus", "typo"]}},
            }
        },
    )
    try:
        assert cfg.resolve_model_reasoning_efforts(
            "inkling",
            provider_id="wandb",
        ) == ["none", "high"]
    finally:
        if original is None:
            cfg.cfg.pop("providers", None)
        else:
            monkeypatch.setitem(cfg.cfg, "providers", original)


def test_acp_guards_win_over_configured_reasoning_efforts(monkeypatch):
    original = cfg.cfg.get("providers")
    monkeypatch.setitem(
        cfg.cfg,
        "providers",
        {"copilot-acp": {"reasoning_efforts": ["high"]}},
    )
    try:
        assert cfg.resolve_model_reasoning_efforts(
            "some-model",
            provider_id="copilot-acp",
        ) == []
    finally:
        if original is None:
            cfg.cfg.pop("providers", None)
        else:
            monkeypatch.setitem(cfg.cfg, "providers", original)


def test_nested_route_deny_wins_over_configured_reasoning_efforts(monkeypatch):
    original = cfg.cfg.get("custom_providers")
    monkeypatch.setitem(
        cfg.cfg,
        "custom_providers",
        [
            {
                "name": "agg",
                "reasoning_efforts": ["low", "high"],
                "models": {
                    "vertex/gemini-image-1.0": {"reasoning_efforts": ["high"]},
                    "vertex/gemini-embedding-001": {"reasoning_efforts": ["high"]},
                },
            }
        ],
    )
    try:
        for model in ("vertex/gemini-image-1.0", "vertex/gemini-embedding-001"):
            assert cfg.resolve_model_reasoning_efforts(
                model,
                provider_id="custom:agg",
            ) == []
    finally:
        if original is None:
            cfg.cfg.pop("custom_providers", None)
        else:
            monkeypatch.setitem(cfg.cfg, "custom_providers", original)


def test_nested_route_deny_wins_for_provider_qualified_hinted_model(monkeypatch):
    # Regression for the deeper bypass: a provider-qualified hint like
    # "@custom:agg:vertex/gemini-image-1.0" must strip BOTH the "@custom:"
    # wrapper AND the named-provider slug "agg:" before the nested-route
    # deny check runs. A naive first-colon split only strips "@custom:",
    # leaving "agg:vertex/gemini-image-1.0" — which no longer starts with
    # "vertex/gemini-" — so the deny is missed and the configured
    # ["low", "high"] leaks through on an image/embedding route.
    original = cfg.cfg.get("custom_providers")
    monkeypatch.setitem(
        cfg.cfg,
        "custom_providers",
        [{"name": "agg", "reasoning_efforts": ["low", "high"]}],
    )
    try:
        for model in (
            "@custom:agg:vertex/gemini-image-1.0",
            "@custom:agg:vertex/gemini-embedding-001",
        ):
            assert cfg.resolve_model_reasoning_efforts(
                model,
                provider_id="custom:agg",
            ) == []
    finally:
        if original is None:
            cfg.cfg.pop("custom_providers", None)
        else:
            monkeypatch.setitem(cfg.cfg, "custom_providers", original)


def test_nested_route_deny_is_boundary_based_not_prefix_based():
    # Structural regression test for the underlying invariant, independent
    # of any particular wrapper/strip scheme: _nested_route_reasoning_denied
    # must catch the vertex/gemini- or gemini_cli/gemini- route no matter
    # how many opaque wrapper layers precede it in the raw string, as long
    # as the route starts at a non-alphanumeric boundary. This was bypassed
    # twice via different prefix-stripping edge cases (PR #5313) before the
    # check itself was made boundary-based instead of prefix-based, so no
    # future wrapper scheme can reintroduce the same class of bug.
    denied_cases = [
        "vertex/gemini-image-1.0",
        "vertex/gemini-embedding-001",
        "@custom:agg:vertex/gemini-image-1.0",
        "agg:vertex/gemini-image-1.0",  # the literal leftover fragment from the historical bug
        "gemini_cli/gemini-imagine-2",
        "outer:inner:vertex/gemini-image-1.0",  # hypothetical deeper future nesting
        "@custom:outer:@custom:inner:vertex/gemini-embedding-001",
    ]
    for model in denied_cases:
        assert cfg._nested_route_reasoning_denied(model) is True, model

    allowed_cases = [
        "vertex/gemini-2.5-pro",
        "notvertex/gemini-image-1.0",  # embedded in a larger token — must NOT match
        "somegemini_cli/gemini-image-1",  # same — embedded substring, not a boundary
        "",
    ]
    for model in allowed_cases:
        assert cfg._nested_route_reasoning_denied(model) is False, model


def test_get_reasoning_status_includes_supported_efforts(monkeypatch):
    monkeypatch.setattr(cfg, "_reasoning_capability", lambda *a, **k: {"known": True, "efforts": ["low"], "source": "test"})
    status = cfg.get_reasoning_status(model_id="gpt-5.5", provider_id="openai-codex")
    assert status["supported_efforts"] == ["none", *cfg.VALID_REASONING_EFFORTS]
    assert status["supports_reasoning_effort"] is True
    assert status["supports_thinking_toggle"] is True


def test_manual_status_includes_max_and_ultra():
    status = cfg.get_reasoning_status(model_id="gpt-5.5", provider_id="openai-codex")
    assert status["supported_efforts"] == ["none", *cfg.VALID_REASONING_EFFORTS]
    assert "ultra" in status["supported_efforts"]


def test_get_reasoning_status_does_not_duplicate_configured_none(monkeypatch):
    monkeypatch.setattr(cfg, "cfg", {"providers": {"test": {"reasoning_efforts": ["none", "none", "high"]}}})
    status = cfg.get_reasoning_status(model_id="test", provider_id="test")
    assert status["supported_efforts"].count("none") == 1
    assert status["supported_efforts"] == ["none", *cfg.VALID_REASONING_EFFORTS]


def test_manual_status_preserves_saved_preference(monkeypatch):
    monkeypatch.setattr(cfg, "_load_yaml_config_file", lambda *a, **k: {"agent": {"reasoning_effort": "max"}})
    status = cfg.get_reasoning_status(model_id="gpt-5.5", provider_id="openai-codex")
    assert status["reasoning_effort"] == "max"
    assert status["requested_reasoning_effort"] == "max"
    assert "Manual override" in status["reasoning_note"]


def test_gemini_does_not_offer_alias_levels():
    for model in ("gemini-3-pro", "gemini-3-flash"):
        assert cfg.coerce_reasoning_effort_for_model("max", model_id=model, provider_id="gemini") == "max"
        assert "xhigh" not in cfg.resolve_model_reasoning_efforts(model, provider_id="gemini")


def test_legacy_anthropic_does_not_offer_max():
    for model in ("claude-3-7-sonnet", "claude-sonnet-4-5", "claude-haiku-4-5",
                  "claude-3-opus-20240229", "claude-3-5-sonnet-20241022",
                  "claude-sonnet-4-20250514", "claude-opus-4-20250514"):
        assert cfg.coerce_reasoning_effort_for_model("max", model_id=model, provider_id="anthropic") == "max"


def test_native_capabilities_do_not_bleed_into_other_routes():
    for model in ("claude-opus-4.6", "claude-sonnet-4.6", "claude-opus-4.7"):
        assert cfg.coerce_reasoning_effort_for_model("max", model_id=model, provider_id="anthropic") == "max"
    assert cfg.coerce_reasoning_effort_for_model("max", model_id="deepseek-reasoner", provider_id="deepseek") == "max"


def test_unsupported_max_not_relabelled_across_openai_routes():
    for provider in OPENAI_FAMILY_PROVIDERS:
        for model in ("gpt-5", "gpt-5.1", "gpt-5.5", "o1", "o3-mini", "o4-mini"):
            assert cfg.coerce_reasoning_effort_for_model("max", model_id=model, provider_id=provider) == "max"


def test_hosted_claude_requires_its_own_route_capabilities():
    for provider in ("azure-foundry", "bedrock"):
        for model in ("claude-sonnet-4-20250514", "claude-opus-4.6"):
            assert cfg.coerce_reasoning_effort_for_model("max", model_id=model, provider_id=provider) == "max"


def test_unknown_provider_preserves_manual_overrides():
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
        assert cfg.coerce_reasoning_effort_for_model(effort, model_id="unknown", provider_id="customprovider") == effort


def test_max_only_offered_in_ui_when_actually_supported():
    # The dropdown gates on resolve_model_reasoning_efforts(): 'max' appears ONLY
    # for models whose supported list includes it (adaptive Claude, DeepSeek), and
    # is absent for legacy/capped models and unknown providers.
    assert "max" in cfg.resolve_model_reasoning_efforts("claude-opus-4.6", provider_id="anthropic")
    assert "max" not in cfg.resolve_model_reasoning_efforts("deepseek-reasoner", provider_id="deepseek")
    assert "max" not in cfg.resolve_model_reasoning_efforts("claude-sonnet-4-5", provider_id="anthropic")
    assert "max" not in cfg.resolve_model_reasoning_efforts("gpt-5.1", provider_id="openai")
    assert "max" not in cfg.resolve_model_reasoning_efforts("gemini-3-pro", provider_id="gemini")


def test_datestamped_claude3_not_reasoning_capable_heuristic():
    # A bare, date-stamped Claude 3.0 id must NOT be treated as reasoning-capable
    # by the heuristic. The minor-version capture previously used `(\d+)`, which
    # swallowed the 8-digit date stamp ("...-20240229") as the minor version so
    # `major==3 and minor>=7` wrongly matched — surfacing reasoning-effort
    # controls for models that don't support them. Claude 3.0/3.5 have no
    # extended-thinking support; only 3.7+ (and 4.x) do.
    for model in (
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-opus",
        "claude-3-5-sonnet-20241022",
    ):
        assert cfg._candidate_supports_reasoning(model) is False, (
            f"{model} must not be reasoning-capable (Claude 3.0/3.5 excluded)"
        )
    # 3.7+ and 4.x (including date-stamped builds) stay reasoning-capable.
    for model in (
        "claude-3-7-sonnet",
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4-5",
        "claude-opus-4-20250514",
        "claude-opus-4.6",
    ):
        assert cfg._candidate_supports_reasoning(model) is True, (
            f"{model} must remain reasoning-capable"
        )


def test_qwen_prefixed_alias_reasoning_detection():
    """Prefixed Qwen IDs (e.g. New-API aliases) must still be detected.

    Regression: "al-qwen3.8-max-preview" normalizes to tokens
    ["al", "qwen3", "8", ...] where "qwen" is NOT a standalone token,
    so the old exact-membership check silently failed.
    """
    # Prefixed Qwen 3+ → reasoning-capable
    for model in (
        "al-qwen3.8-max-preview",
        "al-qwen3.7-max",
        "al-qwen3.7-plus",
        "al-qwen3.6-flash",
        "sn-qwen3-235b-a22b",
    ):
        assert cfg._candidate_supports_reasoning(model) is True, (
            f"{model} must be reasoning-capable (prefixed Qwen 3+)"
        )
    # Bare Qwen 3+ → still works
    for model in (
        "qwen3-235b-a22b",
        "qwen3-32b",
    ):
        assert cfg._candidate_supports_reasoning(model) is True, (
            f"{model} must be reasoning-capable (bare Qwen 3+)"
        )
    # Qwen 2.x → excluded regardless of prefix
    for model in (
        "al-qwen2.5-72b-instruct",
        "qwen2.5-7b-instruct",
        "qwen2-72b",
    ):
        assert cfg._candidate_supports_reasoning(model) is False, (
            f"{model} must NOT be reasoning-capable (Qwen 2.x excluded)"
        )
    # Hybrid IDs with embedded Qwen 2.x must NOT be shadowed by the Qwen
    # branch — they must fall through to the DeepSeek detector.
    for model in (
        "deepseek-r1-distill-qwen2.5-bakeneko-32b",
        "rinna/deepseek-r1-distill-qwen2.5-bakeneko-32b",
    ):
        assert cfg._candidate_supports_reasoning(model) is True, (
            f"{model} must remain reasoning-capable (DeepSeek-R1 hybrid, "
            f"Qwen 2.x must not shadow the DeepSeek detector)"
        )
