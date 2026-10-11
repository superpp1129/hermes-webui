"""Manual full-range controls must survive unknown/local model capabilities."""
import pytest
import yaml
from api import config as cfg

EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
ROUTES = [
    ("openai-codex", "gpt-5.6-sol"),
    ("openai-codex", "gpt-6-astra"),
    ("custom:strata3090", "qwen3.8-flash-next-iq2_xs"),
    ("custom", "qwen3.8-flash-next-iq2_xs"),
    ("azure-foundry", "gpt-5-mini"),
    ("xai-oauth", "grok-4.7"),
    ("byteplus-subscription", "ark-code-latest"),
    ("unknown", "unknown"),
]


@pytest.mark.parametrize("provider,model", ROUTES)
def test_all_routes_offer_full_picker_without_discovery(monkeypatch, provider, model):
    monkeypatch.setattr(cfg, "_load_yaml_config_file", lambda _: {"agent": {"reasoning_effort": "ultra"}})
    status = cfg.get_reasoning_status(model_id=model, provider_id=provider)
    assert status["supported_efforts"] == EFFORTS
    assert status["supports_reasoning_effort"] is True
    assert status["supports_thinking_toggle"] is True
    assert status["reasoning_effort"] == "ultra"
    assert status["reasoning_selection_mode"] == "manual_full_range"


@pytest.mark.parametrize("effort", ["", *EFFORTS])
def test_local_qwen_selection_persists_and_round_trips(tmp_path, monkeypatch, effort):
    path = tmp_path / "config.yaml"
    path.write_text("agent: {}\n", encoding="utf-8")
    monkeypatch.setattr(cfg, "_get_config_path", lambda: path)
    monkeypatch.setattr(cfg, "reload_config", lambda: None)
    status = cfg.set_reasoning_effort(effort, model_id="qwen3.8-flash-next-iq2_xs", provider_id="custom:strata3090")
    assert status["reasoning_effort"] == effort
    saved = yaml.safe_load(path.read_text()).get("agent", {})
    if effort:
        assert saved["reasoning_effort"] == effort
    else:
        assert "reasoning_effort" not in saved
    parsed = cfg.parse_reasoning_effort(effort)
    assert parsed == ({"enabled": False} if effort == "none" else
                      {"enabled": True, "effort": effort} if effort else None)


def test_invalid_vocabulary_still_rejected_before_write(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid selection must not be persisted")
    monkeypatch.setattr(cfg, "_save_yaml_config_file", unexpected)
    with pytest.raises(ValueError, match="Unknown reasoning effort"):
        cfg.set_reasoning_effort("banana", model_id="qwen", provider_id="custom")


def test_ultra_reaches_supported_and_unknown_routes_but_known_ceilings_clamp():
    assert cfg.coerce_reasoning_effort_for_model(
        "ultra", "gpt-5.6-sol", provider_id="openai-codex"
    ) == "ultra"
    assert cfg.coerce_reasoning_effort_for_model(
        "ultra", "qwen3.8-flash-next-iq2_xs", provider_id="custom:strata3090"
    ) == "ultra"
    assert cfg.coerce_reasoning_effort_for_model(
        "ultra", "gpt-5.5", provider_id="openai-codex"
    ) == "xhigh"
