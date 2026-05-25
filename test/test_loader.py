from models.loader import ModelExecutionPlan, get_quantization_config


def test_model_execution_plan_defaults():
    plan = ModelExecutionPlan(mode="dual_model_comparison", model_a=None)
    assert plan.mode == "dual_model_comparison"
    assert plan.warnings == []


def test_model_execution_plan_with_warnings():
    plan = ModelExecutionPlan(
        mode="single_model_adapter_swap",
        model_a=None,
        warnings=["Hidden Dimension Mismatch"],
    )
    assert len(plan.warnings) == 1


def test_get_quantization_config_none():
    assert get_quantization_config("None") is None


def test_get_quantization_config_8bit():
    config = get_quantization_config("8-bit")
    assert config is not None
    assert config.load_in_8bit is True


def test_get_quantization_config_4bit():
    config = get_quantization_config("4-bit")
    assert config is not None
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"


def test_get_quantization_config_unknown():
    assert get_quantization_config("unknown") is None
