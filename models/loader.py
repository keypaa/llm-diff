from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dataclasses import dataclass, field


@dataclass
class ModelExecutionPlan:
    mode: str
    model_a: AutoModelForCausalLM
    model_b: Optional[AutoModelForCausalLM] = None
    tokenizer_a: Optional[AutoTokenizer] = None
    tokenizer_b: Optional[AutoTokenizer] = None
    warnings: list[str] = field(default_factory=list)


def get_quantization_config(quant_mode: str) -> Optional[BitsAndBytesConfig]:
    if quant_mode == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif quant_mode == "8-bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_model_pair(
    model_a_path: str,
    model_b_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    quantization: str = "None",
    allow_mismatch: bool = False,
) -> ModelExecutionPlan:
    quant_config = get_quantization_config(quantization)
    warnings: list[str] = []

    print(f"Loading Model A: {model_a_path} ({quantization} mode)...")
    tokenizer_a = AutoTokenizer.from_pretrained(model_a_path)
    model_a = AutoModelForCausalLM.from_pretrained(
        model_a_path,
        quantization_config=quant_config,
        device_map="auto",
    )

    if adapter_path and not model_b_path:
        print(f"Loading Adapter: {adapter_path} onto Model A...")
        model_a = PeftModel.from_pretrained(model_a, adapter_path)
        return ModelExecutionPlan(
            mode="single_model_adapter_swap",
            model_a=model_a,
            tokenizer_a=tokenizer_a,
            warnings=[],
        )

    elif model_b_path:
        print(f"Loading Model B: {model_b_path} ({quantization} mode)...")
        tokenizer_b = AutoTokenizer.from_pretrained(model_b_path)
        model_b = AutoModelForCausalLM.from_pretrained(
            model_b_path,
            quantization_config=quant_config,
            device_map="auto",
        )

        config_a = model_a.config
        config_b = model_b.config

        if config_a.architectures != config_b.architectures:
            warnings.append(
                f"Architecture Mismatch: Model A is {config_a.architectures}, "
                f"Model B is {config_b.architectures}."
            )

        if config_a.hidden_size != config_b.hidden_size:
            warnings.append(
                f"Hidden Dimension Mismatch: Model A ({config_a.hidden_size}) "
                f"vs Model B ({config_b.hidden_size}). "
                "Direct vector comparison disabled."
            )

        if len(tokenizer_a) != len(tokenizer_b):
            warnings.append(
                "Tokenizer Mismatch: Vocabulary sizes differ. "
                "Token-to-token alignment may be incorrect."
            )

        if warnings and not allow_mismatch:
            raise ValueError(
                "Compatibility validation failed. "
                "Enable 'Authorize Mismatched Models' to proceed. "
                f"Errors: {warnings}"
            )

        return ModelExecutionPlan(
            mode="dual_model_comparison",
            model_a=model_a,
            model_b=model_b,
            tokenizer_a=tokenizer_a,
            tokenizer_b=tokenizer_b,
            warnings=warnings,
        )

    raise ValueError(
        "Invalid configuration. Provide either an adapter path OR a second model path."
    )
