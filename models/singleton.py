import gc
import torch
from models.loader import ModelExecutionPlan, load_model_pair


class ModelManager:
    _instance = None

    def __new__(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._plan = None
            cls._instance._model_a_path = None
            cls._instance._model_b_path = None
            cls._instance._adapter_path = None
            cls._instance._quantization = None
        return cls._instance

    @property
    def current_plan(self) -> ModelExecutionPlan | None:
        return self._plan

    def load_if_needed(
        self,
        model_a_path: str,
        model_b_path: str | None = None,
        adapter_path: str | None = None,
        quantization: str = "None",
        allow_mismatch: bool = False,
    ) -> ModelExecutionPlan:
        if (
            self._plan is not None
            and self._model_a_path == model_a_path
            and self._model_b_path == model_b_path
            and self._adapter_path == adapter_path
            and self._quantization == quantization
        ):
            return self._plan

        self._unload()

        self._plan = load_model_pair(
            model_a_path=model_a_path,
            model_b_path=model_b_path,
            adapter_path=adapter_path,
            quantization=quantization,
            allow_mismatch=allow_mismatch,
        )
        self._model_a_path = model_a_path
        self._model_b_path = model_b_path
        self._adapter_path = adapter_path
        self._quantization = quantization
        return self._plan

    def _unload(self) -> None:
        if self._plan is None:
            return
        del self._plan.model_a
        if self._plan.model_b is not None:
            del self._plan.model_b
        self._plan = None
        self._model_a_path = None
        self._model_b_path = None
        self._adapter_path = None
        self._quantization = None
        gc.collect()
        torch.cuda.empty_cache()

    def clear(self) -> None:
        self._unload()
