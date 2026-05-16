"""Inference HAL interface used by modeld.

This module exposes a stable client/backend API so modeld does not talk to
RKNNLite directly. The current backend uses RKNNRunner under the hood.
"""

from dataclasses import dataclass

import numpy as np

from openpilot.common.swaglog import cloudlog

import os
try:
    from rknnlite.api import RKNNLite
    from openpilot.selfdrive.modeld.runners.rknn_runner import RKNNRunner
    # Broaden /dev/rknpu0 check to also match /dev/rknpu
    HAS_RKNN = any(os.path.exists(p) for p in ['/dev/rknpu0', '/dev/rknpu'])
except Exception as e:
    RKNNLite = None
    RKNNRunner = None
    HAS_RKNN = False
    _RKNN_IMPORT_ERR = str(e)


@dataclass
class ModelConfig:
    name: str
    path: str
    input_shapes: dict[str, tuple[int, ...]]
    output_shapes: dict[str, tuple[int, ...]]
    npu_cores: int = 0


@dataclass
class InferenceResult:
    outputs: dict[str, np.ndarray]
    success: bool = True
    error_message: str | None = None


class NPUBackend:
    def __init__(self):
        self.models: dict[str, dict] = {}


    def _core_str(self, core_idx: int) -> str:
        # Map int to string for use_npu_cores param
        if core_idx == 1:
            return "1"
        if core_idx == 2:
            return "2"
        return "0"

    def load_model(self, config: ModelConfig) -> bool:
        if not HAS_RKNN:
            cloudlog.error(f"HAL backend unavailable: rknnlite import failed: {_RKNN_IMPORT_ERR}")
            return False

        try:
            runner = RKNNRunner(config.path, use_npu_cores=self._core_str(config.npu_cores))
            self.models[config.name] = {
                "runner": runner,
                "input_names": list(config.input_shapes.keys()),
            }
            cloudlog.warning(f"HAL loaded {config.name} on NPU core(s) {self._core_str(config.npu_cores)}")
            return True
        except Exception as e:
            cloudlog.exception(f"HAL failed loading {config.name}: {e}")
            return False

    def infer(self, model_name: str, inputs: dict[str, np.ndarray]) -> InferenceResult:
        if model_name not in self.models:
            return InferenceResult(outputs={}, success=False, error_message=f"Model not loaded: {model_name}")

        model = self.models[model_name]
        try:
            ordered_inputs = []
            for input_name in model["input_names"]:
                if input_name not in inputs:
                    return InferenceResult(outputs={}, success=False, error_message=f"Missing input: {input_name}")
                ordered_inputs.append(inputs[input_name])

            # Print policy input order at startup for verification
            if model_name == "driving_policy" and not hasattr(self, "_policy_input_printed"):
                print(f"[RKNN] Policy input order: {list(model['input_names'])}")
                self._policy_input_printed = True

            out_list = model["runner"].infer(ordered_inputs)
            out = np.asarray(out_list[0], dtype=np.float32).reshape(-1)
            if not np.all(np.isfinite(out)):
                raise RuntimeError(f"{model_name} output contains NaN/Inf")
            return InferenceResult(outputs={"outputs": out}, success=True)
        except Exception as e:
            return InferenceResult(outputs={}, success=False, error_message=str(e))

    def release(self):
        for model in self.models.values():
            try:
                model["runner"].release()
            except Exception:
                pass
        self.models.clear()


class InferenceClient:
    def __init__(self, client_name: str = "modeld"):
        self.client_name = client_name
        self._npu = NPUBackend()

    def npu(self) -> NPUBackend:
        return self._npu

    def release(self):
        self._npu.release()
