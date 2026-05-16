#!/usr/bin/env python3
import numpy as np
from rknnlite.api import RKNNLite


from pathlib import Path

class RKNNRunner:
  """Small wrapper around RKNNLite for persistent model execution, with HAL-inspired improvements."""

  def __init__(self, model_path: str, use_npu_cores: str = "0"):
    self._released = False
    self.rknn = RKNNLite()
    if not Path(model_path).exists():
      raise FileNotFoundError(f"[RKNNRunner] Model not found: {model_path}")
    ret = self.rknn.load_rknn(model_path)
    if ret != 0:
      raise RuntimeError(f"[RKNNRunner] Failed to load model: {model_path}")
    core_mask = self._parse_core_mask(use_npu_cores)
    ret = self.rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
      raise RuntimeError(f"[RKNNRunner] Failed to init runtime (core_mask={core_mask})")
    print(f"[RKNNRunner] Loaded {model_path} pinned to NPU core mask={core_mask}")

  def _parse_core_mask(self, use_npu_cores: str) -> int:
    if use_npu_cores == "all":
      return RKNNLite.NPU_CORE_AUTO
    mapping = {"0": RKNNLite.NPU_CORE_0, "1": RKNNLite.NPU_CORE_1, "2": RKNNLite.NPU_CORE_2}
    mask = 0
    for part in use_npu_cores.split(","):
      p = part.strip()
      if p not in mapping:
        raise ValueError(f"[RKNNRunner] Invalid NPU core: {p}")
      mask |= mapping[p]
    return mask

  def infer(self, inputs) -> list[np.ndarray]:
    # Accept dict or list
    if isinstance(inputs, dict):
      # Use insertion order (Python 3.7+)
      inputs = list(inputs.values())
    # Prefer the high-level call when available.
    if hasattr(self.rknn, 'inference'):
      outputs = self.rknn.inference(inputs=inputs)
    else:
      ret = self.rknn.inputs_set(inputs)
      if ret != 0:
        raise RuntimeError("[RKNNRunner] inputs_set failed — check input shapes match model")
      ret = self.rknn.run()
      if ret != 0:
        raise RuntimeError("[RKNNRunner] run failed")
      outputs = self.rknn.outputs_get()
    if outputs is None:
      raise RuntimeError("RKNN outputs_get returned None")
    return outputs

  def release(self):
    if not self._released:
      self.rknn.release()
      self._released = True

  def __del__(self):
    self.release()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.release()
