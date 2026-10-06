"""Shared type aliases."""

from typing import Any

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
"""Feature matrices, probabilities and float labels."""

IntArray = npt.NDArray[np.int_]
"""Hard (0/1) predictions."""

Metrics = dict[str, float]
"""Metric name -> value, as returned by ``compute_metrics``."""

ModelResult = dict[str, Any]
"""Fitted model plus its predictions and metrics (see ``train_and_evaluate``)."""
