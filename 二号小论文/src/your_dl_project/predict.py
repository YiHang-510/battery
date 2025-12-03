# src/your_dl_project/predict.py
"""
Utility functions to run inference with a trained model.

The code dynamically handles common deep-learning frameworks (PyTorch /
TensorFlow) when they are installed, but it also works with the lightweight
`SimpleModel` provided in ``model.py``.  Every function focuses on a single
responsibility to keep the inference pipeline easy to follow:

1. Load configuration and the serialized model weights.
2. Preprocess arbitrary user input into a numpy array.
3. Run single-sample or batch prediction.
4. Provide a convenient CLI entry-point for scripting.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

from .model import create_model

# Optional imports: keep them local to avoid forcing dependencies.
try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch is optional
    torch = None

try:
    from tensorflow import keras  # type: ignore
except ImportError:  # pragma: no cover - keras is optional
    keras = None


def _ensure_exists(path: str) -> None:
    """Raise a helpful error if a required path is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def load_model_for_inference(model_path: str, config: dict) -> Any:
    """
    Instantiate a model and load weights for inference.

    The helper attempts to infer the proper loading routine based on the file
    extension and available frameworks.  For example:

    - ``*.pth`` / ``*.pt`` -> PyTorch ``state_dict``
    - ``*.h5`` / ``*.keras`` -> TensorFlow/Keras SavedModel
    - ``*.npz`` -> numpy checkpoint for the lightweight SimpleModel

    Parameters
    ----------
    model_path:
        Path to the serialized model weights.
    config:
        Model-related configuration read from the YAML file.
    """

    _ensure_exists(model_path)

    file_ext = os.path.splitext(model_path)[1].lower()
    framework = (config.get("framework") or "pytorch").lower()

    if file_ext in {".h5", ".keras", ".pb"}:
        if keras is None:
            raise RuntimeError(
                "TensorFlow/Keras is not installed but a Keras checkpoint was "
                "requested."
            )
        model = keras.models.load_model(model_path)
        print(f"Loaded TensorFlow/Keras model from {model_path}")
        return model

    # Default: rely on the project's create_model helper.
    model = create_model(config)

    if file_ext in {".pth", ".pt"}:
        if torch is None:
            raise RuntimeError(
                "PyTorch is not installed but a .pth checkpoint was requested."
            )
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded PyTorch model from {model_path}")
        return model

    if file_ext == ".npz":
        weights = np.load(model_path, allow_pickle=True)
        # SimpleModel exposes a convenience hook through setattr to keep things
        # framework agnostic.  Consumers can implement `load_weights` or any
        # custom attribute the training loop writes into the checkpoint.
        if hasattr(model, "load_weights"):
            model.load_weights(weights)
        else:
            setattr(model, "weights", weights)
        print(f"Loaded numpy weights from {model_path}")
        return model

    print(
        "Unknown checkpoint format. Returning an uninitialized model "
        "instance. Predictions may rely on randomly initialized weights."
    )
    return model


def preprocess_input(input_data: Any) -> np.ndarray:
    """
    Convert raw user input into a 2D numpy array.

    Accepts scalars, sequences, numpy arrays, or nested iterables.  Missing
    values are converted to ``np.nan`` for downstream handling.
    """

    if isinstance(input_data, np.ndarray):
        array = input_data.copy()
    elif isinstance(input_data, (Sequence, Iterable)):
        array = np.asarray(list(input_data), dtype=np.float32)
    else:
        array = np.asarray([input_data], dtype=np.float32)

    if array.ndim == 1:
        array = array.reshape(1, -1)

    return array.astype(np.float32)


def _run_forward(model: Any, features: np.ndarray) -> np.ndarray:
    """Execute the forward pass using the best available API."""

    # TensorFlow/Keras model
    if keras is not None and isinstance(model, keras.Model):
        output = model.predict(features, verbose=0)
        return np.asarray(output)

    # PyTorch model
    if torch is not None and isinstance(model, torch.nn.Module):
        with torch.no_grad():
            tensor = torch.from_numpy(features).float()
            output = model(tensor)
            return output.detach().cpu().numpy()

    # Generic numpy-based model (SimpleModel or custom class)
    if hasattr(model, "predict"):
        output = model.predict(features)
        return np.asarray(output)

    if hasattr(model, "forward"):
        output = model.forward(features)
        return np.asarray(output)

    raise AttributeError(
        "The provided model does not implement `predict` or `forward`. "
        "Please update your model class to expose one of these methods."
    )


def _postprocess_logits(logits: np.ndarray) -> Any:
    """Convert raw logits into human-readable predictions."""

    if logits.ndim == 1:
        return logits.argmax(axis=0)

    return logits.argmax(axis=1)


def predict_single(model: Any, input_data: Any) -> Any:
    """Run inference on a single sample."""

    processed_input = preprocess_input(input_data)
    logits = _run_forward(model, processed_input)
    return _postprocess_logits(logits)[0]


def predict_batch(model: Any, input_batch: Any) -> Any:
    """
    Run inference over a batch of samples.

    ``input_batch`` can already be a numpy array, nested list, or generator.
    """

    processed_batch = preprocess_input(input_batch)
    logits = _run_forward(model, processed_batch)
    return _postprocess_logits(logits)


def predict(model_path: str, config_path: str, input_data: Any) -> Any:
    """
    Full prediction pipeline that glues configuration, model, and data.
    """

    _ensure_exists(config_path)

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    model = load_model_for_inference(model_path, config["model"])

    if isinstance(input_data, (list, tuple, np.ndarray)) and np.asarray(
        input_data
    ).ndim > 1:
        return predict_batch(model, input_data)

    if isinstance(input_data, np.ndarray) and input_data.ndim > 1:
        return predict_batch(model, input_data)

    return predict_single(model, input_data)


def _load_input_from_path(input_path: str) -> np.ndarray:
    """Helper used by the CLI to load data from disk."""

    _ensure_exists(input_path)
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".npy":
        return np.load(input_path)
    if ext == ".npz":
        return np.load(input_path)["arr_0"]
    if ext in {".csv", ".txt"}:
        return np.loadtxt(input_path, delimiter=",")

    raise ValueError(f"Unsupported file extension for inference input: {ext}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(description="Run model inference.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the serialized model (defaults to saved_models/best_model.pth).",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to the YAML config (defaults to configs/base_config.yaml).",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="File containing the input sample(s). Supports .npy, .npz, .csv, .txt.",
    )
    parser.add_argument(
        "--input-values",
        type=float,
        nargs="+",
        help="Raw feature values for a single sample, e.g. --input-values 0.1 0.2 0.3",
    )
    return parser


def _resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve default paths relative to the project root."""

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_model = os.path.join(project_root, "saved_models", "best_model.pth")
    default_config = os.path.join(project_root, "configs", "base_config.yaml")

    model_path = args.model_path or default_model
    config_path = args.config_path or default_config

    return model_path, config_path


def _cli_entry() -> None:
    """Entry point for running the script directly."""

    parser = _build_arg_parser()
    args = parser.parse_args()

    model_path, config_path = _resolve_paths(args)

    if args.input_file:
        cli_input = _load_input_from_path(args.input_file)
    elif args.input_values:
        cli_input = np.asarray(args.input_values, dtype=np.float32)
    else:
        raise ValueError(
            "No input provided. Supply --input-file or --input-values."
        )

    predictions = predict(model_path, config_path, cli_input)
    print("Predictions:", predictions)


if __name__ == "__main__":
    _cli_entry()
