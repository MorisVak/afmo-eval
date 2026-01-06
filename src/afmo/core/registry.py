"""Tiny plugin registries used by the GUI and the core library."""
from __future__ import annotations
from typing import Dict, Any, Callable

# ---- Core registries -----------------------------------------------------
FEATURES: Dict[str, Callable[..., Any]] = {}
FC_MODELS: Dict[str, Any] = {}
FC_METRICS_PREDICTABILITY: Dict[str, Callable[..., Any]] = {}
FC_METRICS_EFFECTIVENESS: Dict[str, Callable[..., Any]] = {}
FC_METRICS_SCORES: Dict[str, Callable[..., Any]] = {}
PREDICTORS: Dict[str, Callable[..., Any]] = {}
PREDICTOR_METRICS: Dict[str, Callable[..., Any]] = {}

# ---- Decorators ----------------------------------------------------------
def register_feature(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to register a **time‑series feature** function.

    A feature function accepts a ``pd.Series`` and returns a scalar or
    small JSON‑serialisable object.  Features are *not* shown by the GUI
    unless explicitly used, so adding new ones is backwards‑compatible.
    """
    FEATURES[getattr(func, "name", func.__name__)] = func
    return func

def register_fc_model(func: Callable) -> Callable:
    """Decorator to register a forecasting model class.

    The GUI discovers models from this registry.  A model must expose a
    ``fit(y)`` method and a ``forecast(steps)`` method returning a
    pandas DataFrame; we keep the contract identical to the previous codebase.
    """
    FC_MODELS[getattr(func, "name", func.__name__)] = func
    return func

def register_fc_metric_predictability(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to register a **forecast accuracy metric**.

    A metric is a function ``metric(y_true, y_pred) -> float`` and should
    be **side‑effect free**.  It will only be used when called explicitly
    by the analysis layer or CLI.
    """
    FC_METRICS_PREDICTABILITY[getattr(func, "name", func.__name__)] = func
    return func

def register_fc_metric_effectiveness(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to register a **forecast accuracy metric**.

    A metric is a function ``metric(y_true, y_pred) -> float`` and should
    be **side‑effect free**.  It will only be used when called explicitly
    by the analysis layer or CLI.
    """
    FC_METRICS_EFFECTIVENESS[getattr(func, "name", func.__name__)] = func
    return func

def register_fc_metric_score(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to register a qualitive **forecast accuracy metric**.

    A quality metric is a function ``metric(y_true, y_pred) -> float`` and should
    be **side‑effect free**.  It will only be used when called explicitly
    by the analysis layer or CLI.
    """
    FC_METRICS_SCORES[getattr(func, "name", func.__name__)] = func
    return func

def register_predictor(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to register a cross‑validation/evaluator class."""
    PREDICTORS[getattr(func, "name", func.__name__)] = func
    return func

def register_predictor_metric(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to register a pred evaluation metric class."""
    PREDICTOR_METRICS[getattr(func, "name", func.__name__)] = func
    return func