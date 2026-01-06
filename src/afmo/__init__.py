"""Utilities for AFMo (src/afmo/__init__.py)."""
import logging

# AFMO public API
__version__ = "1.1.0"


def enable_debug_logging(level=logging.DEBUG):
    """
    Enable debug logging for AFMO modules.
    
    Usage:
        import afmo
        afmo.enable_debug_logging()  # Shows all debug messages
        afmo.enable_debug_logging(logging.WARNING)  # Only warnings and errors
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    # Set level for afmo modules specifically
    for mod_name in ['afmo.fc_models', 'afmo.meta_trainer', 'afmo.features', 'afmo.predictors']:
        logging.getLogger(mod_name).setLevel(level)


# IO & session round‑trip
from .io import (
    load_dataframe_multi,
    to_csv_bytes,
    export_session,
    import_session,
)
# Backward-compatible aliases
session_to_json_bytes = export_session
session_from_json_bytes = import_session

# Classic stats (kept for full backwards compatibility)
from .features import *

# New extensibility points (opt‑in in the GUI/analysis)
from .core.registry import FC_MODELS, PREDICTORS, FEATURES, FC_METRICS_SCORES, FC_METRICS_PREDICTABILITY, FC_METRICS_EFFECTIVENESS, register_feature, register_fc_metric_score, register_fc_metric_predictability, register_fc_metric_effectiveness
from .features import compute_features, coefficient_of_variation, get_feature_names_by_tag
from .fc_metrics_predictability import compute_metrics, smape