# utils/persist_utils.py
from __future__ import annotations
from typing import Any, Dict
import math
import numpy as np
import pandas as pd
from datetime import datetime, date, time, timedelta
from collections import defaultdict, OrderedDict
import pickle  # for serializing arbitrary Python objects (e.g. sklearn pipelines)
import base64  # for encoding pickled bytes into JSON-friendly strings

_TAG = "__type__"
_VAL = "value"


def _is_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (str, bool, int, float))


def _finite_number(x: Any) -> Any:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if math.isfinite(xf):
            return xf
        kind = "nan" if math.isnan(xf) else ("inf" if xf > 0 else "-inf")
        return {_TAG: "float", _VAL: kind}
    if isinstance(x, np.generic):
        return _finite_number(x.item())
    return x


def _factory_name(f):
    # recognize common default factories so defaultdicts can be restored
    if f is None:
        return None
    if f is list:
        return "list"
    if f is dict:
        return "dict"
    if f is set:
        return "set"
    if f is int:
        return "int"
    if f is float:
        return "float"
    if f is str:
        return "str"
    return None


def _factory_from_name(name):
    return {
        None: None,
        "list": list,
        "dict": dict,
        "set": set,
        "int": int,
        "float": float,
        "str": str,
    }.get(name, None)


def _pack(obj: Any) -> Any:
    if _is_primitive(obj):
        return _finite_number(obj)

    if isinstance(obj, datetime):
        return {_TAG: "datetime", _VAL: obj.isoformat(timespec="microseconds")}
    if isinstance(obj, date):
        return {_TAG: "date", _VAL: obj.isoformat()}
    if isinstance(obj, time):
        return {_TAG: "time", _VAL: obj.isoformat()}
    if isinstance(obj, timedelta):
        return {_TAG: "timedelta", _VAL: obj.total_seconds()}

    if isinstance(obj, np.generic):
        return _finite_number(obj.item())
    if isinstance(obj, np.ndarray):
        flat = obj.ravel().tolist()
        flat = [_finite_number(x) for x in flat]
        return {
            _TAG: "np.ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            _VAL: flat,
        }

    if isinstance(obj, pd.DataFrame):
        split = obj.to_dict(orient="split")
        split["data"] = _pack(split["data"])
        split["index"] = _pack(split["index"])
        split["columns"] = _pack(split["columns"])
        dtypes = {c: str(dt) for c, dt in obj.dtypes.items()}
        index_name = obj.index.name
        columns_name = obj.columns.name if hasattr(obj.columns, "name") else None
        return {
            _TAG: "pd.DataFrame",
            _VAL: split,
            "dtypes": dtypes,
            "index_name": index_name,
            "columns_name": columns_name,
        }

    if isinstance(obj, pd.Series):
        return {
            _TAG: "pd.Series",
            "dtype": str(obj.dtype),
            "name": obj.name,
            "index": _pack(obj.index.tolist()),
            "values": _pack(obj.tolist()),
        }

    if isinstance(obj, pd.Index):
        return {_TAG: "pd.Index", _VAL: _pack(obj.tolist()), "name": obj.name}

    # OrderedDict: preserve insertion order explicitly
    if isinstance(obj, OrderedDict):
        items = [[_pack(k), _pack(v)] for k, v in obj.items()]
        return {_TAG: "collections.OrderedDict", _VAL: items}

    # defaultdict with known factory
    if isinstance(obj, defaultdict):
        factory = _factory_name(obj.default_factory)
        items = [[_pack(k), _pack(v)] for k, v in obj.items()]
        return {
            _TAG: "collections.defaultdict",
            "factory": factory,
            _VAL: items,
        }

    if isinstance(obj, set):
        return {_TAG: "set", _VAL: [_pack(x) for x in obj]}
    if isinstance(obj, tuple):
        return {_TAG: "tuple", _VAL: [_pack(x) for x in obj]}
    if isinstance(obj, list):
        return [_pack(x) for x in obj]

    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj.keys()):
            return {str(k): _pack(v) for k, v in obj.items()}
        items = [[_pack(k), _pack(v)] for k, v in obj.items()]
        return {_TAG: "dict", "string_keys": False, _VAL: items}

    # fallback to pickle for arbitrary Python objects (e.g. sklearn PCA, pipelines)
    try:
        data = pickle.dumps(obj)
        b64 = base64.b64encode(data).decode("ascii")
        return {
            _TAG: "pickle",
            "cls_module": obj.__class__.__module__,
            "cls_name": obj.__class__.__name__,
            _VAL: b64,
        }
    except Exception:
        # last resort: store string representation
        return {_TAG: "str", _VAL: str(obj)}


def pack_study_results(results: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(results, dict) and _TAG in results and _VAL in results:
        return results
    if not isinstance(results, dict):
        return {_TAG: "root", _VAL: _pack(results)}
    if not all(isinstance(k, str) for k in results.keys()):
        return _pack(results)
    return {k: _pack(v) for k, v in results.items()}


def _unpack(obj: Any) -> Any:
    if _is_primitive(obj):
        return obj

    if isinstance(obj, dict) and _TAG in obj:
        t = obj.get(_TAG)
        v = obj.get(_VAL)
        if t == "float":
            if v == "nan":
                return float("nan")
            if v == "inf":
                return float("inf")
            if v == "-inf":
                return float("-inf")
            return float("nan")
        if t == "datetime":
            return datetime.fromisoformat(v)
        if t == "date":
            return date.fromisoformat(v)
        if t == "time":
            return time.fromisoformat(v)
        if t == "timedelta":
            return timedelta(seconds=float(v))
        if t == "np.ndarray":
            flat = [_unpack(x) for x in v]
            arr = np.array(flat, dtype=np.dtype(obj.get("dtype")))
            shape = tuple(obj.get("shape", (len(arr),)))
            try:
                return arr.reshape(shape)
            except Exception:
                return arr
        if t == "pd.DataFrame":
            split = v
            split = {
                k: _unpack(val) if k in ("data", "index", "columns") else val
                for k, val in split.items()
            }
            df = pd.DataFrame(**split)
            dtypes = obj.get("dtypes", {})
            for c, dt in dtypes.items():
                try:
                    df[c] = df[c].astype(dt)
                except Exception:
                    pass
            if obj.get("index_name") is not None:
                df.index.name = obj["index_name"]
            if obj.get("columns_name") is not None and hasattr(df.columns, "name"):
                df.columns.name = obj["columns_name"]
            return df
        if t == "pd.Series":
            idx = _unpack(obj.get("index", []))
            vals = _unpack(obj.get("values", []))
            s = pd.Series(vals, index=idx, name=obj.get("name"))
            dtype = obj.get("dtype")
            if dtype:
                try:
                    s = s.astype(dtype)
                except Exception:
                    pass
            return s
        if t == "pd.Index":
            return pd.Index(_unpack(v), name=obj.get("name"))
        if t == "collections.OrderedDict":
            return OrderedDict((_unpack(k), _unpack(val)) for k, val in v)
        if t == "collections.defaultdict":
            factory = _factory_from_name(obj.get("factory"))
            dd = defaultdict(factory) if factory else defaultdict()
            for k, val in v:
                dd[_unpack(k)] = _unpack(val)
            return dd
        if t == "set":
            return set(_unpack(x) for x in v)
        if t == "tuple":
            return tuple(_unpack(x) for x in v)
        if t == "dict":
            return {_unpack(k): _unpack(val) for k, val in v}
        if t == "pickle":
            data = base64.b64decode(v.encode("ascii"))
            return pickle.loads(data)
        if t == "str":
            return str(v)
        if t == "root":
            return _unpack(v)

    if isinstance(obj, list):
        return [_unpack(x) for x in obj]

    if isinstance(obj, dict):
        return {k: _unpack(v) for k, v in obj.items()}

    return obj


def unpack_study_results(packed: Dict[str, Any]) -> Dict[str, Any]:
    return _unpack(packed)
