from __future__ import annotations
from typing import Mapping, Iterable, Tuple, Dict, Any
import matplotlib.ticker as mticker
import re

import importlib

plot_mod = importlib.import_module("afmo.plot")

def _collect_registry_name_map() -> dict[str, str]:
    """Build {token -> pretty_name} from all registries exposing objects with .name."""
    try:
        from afmo.core import registry as R
    except Exception:
        return {}
    out: dict[str, str] = {}
    registries = (
        "FEATURES",
        "FC_MODELS",
        "PREDICTORS",
        "FC_METRICS_PREDICTABILITY",
        "FC_METRICS_EFFECTIVENESS",
        "FC_METRICS_SCORES",
        "PREDICTOR_METRICS",
    )
    pairs: list[tuple[str, str]] = []
    for attr in registries:
        d = getattr(R, attr, None)
        if not d:
            continue
        for key, obj in d.items():
            pretty = getattr(obj, "name", None) or str(key)
            tokens = {str(key)}
            for cand in (getattr(obj, "__name__", None), getattr(obj, "__qualname__", None)):
                if cand:
                    tokens.add(str(cand))
            tokens |= {t.replace("_", " ") for t in list(tokens)}
            for t in tokens:
                if t:
                    pairs.append((t, pretty))
    for k, v in sorted(pairs, key=lambda kv: len(kv[0]), reverse=True):
        out[k] = v
    return out

def _make_replacer(mapping: Mapping[str, str], *, case_insensitive: bool = False, whole_words: bool = False):
    """Create a fast multi-token string replacer."""
    flags = re.IGNORECASE if case_insensitive else 0
    parts: list[str] = []
    repl_map: dict[str, str] = {}
    for k, v in mapping.items():
        if not k:
            continue
        pat = re.escape(k)
        if whole_words:
            pat = rf"\b{pat}\b"
        parts.append(pat)
        repl_map[k.lower() if case_insensitive else k] = v
    if not parts:
        return None, (lambda s: s)
    rx = re.compile("|".join(parts), flags)

    def replace(text: str) -> str:
        if not isinstance(text, str):
            return text
        if text.startswith("$") and text.endswith("$"):  # keep pure TeX intact
            return text
        def _sub(m: re.Match) -> str:
            s = m.group(0)
            key = s.lower() if case_insensitive else s
            return repl_map.get(key, s)
        return rx.sub(_sub, text)

    return rx, replace

class _ProxyFormatter(mticker.Formatter):
    """Proxy to an existing formatter that preserves all behavior and only post-processes labels."""
    def __init__(self, base: mticker.Formatter, replace_fn):
        super().__init__()
        self._base = base
        self._replace = replace_fn

    # Main formatting
    def __call__(self, x, pos=None):
        return self._replace(self._base(x, pos))

    # Forward important hooks so ScalarFormatter/LogFormatter keep working
    def set_locs(self, locs):
        if hasattr(self._base, "set_locs"):
            self._base.set_locs(locs)

    def set_axis(self, axis):
        if hasattr(self._base, "set_axis"):
            self._base.set_axis(axis)
        super().set_axis(axis)

    def create_dummy_axis(self, *a, **k):
        if hasattr(self._base, "create_dummy_axis"):
            self._base.create_dummy_axis(*a, **k)

    def get_offset(self):
        return self._base.get_offset() if hasattr(self._base, "get_offset") else ""

    # Preserve ScalarFormatter knobs if present
    def set_useOffset(self, *args, **kwargs):
        if hasattr(self._base, "set_useOffset"):
            self._base.set_useOffset(*args, **kwargs)

    def set_scientific(self, *args, **kwargs):
        if hasattr(self._base, "set_scientific"):
            self._base.set_scientific(*args, **kwargs)


def apply_pretty_names_to_fig(fig, lib: str,
                              mapping: Mapping[str, str] | None = None,
                              *, case_insensitive: bool = True,
                              whole_words: bool = False) -> None:
    """Apply replacements to a single figure (Matplotlib or Plotly), including tick labels."""
    mapping = mapping or _collect_registry_name_map()
    _, replace = _make_replacer(mapping, case_insensitive=case_insensitive, whole_words=whole_words)

    if lib == "mpl":
        # Wrap tick formatters so regenerated ticks are also pretty
        import matplotlib.text as mtext
        import matplotlib.ticker as mticker

        def _wrap_axis_formatters(axis):
            try:
                fmt = axis.get_major_formatter()
                if fmt is not None and not isinstance(fmt, _ProxyFormatter):
                    axis.set_major_formatter(_ProxyFormatter(fmt, replace))
            except Exception:
                pass
            try:
                fmtm = axis.get_minor_formatter()
                if fmtm is not None and not isinstance(fmtm, _ProxyFormatter):
                    axis.set_minor_formatter(_ProxyFormatter(fmtm, replace))
            except Exception:
                pass


        # Apply to all axes (includes colorbar axes)
        for ax in list(getattr(fig, "get_axes", lambda: [])()):
            for ax_name in ("xaxis", "yaxis", "zaxis"):
                if hasattr(ax, ax_name):
                    _wrap_axis_formatters(getattr(ax, ax_name))

        # Update any existing Text objects (titles, axis labels, tick labels, annotations)
        for txt in fig.findobj(match=mtext.Text):
            try:
                txt.set_text(replace(txt.get_text()))
            except Exception:
                pass

        # Trigger a redraw so wrapped formatters take effect
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass

    elif lib == "plotly":
        lay = fig.layout

        # Titles, legend titles, annotations, trace names, colorbar titles (as before)
        if getattr(lay.title, "text", None):
            lay.title.text = replace(lay.title.text)
        for attr in dir(lay):
            if attr.startswith(("xaxis", "yaxis")):
                ax = getattr(lay, attr, None)
                if ax and getattr(ax.title, "text", None):
                    ax.title.text = replace(ax.title.text)
        if getattr(lay, "legend", None) and getattr(lay.legend, "title", None) and getattr(lay.legend.title, "text", None):
            lay.legend.title.text = replace(lay.legend.title.text)
        if getattr(lay, "annotations", None):
            for ann in lay.annotations:
                if getattr(ann, "text", None):
                    ann.text = replace(ann.text)
        for tr in getattr(fig, "data", []):
            if getattr(tr, "name", None):
                tr.name = replace(tr.name)
        for attr in dir(lay):
            if attr.startswith("coloraxis"):
                ca = getattr(lay, attr, None)
                if ca and getattr(ca, "colorbar", None) and getattr(ca.colorbar.title, "text", None):
                    ca.colorbar.title.text = replace(ca.colorbar.title.text)
        for tr in getattr(fig, "data", []):
            cb = getattr(tr, "colorbar", None)
            if cb and getattr(cb, "title", None) and getattr(cb.title, "text", None):
                cb.title.text = replace(cb.title.text)

        # For categorical tick labels, set tickmode='array' with replaced ticktext
        def _axis_attr_from_ref(ref: str, axis: str) -> str:
            # ref is 'x','x2','y','y3', return 'xaxis','xaxis2','yaxis','yaxis3'
            suffix = "" if ref in ("x", "y") else ref[1:]
            return f"{axis}axis{suffix}"

        # Collect categorical labels per axis from traces
        cats: dict[str, set] = {}
        for tr in getattr(fig, "data", []):
            xref = getattr(tr, "xaxis", "x")
            yref = getattr(tr, "yaxis", "y")
            xattr = _axis_attr_from_ref(xref, "x")
            yattr = _axis_attr_from_ref(yref, "y")

            xs = getattr(tr, "x", None)
            ys = getattr(tr, "y", None)

            if isinstance(xs, (list, tuple)) and any(isinstance(v, str) for v in xs):
                cats.setdefault(xattr, set()).update([v for v in xs if isinstance(v, str)])
            if isinstance(ys, (list, tuple)) and any(isinstance(v, str) for v in ys):
                cats.setdefault(yattr, set()).update([v for v in ys if isinstance(v, str)])

        # Apply ticktext/tickvals only for axes that have string categories
        for attr, labels in cats.items():
            if not labels:
                continue
            ax = getattr(lay, attr, None)
            if not ax:
                continue
            vals = list(labels)
            texts = [replace(v) for v in vals]
            ax.tickmode = "array"
            ax.tickvals = vals
            ax.ticktext = texts

def apply_pretty_names_to_collector(col, mapping: Mapping[str, str] | None = None,
                                    *, case_insensitive: bool = True,
                                    whole_words: bool = False) -> None:
    """Apply replacements to all PlotArtifacts inside a PlotCollector, including tick labels."""
    mapping = mapping or _collect_registry_name_map()
    _, replace = _make_replacer(mapping, case_insensitive=case_insensitive, whole_words=whole_words)
    for art in getattr(col, "all", lambda: [])():
        if hasattr(art, "title") and isinstance(art.title, str):
            art.title = replace(art.title)
        try:
            apply_pretty_names_to_fig(art.fig, art.lib, mapping=mapping,
                                      case_insensitive=case_insensitive,
                                      whole_words=whole_words)
        except Exception:
            pass
