from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

Split = Literal["train", "val", "test"]
Lib = Literal["mpl", "plotly"]

@dataclass
class PlotArtifact:
    id: str
    title: str
    kind: str
    split: Split
    fig: Any
    lib: Lib
    meta: dict = field(default_factory=dict)

    def to_png(self) -> bytes:
        import io
        if self.lib == "mpl":
            import matplotlib.pyplot as plt  # noqa: F401
            buf = io.BytesIO()
            self.fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            return buf.getvalue()
        else:
            try:
                return self.fig.to_image(format="png", scale=2)
            except Exception as e:
                raise RuntimeError("Plotly export failed.", self.fig) from e

    def to_pdf(self) -> bytes:
        import io
        if self.lib == "mpl":
            buf = io.BytesIO()
            self.fig.savefig(buf, format="pdf", bbox_inches="tight")
            return buf.getvalue()
        else:
            try:
                return self.fig.to_image(format="pdf")
            except Exception as e:
                raise RuntimeError("Plotly export failed.", self.fig) from e


class PlotCollector:
    def __init__(self) -> None:
        self._items: list[PlotArtifact] = []

    def add(self, art: PlotArtifact) -> None:
        self._items.append(art)

    def extend(self, arts: Iterable[PlotArtifact]) -> None:
        self._items.extend(list(arts))

    def all(self) -> list[PlotArtifact]:
        return list(self._items)

    def by_split(self) -> dict[str, list[PlotArtifact]]:
        out: dict[str, list[PlotArtifact]] = {"train": [], "val": [], "test": []}
        for a in self._items:
            out.setdefault(a.split, []).append(a)
        return out

    def as_zip_bytes(self, formats: tuple[str, ...] = ("png", "pdf")) -> bytes:
        import zipfile, io, re, unicodedata
        def slug(s: str) -> str:
            s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
            s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")
            return s or "plot"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            if not self._items:
                zf.writestr("README.txt","No plots were collected.")
            for art in self._items:
                base = slug(f"{art.split}_{art.kind}_{art.title}")
                if "png" in formats:
                    zf.writestr(f"{base}.png", art.to_png())
                if "pdf" in formats:
                    zf.writestr(f"{base}.pdf", art.to_pdf())
        return buf.getvalue()