"""Utilities for AFMo (app/AFMo.py)"""
import _bootstrap  # ensure src/ is on sys.path if not installed
import streamlit as st

st.set_page_config(
    page_title="Main Page",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from datetime import datetime
from utils.state_utils import ensure_state
from utils.theme_utils import theme_header

ensure_state()
theme_header("Accuracy of Forecasting Models (AFMo)", key="hdr_app")

with st.sidebar:
    st.title("⚙️ Settings")
    st.caption("Global settings & tips")
    st.text_area("Notes (global)", key="notes", height=120)
    st.divider()
    st.markdown("")
    st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.title("📈 Accuracy of Forecasting Models (AFMo)")
st.write(
    "A web-based UI for time-series forecasting and evaluation** "
    "(first column = datetime, others = time series), **Forecasting**, **Evaluation**, and Evaluation Analysis."
)