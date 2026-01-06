"""Forecast & Accuracy page (wrapper) — delegates to subpages."""
import _bootstrap  # ensure app/ and src/ are on sys.path
import streamlit as st

from pages._subpages.forecast_section import run_forecast_section
from pages._subpages.evaluation_section import run_evaluation_section

st.session_state["_active_page"] = "Forecast"

# Render the two sections
run_forecast_section()
run_evaluation_section()
