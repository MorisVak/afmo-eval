import streamlit as st
import pandas as pd

def ensure_state():
    ss = st.session_state
    #### general 
    if "data" not in ss:
        ss["data"] = None # full data
    if "freq" not in ss:
        ss["freq"] = None
    if "notes" not in ss:
        ss["notes"] = ""
    if "theme" not in ss:
        ss["theme"] = "dark"
    if "_active_page" not in ss:
        ss["_active_page"] = None
    #### Page Forecast
    if "active_col" not in ss:
        ss["active_col"] = None
    if "selected_cols" not in ss:
        ss["selected_cols"] = []
    if "models" not in ss:
        ss["models"] = {}
    if "forecasts" not in ss:
        ss["forecasts"] = {}       # optional cache of forecasts
    if "model_name" not in ss:
        ss["model_name"] = {}
    if "model_params" not in ss:
        ss["model_params"] = {}
    if "fc_horizon" not in ss:
        ss["fc_horizon"] = 12      # default global holdout size
    if "predictions" not in ss:
        ss["predictions"] = {}
    if "last_horizon" not in ss:
        ss["last_horizon"] = 12
    if "oos_forecasts" not in ss:
        ss["oos_forecasts"] = {}
    #### Page study
    if "study_results" not in ss:
        ss["study_results"] = None
    if "study_results_pack" not in ss:
        ss["study_results_pack"] = None
    if "plot_collector" not in ss:
        ss["plot_collector"] = None
    if "toggle_seasonal" not in ss:
        ss["toggle_seasonal"] = None

def get_current_dataframe():
    ss = st.session_state
    return ss.get("data", None)
