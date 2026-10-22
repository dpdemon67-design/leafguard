"""LeafGuard – Streamlit demo.   Run:  streamlit run app.py
Tab 1: offline leaf-disease screening (ONNX, on-device) with explanation + Gujarati/Hindi guidance.
Tab 2: separate localised pest-risk forecast (weather + crop), explainable, works offline via manual/cached data."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

import pest_forecast as pf
from guidance import (DISEASE_ADVICE, LANGS, LEVELS, PEST_ADVICE, UI, disease_category, pretty_class)

MODEL_DIR = Path(__file__).parent / "models"
st.set_page_config(page_title="LeafGuard", page_icon="🌿", layout="centered")

lang = st.sidebar.radio("Language / ભાષા / भाषा", list(LANGS), format_func=lambda k: LANGS[k])
T = lambda key: UI[key][lang]


@st.cache_resource
def _load_classifier():
    from disease_model import DiseaseClassifier
    return DiseaseClassifier()


def get_classifier():
    if not (MODEL_DIR / "leaf_model.onnx").exists() or not (MODEL_DIR / "labels.json").exists():
        return None
    return _load_classifier()


@st.cache_resource
def _warm_pest_model():
    return pf.load_model()


clf = get_classifier()
if clf is not None:
    clf.conf_threshold = st.sidebar.slider("Confidence threshold", 0.40, 0.95, 0.70, 0.05)
    info = json.loads((MODEL_DIR / "labels.json").read_text())
    st.sidebar.caption(f"Model: MobileNetV3-Small ONNX · {info.get('size_mb', '?')} MB · "
                       f"~{info.get('cpu_ms_per_image', '?')} ms/img on CPU · "
                       f"val acc {info.get('val_accuracy', '?')}")
    st.sidebar.caption("Leaf inference runs fully on this device – no internet needed.")

st.title("🌿 " + T("title"))
tab_leaf, tab_pest = st.tabs([T("tab_leaf"), T("tab_pest")])

# ------------------------------------------------------------------ TAB 1: leaf disease
with tab_leaf:
    if clf is None:
        st.error(T("no_model"))
    else:
        up = st.file_uploader(T("upload"), type=["jpg", "jpeg", "png"])
        with st.expander(T("camera")):
            cam = st.camera_input(T("camera"), label_visibility="collapsed")
        src = up or cam
        if src is not None:
            img = Image.open(src)
            res = clf.predict(img)
            name, conf = res["top"][0]
            c1, c2 = st.columns(2)
            c1.image(res["crop"], caption="Analysed region", use_container_width=True)
            with st.spinner("..."):
                heat = clf.explain(res["x"], res["class_index"])
            from disease_model import overlay
            c2.image(overlay(res["crop"], heat), caption=T("why"), use_container_width=True)

            if res["uncertain"]:
                st.warning(f"**{T('uncertain_title')}**\n\n{T('uncertain_msg')}")
            else:
                st.success(f"**{T('likely')}:** {pretty_class(name)}")
            st.metric(T("confidence"), f"{conf:.0%}")
            st.progress(min(max(conf, 0.0), 1.0))

            st.markdown(f"**{T('other_options')}**")
            for n_, p_ in res["top"]:
                st.write(f"- {pretty_class(n_)} — {p_:.0%}")

            if not res["uncertain"]:
                st.markdown(f"### {T('next_steps')}")
                for step in DISEASE_ADVICE[disease_category(name)][lang]:
                    st.write("• " + step)
            st.info(T("expert"))

# ------------------------------------------------------------------ TAB 2: pest risk
with tab_pest:
    _warm_pest_model()
    c1, c2 = st.columns(2)
    district = c1.selectbox(T("district"), list(pf.DISTRICTS))
    crop = c2.selectbox(T("crop"), list(pf.CROPS))
    mode = st.radio(T("mode"), ["live", "manual"], horizontal=True,
                    format_func=lambda m: T("mode_live") if m == "live" else T("mode_manual"))
    manual = None
    if mode == "manual":
        m1, m2, m3 = st.columns(3)
        mt = m1.number_input("Avg temp (°C)", 0.0, 50.0, 30.0, 0.5)
        mrh = m2.number_input("Avg humidity (%)", 0.0, 100.0, 70.0, 1.0)
        mrain = m3.number_input("Rain last 7 d (mm)", 0.0, 500.0, 20.0, 1.0)
        m4, m5 = st.columns(2)
        mfrh = m4.number_input("Forecast humidity (%)", 0.0, 100.0, 70.0, 1.0)
        mfrain = m5.number_input("Forecast rain 3 d (mm)", 0.0, 300.0, 10.0, 1.0)
        manual = (mt, mrh, mrain, mfrh, mfrain)

    if st.button(T("forecast_btn"), type="primary"):
        lat, lon = pf.DISTRICTS[district]
        try:
            if mode == "live":
                daily, meta = pf.fetch_weather(lat, lon)
            else:
                daily = pf.manual_daily(*manual)
                meta = {"source": "manual entry (offline)", "lat": lat, "lon": lon,
                        "fetched_at": "n/a", "first_day": str(daily["date"].iloc[0]),
                        "last_day": str(daily["date"].iloc[-1])}
            out = pf.predict_risk(crop, daily, meta)
        except Exception as e:
            st.error(f"{e}  →  switch to '{UI['mode_manual']['en']}'.")
            st.stop()

        lvl = out["level"]
        color = {"Low": "#2e7d32", "Moderate": "#ef6c00", "High": "#c62828"}[lvl]
        st.markdown(f"**{T('risk_level')}** ({district}, {crop})")
        st.markdown(f"<div style='background:{color};color:white;padding:14px;border-radius:10px;"
                    f"font-size:28px;font-weight:700;text-align:center'>{LEVELS[lvl][lang]}</div>",
                    unsafe_allow_html=True)
        st.write("")
        st.write(PEST_ADVICE[lvl][lang])
        if out["conservative"]:
            st.warning(T("conservative"))
        st.caption(f"Typical pests/diseases to watch: {out['typical_pests']}")

        probs = pd.DataFrame({"Probability": [out["probs"][l] for l in pf.LEVEL_ORDER]},
                             index=[LEVELS[l][lang] for l in pf.LEVEL_ORDER])
        st.bar_chart(probs)

        st.markdown(f"**{T('factors')}**")
        st.dataframe(pd.DataFrame(out["factors"]).rename(columns={
            "feature": "Factor", "value": "Value", "effect": "Effect", "strength": "Strength"}),
            hide_index=True, use_container_width=True)

        st.markdown(f"**{T('provenance')}**")
        m = out["meta"]
        st.caption(f"Location: {district} ({m.get('lat')}, {m.get('lon')}) · Weather: {m.get('source')} · "
                   f"Fetched: {m.get('fetched_at')} · Window: {out['window']}")
        st.caption(f"Model trained on: {out['data_source']} · accuracy {out['model_accuracy']:.0%} "
                   f"vs. humidity-only baseline {out['baseline_accuracy']:.0%}")
        st.line_chart(daily.set_index("date")[["tmean", "rh"]])
        st.bar_chart(daily.set_index("date")[["rain"]])
        st.info(T("expert"))
