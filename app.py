import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Risiko Maternal",
    page_icon="🏥",
    layout="wide",
)

# ── Minimal CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #F0F4F8; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(__file__), "best_model_final.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Clinical Scoring ──────────────────────────────────────────────────────────
def calculate_score(age, systolic, diastolic, heart_rate, bmi, body_temp, prev_comp):
    MAP = (systolic + 2 * diastolic) / 3
    pulse_pressure = systolic - diastolic
    fever = body_temp > 100.4

    rows = []

    def add(factor, value, unit, score):
        rows.append({"Faktor Risiko": factor, "Kondisi": f"{value} {unit}".strip(), "Skor": score})

    add("Riwayat Komplikasi", "Ya" if prev_comp else "Tidak", "", 5 if prev_comp else 0)

    bmi_score = 4 if bmi >= 30 else (2 if bmi >= 25 else 0)
    add("BMI", f"{bmi:.1f}", "kg/m²", bmi_score)

    sys_score = 5 if systolic >= 160 else (3 if systolic >= 140 else (1 if systolic >= 130 else 0))
    add("Sistolik", systolic, "mmHg", sys_score)

    dia_score = 5 if diastolic >= 110 else (3 if diastolic >= 90 else (1 if diastolic >= 85 else 0))
    add("Diastolik", diastolic, "mmHg", dia_score)

    map_score = 4 if MAP >= 107 else (2 if MAP >= 100 else 0)
    add("MAP", f"{MAP:.1f}", "mmHg", map_score)

    hr_score = 3 if heart_rate > 110 else (2 if heart_rate > 100 else 0)
    add("Detak Jantung", heart_rate, "bpm", hr_score)

    age_score = 2 if (age < 18 or age > 35) else 0
    add("Usia", age, "tahun", age_score)

    pp_score = 2 if pulse_pressure >= 60 else (1 if pulse_pressure >= 50 else 0)
    add("Pulse Pressure", pulse_pressure, "mmHg", pp_score)

    add("Demam", "Ya" if fever else "Tidak", "", 3 if fever else 0)

    total = sum(r["Skor"] for r in rows)
    return rows, total, MAP, pulse_pressure

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏥 Sistem Prediksi Risiko Maternal")
st.caption("Maternal Risk Assessment & Decision Support System · v2.0")
st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT: FORM | RESULT
# ══════════════════════════════════════════════════════════════════════════════
col_form, col_result = st.columns([1, 1], gap="large")

# ── FORM ──────────────────────────────────────────────────────────────────────
with col_form:
    st.subheader("📋 Data Pasien Ibu Hamil")

    st.markdown("**🫀 Tanda Vital**")

    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Usia (tahun)", min_value=10, max_value=60, value=25, step=1)
    with c2:
        heart_rate = st.number_input("Detak Jantung (bpm)", min_value=40, max_value=200, value=75, step=1)

    c3, c4 = st.columns(2)
    with c3:
        systolic = st.number_input("Sistolik (mmHg)", min_value=50, max_value=250, value=110, step=1)
    with c4:
        diastolic = st.number_input("Diastolik (mmHg)", min_value=30, max_value=180, value=70, step=1)

    c5, c6 = st.columns(2)
    with c5:
        body_temp = st.number_input("Suhu Tubuh (°F)", min_value=95.0, max_value=108.0, value=98.6, step=0.1, format="%.1f")
    with c6:
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=22.5, step=0.1, format="%.1f")

    st.markdown("**📁 Riwayat Klinis**")
    prev_comp = st.selectbox("Riwayat Komplikasi Sebelumnya", ["Tidak", "Ya"]) == "Ya"

    st.warning("⚠️ **PERHATIAN:** Sistem ini adalah alat bantu skrining awal. Bukan pengganti diagnosis dokter.")

    predict_btn = st.button("🔍 Prediksi Risiko", use_container_width=True, type="primary")

# ── RESULT ────────────────────────────────────────────────────────────────────
with col_result:
    if predict_btn:
        if diastolic >= systolic:
            st.error("Tekanan diastolik tidak boleh lebih besar atau sama dengan sistolik.")
            st.stop()

        rows, total_score, MAP, pulse_pressure = calculate_score(
            age, systolic, diastolic, heart_rate, bmi, body_temp, prev_comp
        )

        # ML Prediction
        ml_label = None
        ml_conf = None
        if model:
            try:
                X = np.array([[age, systolic, diastolic, MAP, heart_rate, bmi, body_temp, int(prev_comp)]])
                ml_label = model.predict(X)[0]
                if hasattr(model, "predict_proba"):
                    ml_conf = max(model.predict_proba(X)[0]) * 100
            except Exception:
                try:
                    X = np.array([[age, systolic, diastolic, MAP, heart_rate, bmi, body_temp]])
                    ml_label = model.predict(X)[0]
                    if hasattr(model, "predict_proba"):
                        ml_conf = max(model.predict_proba(X)[0]) * 100
                except Exception:
                    ml_label = None

        # Final risk
        ml_high = str(ml_label).lower() in ["high risk", "1", "high", "2"] if ml_label is not None else False
        is_high = total_score >= 7 or ml_high

        # ── Risk Banner ────────────────────────────────────────────────────────
        st.subheader("📊 Hasil Prediksi")

        if is_high:
            st.error("## ⚠️ HIGH RISK — Risiko Tinggi Terdeteksi")
            st.markdown("Pasien memerlukan **pemantauan ketat** dan evaluasi segera oleh tenaga medis.")
        else:
            st.success("## ✅ LOW RISK — Risiko Rendah")
            st.markdown("Kondisi pasien dalam batas aman. Tetap lanjutkan **pemantauan rutin antenatal**.")

        # ── Score Metrics ──────────────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Total Skor Klinis",
            f"{total_score} / 25",
            delta="High Risk" if is_high else "Low Risk",
            delta_color="inverse" if is_high else "normal",
        )
        m2.metric("MAP", f"{MAP:.1f} mmHg", help="Mean Arterial Pressure = (Sistolik + 2×Diastolik) / 3")
        m3.metric("Pulse Pressure", f"{pulse_pressure} mmHg", help="Sistolik − Diastolik")

        if ml_label is not None:
            conf_str = f" (Confidence: {ml_conf:.1f}%)" if ml_conf else ""
            st.info(f"🤖 **ML Model:** Prediksi → `{ml_label}`{conf_str}")

        st.divider()

        # ── Scoring Table ──────────────────────────────────────────────────────
        st.markdown("**📋 Clinical Scoring Breakdown**")

        df = pd.DataFrame(rows)

        def highlight_score(val):
            if isinstance(val, int) and val > 0:
                return "background-color: #FEE2E2; color: #DC2626; font-weight: bold"
            return ""

        styled = df.style.map(highlight_score, subset=["Skor"])
        st.dataframe(styled, use_container_width=True, hide_index=True)
        st.caption("Threshold: Skor ≥ 7 → High Risk · Skor < 7 → Low Risk")

        st.divider()

        # ── Recommendations ────────────────────────────────────────────────────
        st.markdown("**📌 Rekomendasi Tindakan Klinis**")

        if is_high:
            st.markdown("""
- 🔴 Rujuk segera ke dokter spesialis obstetri dan ginekologi
- 🔴 Pantau tekanan darah setiap 15–30 menit
- 🟡 Pertimbangkan pemeriksaan lab: proteinuria, CBC, fungsi ginjal & hati
- 🟡 Siapkan manajemen preeklampsia/eklamsia jika diperlukan
- 🟡 Edukasi keluarga tanda bahaya yang memerlukan pertolongan darurat
""")
        else:
            st.markdown("""
- 🟢 Lanjutkan kunjungan antenatal rutin sesuai jadwal
- 🟢 Pantau tekanan darah minimal setiap kunjungan
- 🟡 Edukasi tanda bahaya: perdarahan, nyeri kepala hebat, pandangan kabur
- 🟡 Anjurkan pola makan sehat dan aktivitas fisik ringan yang sesuai
""")

    else:
        st.info("👈 Lengkapi data pasien di sebelah kiri, lalu klik **Prediksi Risiko**.")
        st.markdown("""
**Sistem ini menghitung:**
- 🧮 **Clinical Scoring** — berdasarkan 9 faktor risiko klinis
- 🤖 **ML Model Prediction** — dari model yang telah dilatih
- 📌 **Rekomendasi** tindakan klinis sesuai hasil
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("© 2025 Sistem Prediksi Risiko Maternal · Hanya untuk keperluan klinis terotorisasi · SYS v2.0.0")
