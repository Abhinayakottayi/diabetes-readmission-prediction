from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
 
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_pipeline.joblib"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
 
THRESHOLD = 0.437         
BASE_RATE = 0.09
TARGET = "readmit_30"
ID_COLS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
MED_COLS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
            "glimepiride", "acetohexamide", "glipizide", "glyburide",
            "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
            "miglitol", "troglitazone", "tolazamide", "insulin",
            "glyburide-metformin", "glipizide-metformin",
            "glimepiride-pioglitazone", "metformin-rosiglitazone",
            "metformin-pioglitazone"]
SHOWN_MEDS = ["insulin", "metformin", "glipizide", "glyburide",
              "glimepiride", "pioglitazone", "rosiglitazone"]
 
AGE_BANDS = {f"{i}-{i + 10}": i + 5 for i in range(0, 100, 10)}
 

ADMISSION_TYPE = {"1": "Emergency", "2": "Urgent", "3": "Elective",
                  "4": "Newborn", "5": "Not available", "6": "Not available (NULL)",
                  "7": "Trauma center", "8": "Not mapped"}
DISCHARGE = {"1": "Discharged to home", "2": "Short-term hospital",
             "3": "Skilled nursing facility", "4": "Intermediate care facility",
             "5": "Other inpatient care", "6": "Home with home health service",
             "7": "Left against medical advice", "8": "Home under IV therapy",
             "22": "Rehab facility", "23": "Long-term care hospital",
             "24": "Nursing facility", "25": "Not mapped"}
ADMISSION_SOURCE = {"1": "Physician referral", "2": "Clinic referral",
                    "3": "HMO referral", "4": "Transfer from hospital",
                    "5": "Transfer from skilled nursing facility",
                    "6": "Transfer from other facility",
                    "7": "Emergency room", "17": "Not available (NULL)",
                    "20": "Not mapped"}
 
 
def label(code, mapping):
    return f"{mapping.get(code, f'Code {code}')} ({code})"
 
 
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)
 
 
@st.cache_data
def load_reference(expected_cols):
    """Dropdown options and fallback defaults, taken from the training data."""
    train = pd.read_csv(TRAIN_PATH, keep_default_na=False,
                        dtype={c: str for c in ID_COLS})
    X = train.drop(columns=[TARGET])[list(expected_cols)]
    defaults, options = {}, {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            defaults[col] = float(X[col].median())
        else:
            defaults[col] = X[col].mode().iloc[0]
            options[col] = sorted(X[col].unique())
    return defaults, options
 
 
@st.cache_data
def load_test_scores(_pipe, expected_cols):
    """Model scores on the held-out test set, for percentile context."""
    if not TEST_PATH.exists():
        return None
    test = pd.read_csv(TEST_PATH, keep_default_na=False,
                       dtype={c: str for c in ID_COLS})
    X = test.drop(columns=[TARGET])[list(expected_cols)]
    return np.sort(_pipe.predict_proba(X)[:, 1])
 
 
st.set_page_config(page_title="Diabetes Readmission Risk", page_icon="🏥",
                   layout="wide")
st.title("Diabetes 30-Day Readmission Risk")
st.caption("Educational demo. Not a clinical tool. Do not use for real patient decisions.")
 
missing = [p for p in (MODEL_PATH, TRAIN_PATH) if not p.exists()]
if missing:
    st.error("Missing files: " + ", ".join(str(p.relative_to(ROOT)) for p in missing)
             + ". Run the notebooks first and start this app from the project root.")
    st.stop()
 
pipe = load_pipeline()
expected = tuple(pipe.named_steps["prep"].feature_names_in_)
defaults, options = load_reference(expected)
 
 
def pick(label_text, col, mapping=None, default=None):
    opts = options[col]
    start = default if default in opts else defaults[col]
    fmt = (lambda c: label(c, mapping)) if mapping else str
    return st.selectbox(label_text, opts, index=opts.index(start), format_func=fmt)
 
 
with st.form("patient"):
    st.subheader("Patient and stay")
    c1, c2, c3 = st.columns(3)
    with c1:
        age_band = st.selectbox("Age band", list(AGE_BANDS), index=7)
        gender = pick("Gender", "gender")
        race = pick("Race", "race")
        time_in_hospital = st.slider("Days in hospital", 1, 14, 4)
    with c2:
        admission_type = pick("Admission type", "admission_type_id",
                              ADMISSION_TYPE, default="1")
        admission_source = pick("Admission source", "admission_source_id",
                                ADMISSION_SOURCE, default="7")
        discharge = pick("Discharge disposition", "discharge_disposition_id",
                         DISCHARGE, default="1")
        specialty = pick("Medical specialty", "medical_specialty")
    with c3:
        payer = pick("Payer code", "payer_code")
        diag_1 = pick("Primary diagnosis group", "diag_1_group")
        diag_2 = pick("Secondary diagnosis group", "diag_2_group")
        diag_3 = pick("Third diagnosis group", "diag_3_group")
 
    st.subheader("Prior use of healthcare (past year)")
    c1, c2, c3 = st.columns(3)
    with c1:
        n_inpatient = st.number_input("Inpatient visits", 0, 20, 0)
    with c2:
        n_emergency = st.number_input("Emergency visits", 0, 20, 0)
    with c3:
        n_outpatient = st.number_input("Outpatient visits", 0, 30, 0)
 
    st.subheader("This encounter")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_lab = st.slider("Lab procedures", 1, 130, 43)
    with c2:
        n_proc = st.slider("Other procedures", 0, 6, 1)
    with c3:
        n_med = st.slider("Medications given", 1, 80, 15)
    with c4:
        n_diag = st.slider("Number of diagnoses", 1, 16, 8)
 
    st.subheader("Labs and diabetes treatment")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        a1c = pick("A1C result", "A1Cresult")
    with c2:
        glu = pick("Max glucose serum", "max_glu_serum")
    with c3:
        change = pick("Medication change", "change")
    with c4:
        diabetes_med = pick("On diabetes medication", "diabetesMed")
 
    med_values = {}
    cols = st.columns(len(SHOWN_MEDS))
    for col, med in zip(cols, SHOWN_MEDS):
        with col:
            med_values[med] = st.selectbox(med.capitalize(),
                                           ["No", "Steady", "Up", "Down"])
 
    submitted = st.form_submit_button("Estimate risk", type="primary")
 
if submitted:
    row = dict(defaults)
    row.update({
        "gender": gender, "race": race, "age_mid": AGE_BANDS[age_band],
        "time_in_hospital": time_in_hospital,
        "admission_type_id": admission_type,
        "admission_source_id": admission_source,
        "discharge_disposition_id": discharge,
        "medical_specialty": specialty, "payer_code": payer,
        "diag_1_group": diag_1, "diag_2_group": diag_2, "diag_3_group": diag_3,
        "number_inpatient": n_inpatient, "number_emergency": n_emergency,
        "number_outpatient": n_outpatient,
        "num_lab_procedures": n_lab, "num_procedures": n_proc,
        "num_medications": n_med, "number_diagnoses": n_diag,
        "A1Cresult": a1c, "max_glu_serum": glu,
        "change": change, "diabetesMed": diabetes_med,
    })
    for med in MED_COLS:                      
        if med in row:
            row[med] = med_values.get(med, "No")

    meds = [row[m] for m in MED_COLS if m in row]
    row["total_prior_visits"] = n_inpatient + n_emergency + n_outpatient
    row["any_prior_inpatient"] = int(n_inpatient > 0)
    row["n_diabetes_meds"] = sum(v != "No" for v in meds)
    row["n_meds_changed"] = sum(v in ("Up", "Down") for v in meds)
    row["meds_per_day"] = n_med / time_in_hospital
    row["procedures_total"] = n_lab + n_proc
 
    X_new = pd.DataFrame([row])[list(expected)]
    for c in ID_COLS:
        if c in X_new:
            X_new[c] = X_new[c].astype(str)
 
    score = float(pipe.predict_proba(X_new)[0, 1])
    flagged = score >= THRESHOLD
 
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.metric("Model score", f"{score:.3f}")
        st.metric("Decision threshold", f"{THRESHOLD:.3f}")
    with right:
        if flagged:
            st.warning("**Flagged as higher risk** of readmission within 30 days.")
        else:
            st.success("**Not flagged.** Score is below the decision threshold.")
 
        scores = load_test_scores(pipe, expected)
        if scores is not None:
            pct = np.searchsorted(scores, score) / len(scores) * 100
            st.write(f"This score is higher than about **{pct:.0f}%** of "
                     f"patients in the held-out test set.")
        st.caption(
            f"About {BASE_RATE:.0%} of patients in this dataset were readmitted "
            "within 30 days. The model was trained with class weights, so the "
            "score is a ranking score, not a literal probability. At this "
            "threshold, roughly 85% of flagged patients in testing were not "
            "readmitted."
        )
 
with st.expander("About this model"):
    st.markdown(
        "- Random Forest trained on the Diabetes 130-US Hospitals dataset (1999-2008).\n"
        "- Test ROC-AUC 0.658, PR-AUC 0.179, recall 51% and precision 15% at the threshold.\n"
        "- Moderate predictive power only. Fairness across groups has not been assessed.\n"
        "- Inputs not shown here use the typical (median or most common) training value."
    )
 
