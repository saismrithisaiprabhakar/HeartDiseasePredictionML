import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import find_dotenv, load_dotenv
import os  
from groq import Groq
from cryptography.fernet import Fernet

st.set_page_config(page_title="Heart Disease Risk (XGBoost)", page_icon="❤️", layout="wide")
st.title("Heart Disease Risk (XGBoost)")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    with open("metadata.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()
#print("Pipeline steps:", list(model.named_steps.keys()))

# Pull fitted encoder/categories from the pipeline
prep = model.named_steps["prep"]
ohe  = prep.named_transformers_["cat"]
num_cols_fitted = list(prep.transformers_[0][2])
cat_cols_fitted = list(prep.transformers_[1][2])
#print(num_cols_fitted)
#print(cat_cols_fitted)

def _to_py(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    return x
    
def _coerce_cat_to_option(val, options):
    """Coerce a raw dataframe value to a valid selectbox option."""
    if not options:
        return None
    opts = [_to_py(v) for v in options]
    v = _to_py(val)
    if v in opts:
        return v
    # common casts (e.g., '1' vs 1)
    candidates = [v, str(v)]
    try:
        fv = float(v)
        if fv.is_integer():
            candidates.append(int(fv))
    except Exception:
        pass
    for c in candidates:
        if c in opts:
            return c
    return opts[0]
    
cat_values = {feat: [_to_py(v) for v in cats]
              for feat, cats in zip(cat_cols_fitted, ohe.categories_)}
              

# Sidebar (this must be real code, not quoted text)
with st.sidebar:
    st.header("About this score")
    st.write(
        "- This shows the **predicted probability of heart disease (class 1)** "
        "from the model (`predict_proba`).\n"
        "- Score is 0–1 (higher = higher estimated risk).\n"
        "- No threshold/decision is applied."
    )
    st.caption("Demo only. Not medical advice.")

# Build inputs
FEATURES = meta["feature_order"]

fm = meta.get("feature_means", {})
fs = meta.get("feature_stds", {})
    # ensure floats
feature_means = {k: float(v) for k, v in fm.items()} if fm else {}
feature_stds  = {k: float(v) if float(v) != 0 else np.nan for k, v in fs.items()} if fs else {}

fitted_order = num_cols_fitted + cat_cols_fitted
if set(FEATURES) != set(fitted_order):
    st.warning("Feature mismatch between metadata.json and the fitted preprocessor. "
               "Using fitted preprocessor’s column order.")
    FEATURES = fitted_order
    
    
@st.cache_data
def try_load_data():
    try:
        # change path/name if needed; commit a small sample for Streamlit Cloud
        df_data = pd.read_csv("./heart_test.csv")
        df_data = df_data.drop(['predicted_labels', 'predicted_probability'], inplace = True)
        return df_data
    except Exception:
        return None

df_random = try_load_data()

st.write("### Quick fill")
btn = st.button("🎲 Insert Data", disabled=(df_random is None))
if btn:
    if df_random is None or len(df_random) == 0:
        st.warning("No dataset found or dataset is empty.")
    else:
        row = df_random.sample(1).iloc[0]
        for feat in FEATURES:
            if feat in cat_values:
                st.session_state[f"sel_{feat}"] = _coerce_cat_to_option(row.get(feat, None), cat_values[feat])
            else:
                v = row.get(feat, np.nan)
                try:
                    st.session_state[f"in_{feat}"] = float(v) if pd.notna(v) else 0.0
                except Exception:
                    st.session_state[f"in_{feat}"] = 0.0
        st.rerun()

if df_random is None:
    st.caption("Tip: Commit a small sample dataset at `data/heart.csv` to enable the random-fill button.")

st.subheader("Enter Features")
vals = {}
c1, c2 = st.columns(2)
feature_ranges = meta.get("feature_ranges", {})

for i, feat in enumerate(FEATURES):
    with (c1 if i % 2 == 0 else c2):
        if feat in cat_values:
            choice = st.selectbox(
                feat,
                options=cat_values[feat],
                key=f"sel_{feat}"
            )
            vals[feat] = choice
        else:
            rng = feature_ranges.get(feat)
            key = f"in_{feat}"
            if rng:
                min_val = float(rng["min"])
                max_val = float(rng["max"])
                label = f"{feat} (range: {min_val:.1f} – {max_val:.1f})"

                if key in st.session_state:
                    vals[feat] = st.number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        key=key
                    )
                else:
                    default_val = (min_val + max_val) / 2
                    vals[feat] = st.number_input(
                        label,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        key=key
                    )
            else:
                if key in st.session_state:
                    vals[feat] = st.number_input(feat, key=key)
                else:
                    vals[feat] = st.number_input(feat, value=0.0, key=key)
                    
def _decrypt_with_fernet(enc_text: str, fernet_key_b64: str) -> str:
    """
    Decrypts a Fernet-encrypted base64 string using a base64 Fernet key.
    Returns the plaintext string.
    """
    if not enc_text:
        raise RuntimeError("Encrypted text is empty.")
    if not fernet_key_b64:
        raise RuntimeError("FERNET_KEY is missing.")

    try:
        f = Fernet(fernet_key_b64.encode())
        plaintext_bytes = f.decrypt(enc_text.encode())
        return plaintext_bytes.decode()
    except Exception as e:
        raise RuntimeError(f"Unexpected decrypt error: {e}")

def get_groq_api_key() -> str:
    # 1) Get Fernet key **from Streamlit secrets**
    try:
        fernet_key_b64 = st.secrets["FERNET_KEY"]
    except KeyError:
        # Optional: allow local dev fallback via env var
        fernet_key_b64 = os.environ.get("FERNET_KEY")
        if not fernet_key_b64:
            raise RuntimeError("FERNET_KEY not found in st.secrets or environment.")

    # 2) Load your .env that contains the **encrypted** Groq key
    dotenv_path = find_dotenv(filename="groq_api.env", usecwd=True)
    if not dotenv_path:
        raise RuntimeError("Could not find groq_api.env")
    load_dotenv(dotenv_path=dotenv_path, override=False)

    # 3) Read encrypted key from env. Prefer GROQ_API_KEY_ENC; fallback to GROQ_API_KEY.
    enc = os.getenv("GROQ_API_KEY_ENC") or os.getenv("GROQ_API_KEY")
    if not enc:
        raise RuntimeError("Encrypted key not found in groq_api.env (GROQ_API_KEY_ENC / GROQ_API_KEY).")

    # 4) Decrypt with Fernet key from st.secrets
    return _decrypt_with_fernet(enc_text=enc, fernet_key_b64=fernet_key_b64)


def llm_explanation_with_groq(probability: float, inputs: dict) -> str:
    """
    Calls Groq to turn a probability + inputs into a layperson summary.
    """

    # Load API key - uncomment before commiting to git.
    dotenv_path = find_dotenv(filename="groq_api.env", usecwd=True)
    load_dotenv(dotenv_path=dotenv_path, override=True)
    key = get_groq_api_key()
    
    #Comment/remove before putting to github
    """otenv_path = find_dotenv(filename="groq_api_local.env", usecwd=True)
    load_dotenv(dotenv_path=dotenv_path, override=True)
    key = os.getenv("GROQ_API_KEY")"""


    # Initialize client
    client = Groq(api_key=key)
    MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

    # Compute percentage
    pct = round(probability * 100, 1)

    # Prompt
    prompt = f"""
You are a compassionate, knowledgeable doctor explaining heart-health test results to a patient in simple, calm language.
Your tone should be professional, warm, and reassuring — not alarming.The inputs include features like age, sex, chest pain type, resting BP, 
cholesterol, fasting blood sugar, resting ECG results,exercise-induced angina, oldpeak (ST depression after exercise), slope of the ST segment, 
ca (number of major vessels seen in fluoroscopy), and thal (thalassemia).
Task:
Write 3-4 short bullet points explanation summarizing the results below as a doctor explains a patient.
Inputs:
Estimated probability of heart disease (class 1): {pct}%
Model inputs (verbatim): {inputs}
Guidelines:
Begin by stating what the estimated percentage means — it is a model-based statistical estimate, not a diagnosis.
If the probability is high (> 70 %), gently explain that the result suggests a higher chance of heart-related concerns and that a discussion with a healthcare provider would be helpful.
For each key feature mentioned in the inputs, give a short explanation of what it measures,typical range, and why this value might or might not suggest risk use your medical knowledge.
Summarize the findings calmly: note which readings fall in the usual range and which might warrant follow-up. Do not have more than 4 points with less than 12 words.
End with a clear disclaimer:“This explanation is for educational purposes to help you understand your results and is not a medical diagnosis.”"""

    # Call Groq
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.1,
        max_tokens=220,
        messages=[
            {"role": "system", "content": "You write clear, patient-friendly summaries."},
            {"role": "user", "content": prompt},
        ],
    )

    return resp.choices[0].message.content.strip()

def _is_one(val):
    # handle 1, "1", True consistently
    return str(val).strip().lower() in {"1", "true", "1.0"}

def _as_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


if st.button("Get risk score"):
    # 1) Build input row for prediction
    X_row = pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)
    print(X_row)
    # 2) Predict probability
    try:
        p1 = float(model.predict_proba(X_row)[:, 1][0])
    except AttributeError:
        p1 = float(model.predict(X_row)[0])

    # 3) Show probability
    st.success(f"**Predicted probability (class 1):** {p1:.3f}")
    st.write("Risk score (0–1)")
    st.progress(min(max(p1, 0.0), 1.0))

    # 4) Build review table with flags
    AGE_CUT = 62
    OLDPEAK_CUT = 1.7
    CHOL_CUT = 240

    rows = []
    for feat in FEATURES:
        val = vals.get(feat)
        flagged = False
        reason  = ""

        # ---- Numeric thresholds (no mean-based logic) ----
        if feat not in cat_values:
            v = _as_float(val)

            if feat == "age" and not np.isnan(v) and v > AGE_CUT:
                flagged = True
                reason  = f"age > {AGE_CUT}"

            elif feat == "oldpeak" and not np.isnan(v) and v > OLDPEAK_CUT:
                flagged = True
                reason  = f"oldpeak > {OLDPEAK_CUT}"

            elif feat == "chol" and not np.isnan(v) and v > CHOL_CUT:
                flagged = True
                reason  = f"chol > {CHOL_CUT}"

        # ---- Categorical thresholds ----
        else:
            if feat == "exang" and _is_one(val):
                flagged = True
                reason  = "exang == 1"

        rows.append({
            "feature": feat,
            "entered_value": val,
            # keep mean if you already compute feature_means; otherwise leave ""
            "rule_triggered": reason,
            "flag": flagged
        })

    df_view = pd.DataFrame(rows)
    

    # 5) Style: shades of red for numeric > mean (stronger if farther above mean).
    #    For categorical flags, use a solid red highlight.
    def _cell_style(row):
        styles = [""] * len(df_view.columns)
        if "entered_value" not in df_view.columns:
            return styles

        idx_val  = df_view.columns.get_loc("entered_value")
        idx_rule = df_view.columns.get_loc("rule_triggered")

        feat = row.get("feature")
        val  = row.get("entered_value")
        v    = _as_float(val)

    # exang == 1 (categorical)
        if feat == "exang" and _is_one(val):
            styles[idx_val]  = "background-color: rgba(255,0,0,0.45); color:#7a0010; font-weight:700;"
            styles[idx_rule] = "color:#7a0010; font-weight:700;"
            return styles

    # age > 62 with gradient (stronger when farther above 62)
        if feat == "age" and not np.isnan(v) and v > AGE_CUT:
            gap   = max(0.0, v - AGE_CUT)
            alpha = max(0.25, min(0.70, 0.25 + gap * 0.03))  # tune gradient here
            styles[idx_val]  = f"background-color: rgba(255,0,0,{alpha}); color:#7a0010; font-weight:700;"
            styles[idx_rule] = "color:#7a0010; font-weight:700;"
            return styles

    # oldpeak > 1.7 or chol > 240 (solid red)
        if feat == "oldpeak" and not np.isnan(v) and v > OLDPEAK_CUT:
            styles[idx_val]  = "background-color: rgba(255,0,0,0.45); color:#7a0010; font-weight:700;"
            styles[idx_rule] = "color:#7a0010; font-weight:700;"
            return styles

        if feat == "chol" and not np.isnan(v) and v > CHOL_CUT:
            styles[idx_val]  = "background-color: rgba(255,0,0,0.45); color:#7a0010; font-weight:700;"
            styles[idx_rule] = "color:#7a0010; font-weight:700;"
            return styles

        return styles

# show the same columns; cast entered_value to str for safe rendering
    df_view = df_view[['feature', 'entered_value', 'rule_triggered']]
    df_view["entered_value"] = df_view["entered_value"].astype(str)

    st.subheader("Entered values vs thresholds")
    st.dataframe(
    df_view.style.apply(_cell_style, axis=1),
        use_container_width=True
    )
    # 4️⃣ Call Groq for a natural explanation
    explanation = llm_explanation_with_groq(p1, vals)

    # 5️⃣ Display it
    st.markdown("### Human-readable explanation (LLM)")
    if explanation:
        st.write(explanation)
    else:
        st.info("Could not generate an explanation. Please check your Groq API key.")
