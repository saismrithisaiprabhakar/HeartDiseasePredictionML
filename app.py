import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Risk (XGBoost)", page_icon="❤️", layout="wide")
st.title("Heart Disease Risk (XGBoost)")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    with open("metadata.json") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()

# Pull fitted encoder/categories from the pipeline
prep = model.named_steps["prep"]
ohe  = prep.named_transformers_["cat"]
num_cols_fitted = list(prep.transformers_[0][2])
cat_cols_fitted = list(prep.transformers_[1][2])

def _to_py(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    return x

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
fitted_order = num_cols_fitted + cat_cols_fitted
if set(FEATURES) != set(fitted_order):
    st.warning("Feature mismatch between metadata.json and the fitted preprocessor. "
               "Using fitted preprocessor’s column order.")
    FEATURES = fitted_order

st.subheader("Enter Features")
vals = {}
c1, c2 = st.columns(2)

for i, feat in enumerate(FEATURES):
    with (c1 if i % 2 == 0 else c2):
        if feat in cat_values:
            choice = st.selectbox(feat, options=cat_values[feat], key=f"sel_{feat}")
            vals[feat] = choice
        else:
            if feat == "oldpeak":
                vals[feat] = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            elif feat == "age_oldpeak":
                vals[feat] = st.number_input("age_oldpeak (engineered)", value=54.0, step=0.1, format="%.2f")
            elif feat == "ca" and feat not in cat_values:
                vals[feat] = st.number_input("Number of vessels (ca)", min_value=0, max_value=4, value=0, step=1)
            else:
                vals[feat] = st.number_input(feat, value=0.0)
def llm_explanation_with_groq(probability: float, inputs: dict) -> str:
    """
    Calls Groq to turn a probability + inputs into a layperson summary.
    """
    from dotenv import find_dotenv, load_dotenv
    import os
    from groq import Groq

    # Load API key
    dotenv_path = find_dotenv(filename="groq_api.env", usecwd=True)
    load_dotenv(dotenv_path=dotenv_path, override=True)
    key = os.getenv("GROQ_API_KEY")

    # Initialize client
    client = Groq(api_key=key)
    MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

    # Compute percentage
    pct = round(probability * 100, 1)

    # Prompt
    prompt = f"""
You are a compassionate, knowledgeable doctor explaining heart-health test results to a patient in simple, calm language.
Your tone should be professional, warm, and reassuring — not alarming.
Task:
Write a 6–8 sentence explanation summarizing the results below in a way a patient can easily understand.
Inputs:
Estimated probability of heart disease (class 1): {pct}%
Model inputs (verbatim): {inputs}
Guidelines:
Begin by stating what the estimated percentage means — it is a model-based statistical estimate, not a diagnosis.
If the probability is high (> 70 %), gently explain that the result suggests a higher chance of heart-related concerns and that a discussion with a healthcare provider would be helpful.
For each key feature mentioned in the inputs, give a short explanation of what it measures,typical range, and why this value might or might not suggest risk.
Summarize the findings calmly: note which readings fall in the usual range and which might warrant follow-up.
End with a clear disclaimer:

“This explanation is for educational purposes to help you understand your results and is not a medical diagnosis.”"""

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

   


if st.button("Get risk score"):
    # 1️⃣ Make a DataFrame for a single record
    X_row = pd.DataFrame([[vals[f] for f in FEATURES]], columns=FEATURES)

    # 2️⃣ Predict probability
    p1 = float(model.predict_proba(X_row)[:, 1][0])

    # 3️⃣ Show numeric prediction
    st.success(f"**Predicted probability (class 1):** {p1:.3f}")
    st.write("Risk score (0–1)")
    st.progress(min(max(p1, 0.0), 1.0))

    # 4️⃣ Call Groq for a natural explanation
    explanation = llm_explanation_with_groq(p1, vals)

    # 5️⃣ Display it
    st.markdown("### Human-readable explanation (LLM)")
    if explanation:
        st.write(explanation)
    else:
        st.info("Could not generate an explanation. Please check your Groq API key.")
