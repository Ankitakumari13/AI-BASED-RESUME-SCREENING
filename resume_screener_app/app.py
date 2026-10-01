import io
from pathlib import Path

import pandas as pd
import streamlit as st

from screening import ROLES, Screener, extract_text

BASE = Path(__file__).parent

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")


@st.cache_resource
def load_screener():
    return Screener()


def load_demo():
    st.session_state["jd_text"] = (BASE / "sample_jd.txt").read_text()
    st.session_state["demo"] = True


def clear_demo():
    st.session_state["demo"] = False


# ------------------------------------------------------------------ header
st.title("📄 AI-Based Resume Screening & Candidate Shortlisting")
st.caption("Paste a job description, upload resumes, and get a ranked shortlist with the reasons behind each score.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Shortlist size (top N)", 1, 20, 5)
    role = st.selectbox("Role (optional)", ["Auto-detect from job description"] + ROLES)
    st.divider()
    st.button("Load demo job + resumes", on_click=load_demo, width="stretch")
    st.button("Clear demo", on_click=clear_demo, width="stretch")
    st.divider()
    st.info(
        "This is a decision-support tool trained on a **synthetic** dataset. "
        "A human should always make the final hiring decision."
    )

# ------------------------------------------------------------------ inputs
left, right = st.columns(2)
with left:
    st.subheader("1. Job description")
    jd_text = st.text_area("Paste the job description", key="jd_text", height=220,
                           placeholder="e.g. Data Scientist opening: hands-on experience in Python, machine learning, SQL... 3+ years of experience, Master's degree.")
with right:
    st.subheader("2. Resumes")
    files = st.file_uploader("Upload resumes (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if st.session_state.get("demo"):
        st.success("Demo resumes loaded (6 candidates from `sample_resumes/`).")

run = st.button("🔍 Screen candidates", type="primary", width="stretch")

# ------------------------------------------------------------------ run
if run:
    names, texts = [], []
    for f in files or []:
        try:
            txt = extract_text(f.read(), f.name)
        except Exception as e:  # unreadable file
            st.warning(f"Could not read {f.name}: {e}")
            continue
        if txt.strip():
            names.append(f.name)
            texts.append(txt)
        else:
            st.warning(f"{f.name}: no text found (scanned PDF?). Skipped.")
    if st.session_state.get("demo"):
        for p in sorted((BASE / "sample_resumes").glob("*.txt")):
            names.append(p.name)
            texts.append(p.read_text())

    if not jd_text.strip():
        st.error("Please paste a job description.")
    elif not texts:
        st.error("Please upload at least one resume (or load the demo).")
    else:
        sel_role = "" if role.startswith("Auto") else role
        with st.spinner("Screening..."):
            result, jd_skills = load_screener().score(texts, names, jd_text, sel_role)
        result["Shortlisted"] = (result["Rank"] <= top_n) & (result["Recommendation"] == "Recommend")
        st.session_state["result"] = result
        st.session_state["texts"] = dict(zip(names, texts))
        st.session_state["jd_skills"] = jd_skills

# ------------------------------------------------------------------ results
if "result" in st.session_state:
    result = st.session_state["result"]
    texts = st.session_state["texts"]
    jd_skills = st.session_state["jd_skills"]

    st.divider()
    st.subheader("3. Results")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates screened", len(result))
    m2.metric("Recommended", int((result["Recommendation"] == "Recommend").sum()))
    m3.metric("Shortlisted (top N, recommended)", int(result["Shortlisted"].sum()))
    m4.metric("Average match", f"{result['Match score'].mean():.0%}")
    if jd_skills:
        st.write("**Skills detected in the job description:** " + ", ".join(jd_skills))
    else:
        st.warning("No known skills were detected in the job description, so the skill-match score may be unreliable.")

    show = result[["Rank", "Candidate", "Match score", "Skill match", "Experience (yrs)", "Recommendation", "Shortlisted"]]
    st.dataframe(
        show, hide_index=True, width="stretch",
        column_config={
            "Match score": st.column_config.ProgressColumn("Match score", min_value=0.0, max_value=1.0, format="percent"),
            "Skill match": st.column_config.ProgressColumn("Skill match", min_value=0.0, max_value=1.0, format="percent"),
        },
    )
    st.bar_chart(result.set_index("Candidate")["Match score"], horizontal=True)

    st.subheader("Why this ranking?")
    for _, r in result.iterrows():
        icon = "✅" if r["Shortlisted"] else "▫️"
        with st.expander(f"{icon} #{r['Rank']}  {r['Candidate']}  —  {r['Match score']:.0%} match  ({r['Recommendation']})"):
            c1, c2 = st.columns(2)
            c1.markdown("**Matched skills**  \n" + (r["Matched skills"] or "_none_"))
            c2.markdown("**Missing skills**  \n" + (r["Missing skills"] or "_none_"))
            st.markdown(f"**Experience found:** {r['Experience (yrs)']} years")
            st.text_area("Resume preview", texts[r["Candidate"]][:2500], height=180, disabled=True,
                         key=f"prev_{r['Candidate']}")

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download ranked shortlist (CSV)", csv, "shortlist.csv", "text/csv")
