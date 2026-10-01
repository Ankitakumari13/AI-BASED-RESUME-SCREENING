# AI-Based Resume Screening & Candidate Shortlisting (Streamlit app)

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the URL shown in the terminal (usually http://localhost:8501).

## Use
1. Paste a job description (or click **Load demo job + resumes** in the sidebar).
2. Upload resumes (PDF, DOCX, TXT).
3. Click **Screen candidates** to get a ranked shortlist, matched/missing skills per candidate, and a CSV download.

## Files
- `app.py` - Streamlit front end
- `screening.py` - cleaning/anonymisation, feature engineering and scoring
- `resume_screening_model.joblib` - trained Logistic Regression + TF-IDF vectorizers
- `sample_jd.txt`, `sample_resumes/` - demo data

## Notes
- The model is trained on a synthetic dataset (see `generate_dataset.py` in the project). Treat scores as decision support, not a hiring decision.
- Scanned (image-only) PDFs have no text layer and are skipped.
- `scikit-learn` is pinned to the version used for training so the saved model loads correctly.
