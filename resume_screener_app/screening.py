"""Core logic for the AI Resume Screening system (feature engineering + scoring)."""
import io, re
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

MODEL_PATH = Path(__file__).parent / "resume_screening_model.joblib"

ROLES = ['Data Scientist', 'Machine Learning Engineer', 'Data Analyst', 'Data Engineer', 'Backend Developer', 'Frontend Developer', 'Full Stack Developer', 'DevOps Engineer', 'Cloud Engineer', 'QA Engineer', 'Cybersecurity Analyst', 'UI/UX Designer', 'Business Analyst', 'Digital Marketing Specialist', 'HR Specialist']

SKILLS = ['a/b testing', 'accessibility', 'adobe xd', 'agile', 'airflow', 'ansible', 'api testing', 'aws', 'azure', 'bash', 'brd', 'canva', 'ci/cd', 'cloud security', 'cloudformation', 'computer vision', 'confluence', 'content marketing', 'copywriting', 'crm', 'css', 'cypress', 'data cleaning', 'data visualization', 'data warehousing', 'deep learning', 'design systems', 'docker', 'email marketing', 'employee relations', 'etl', 'excel', 'figma', 'firewalls', 'gcp', 'git', 'google ads', 'google analytics', 'graphql', 'hadoop', 'helm', 'hris', 'html', 'hubspot', 'incident response', 'iso 27001', 'java', 'javascript', 'jenkins', 'jest', 'jira', 'kafka', 'kubernetes', 'labor law', 'linux', 'looker', 'machine learning', 'manual testing', 'microservices', 'mlops', 'mongodb', 'monitoring', 'motion design', 'network security', 'networking', 'next.js', 'nlp', 'node.js', 'onboarding', 'owasp', 'pandas', 'payroll', 'penetration testing', 'performance management', 'performance testing', 'postgresql', 'postman', 'power bi', 'process modeling', 'prometheus', 'prototyping', 'python', 'pytorch', 'r', 'react', 'recruitment', 'redis', 'redux', 'requirements gathering', 'responsive design', 'rest apis', 'risk assessment', 'scala', 'scikit-learn', 'security', 'selenium', 'sem', 'seo', 'serverless', 'siem', 'sketch', 'snowflake', 'social media marketing', 'sourcing', 'spark', 'splunk', 'spring boot', 'sql', 'stakeholder management', 'statistics', 'tableau', 'talent acquisition', 'tensorflow', 'terraform', 'test automation', 'test cases', 'training and development', 'typescript', 'uml', 'usability testing', 'user research', 'webpack', 'wireframing', 'wireshark', 'workday']

SYN = {'ml': 'machine learning', 'sklearn': 'scikit-learn', 'powerbi': 'power bi', 'js': 'javascript', 'nodejs': 'node.js',
       'restful services': 'rest apis', 'k8s': 'kubernetes', 'postgres': 'postgresql', 'amazon web services': 'aws',
       'search engine optimization': 'seo', 'ga4': 'google analytics', 'hr information systems': 'hris', 'neural networks': 'deep learning',
       'automation testing': 'test automation', 'siem tools': 'siem', 'etl pipelines': 'etl', 'ci/cd pipelines': 'ci/cd',
       'natural language processing': 'nlp'}


def normalise(text):
    for k, v in SYN.items():
        text = re.sub(r'(?<![a-z0-9])' + re.escape(k) + r'(?![a-z0-9])', v, text)
    return text


def skills_in(text):
    text = normalise(text)
    return {s for s in SKILLS if re.search(r'(?<![a-z])' + re.escape(s) + r'(?![a-z])', text)}


def years_exp(text):
    m = re.findall(r'(\d+)\+?\s*years', text)
    return max(map(int, m)) if m else 0


def clean_resume(text):
    """Anonymise (drop name/email/phone/links) and lowercase, same as in training."""
    m = re.match(r"Here's a professional resume for ([^:\n]+):", text)
    if m:
        name, t = m.group(1).strip(), re.sub(r"^Here's a professional resume for [^:\n]+:\s*", '', text)
    else:
        name, _, t = text.partition('\n')
        name = name.strip()
    if name:
        t = t.replace(name, ' ')
    t = re.sub(r'\[([^\]]*)\]\([^)]*\)', ' ', t)
    t = re.sub(r'\S+@\S+', ' ', t)
    t = re.sub(r'(https?://|www\.)\S+|linkedin\.com/\S+|github\.com/\S+', ' ', t)
    t = re.sub(r'\+?\d[\d\-\s().]{8,}\d', ' ', t)
    t = re.sub(r'[*#_`>]+', ' ', t)
    return re.sub(r'\s+', ' ', t).strip().lower()


def infer_role(jd):
    low = jd.lower()
    for r in sorted(ROLES, key=len, reverse=True):
        if r.lower() in low:
            return r
    return ''


def extract_text(data: bytes, filename: str) -> str:
    """Read text from an uploaded PDF, DOCX or TXT file."""
    name = filename.lower()
    if name.endswith('.pdf'):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return '\n'.join((p.extract_text() or '') for p in reader.pages)
    if name.endswith('.docx'):
        from docx import Document
        return '\n'.join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return data.decode('utf-8', errors='ignore')


class Screener:
    def __init__(self, path=MODEL_PATH):
        art = joblib.load(path)
        self.model = art['model']
        self.tfidf_shared = art['tfidf_shared']
        self.tfidf_res = art['tfidf_res']
        self.tfidf_jd = art['tfidf_jd']
        self.scale = art['scale']

    def _features(self, resumes_clean, jd_clean, role):
        R = self.tfidf_shared.transform(resumes_clean)
        J = self.tfidf_shared.transform([jd_clean] * len(resumes_clean))
        cos = np.asarray(R.multiply(J).sum(1)).ravel()
        js = skills_in(jd_clean)
        rs = [skills_in(t) for t in resumes_clean]
        F = pd.DataFrame({
            'cosine_sim': cos,
            'skill_overlap': [len(r & js) / max(len(js), 1) for r in rs],
            'n_resume_skills': [len(r) for r in rs],
            'n_jd_skills': [len(js)] * len(rs),
            'years_exp': [years_exp(t) for t in resumes_clean],
            'resume_len': [len(t.split()) for t in resumes_clean],
            'jd_len': [len(jd_clean.split())] * len(rs),
            'role_in_resume': [int(role.lower() in t) for t in resumes_clean],
        })
        return F, rs, js

    def score(self, resumes, names, job_description, role=''):
        """Return a DataFrame ranked by selection probability."""
        role = role or infer_role(job_description)
        jd_clean = re.sub(r'\s+', ' ', job_description.lower()).strip()
        cleaned = [clean_resume(t) for t in resumes]
        F, rs, js = self._features(cleaned, jd_clean, role)
        X = hstack([self.tfidf_res.transform(cleaned), self.tfidf_jd.transform([jd_clean] * len(cleaned)),
                    csr_matrix((F / self.scale).values)]).tocsr()
        p = self.model.predict_proba(X)[:, 1]
        out = pd.DataFrame({
            'Candidate': names,
            'Match score': p,
            'Skill match': F['skill_overlap'].values,
            'Experience (yrs)': F['years_exp'].values,
            'Matched skills': [', '.join(sorted(r & js)) for r in rs],
            'Missing skills': [', '.join(sorted(js - r)) for r in rs],
        })
        out['Recommendation'] = np.where(out['Match score'] >= 0.5, 'Recommend', 'Not recommended')
        out = out.sort_values('Match score', ascending=False).reset_index(drop=True)
        out.insert(0, 'Rank', out.index + 1)
        return out, sorted(js)
