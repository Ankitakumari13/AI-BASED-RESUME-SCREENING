"""
Synthetic dataset generator: AI-Based Resume Screening and Candidate Shortlisting
-------------------------------------------------------------------------------
Each row = one candidate resume screened against one job description.
The Decision (select/reject) comes from a hidden, weighted match score
(required skills, preferred skills, experience, education, projects, certifications, role fit)
plus random noise, so the label is learnable but not trivial.

Run:  python generate_dataset.py            -> resume_screening_dataset.csv
"""
import random, re
import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 10000
N_JDS = 1500
SELECT_RATE = 0.45          # approx share of 'select'
NOISE_SD = 0.07             # label noise (higher = harder task)

rng = random.Random(SEED)
nprng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------- roles & skills
ROLES = {
 'Data Scientist': dict(core=['python','sql','machine learning','statistics','pandas','scikit-learn','data visualization','deep learning','nlp','tensorflow'],
                        nice=['pytorch','spark','aws','tableau','a/b testing','docker']),
 'Machine Learning Engineer': dict(core=['python','machine learning','deep learning','tensorflow','pytorch','docker','mlops','sql','scikit-learn','aws'],
                        nice=['kubernetes','spark','nlp','computer vision','airflow','git']),
 'Data Analyst': dict(core=['sql','excel','power bi','tableau','python','statistics','data visualization','data cleaning'],
                        nice=['pandas','a/b testing','looker','r','google analytics']),
 'Data Engineer': dict(core=['python','sql','spark','etl','airflow','kafka','aws','data warehousing','hadoop'],
                        nice=['docker','snowflake','scala','terraform','git']),
 'Backend Developer': dict(core=['java','python','sql','rest apis','microservices','docker','git','spring boot','postgresql'],
                        nice=['kubernetes','redis','aws','kafka','mongodb']),
 'Frontend Developer': dict(core=['javascript','react','html','css','typescript','git','rest apis','responsive design'],
                        nice=['redux','next.js','webpack','jest','figma']),
 'Full Stack Developer': dict(core=['javascript','react','node.js','sql','mongodb','rest apis','git','html','css'],
                        nice=['typescript','docker','aws','graphql','redis']),
 'DevOps Engineer': dict(core=['linux','docker','kubernetes','ci/cd','jenkins','terraform','aws','git','bash','monitoring'],
                        nice=['ansible','python','prometheus','azure','helm']),
 'Cloud Engineer': dict(core=['aws','azure','terraform','linux','networking','docker','kubernetes','security','python'],
                        nice=['gcp','ansible','ci/cd','cloudformation','serverless']),
 'QA Engineer': dict(core=['manual testing','selenium','test automation','jira','sql','api testing','python','test cases'],
                        nice=['postman','jenkins','cypress','agile','performance testing']),
 'Cybersecurity Analyst': dict(core=['network security','siem','incident response','linux','penetration testing','firewalls','risk assessment','python'],
                        nice=['splunk','owasp','cloud security','iso 27001','wireshark']),
 'UI/UX Designer': dict(core=['figma','user research','wireframing','prototyping','design systems','usability testing','adobe xd'],
                        nice=['html','css','accessibility','motion design','sketch']),
 'Business Analyst': dict(core=['requirements gathering','sql','excel','power bi','stakeholder management','jira','process modeling','agile'],
                        nice=['tableau','python','brd','uml','confluence']),
 'Digital Marketing Specialist': dict(core=['seo','google analytics','content marketing','social media marketing','google ads','email marketing','copywriting'],
                        nice=['hubspot','a/b testing','canva','sem','crm']),
 'HR Specialist': dict(core=['recruitment','onboarding','employee relations','hris','payroll','talent acquisition','performance management'],
                        nice=['labor law','sourcing','workday','training and development','excel']),
}
SYNONYMS = {  # alternative ways a resume may write a skill (makes pure keyword matching imperfect)
 'machine learning': ['ML','machine learning'], 'scikit-learn': ['sklearn','scikit-learn'],
 'natural language processing': ['NLP'], 'nlp': ['NLP','natural language processing'],
 'power bi': ['Power BI','PowerBI'], 'javascript': ['JavaScript','JS'], 'node.js': ['Node.js','NodeJS'],
 'rest apis': ['REST APIs','RESTful services'], 'ci/cd': ['CI/CD','CI/CD pipelines'], 'postgresql': ['PostgreSQL','Postgres'],
 'kubernetes': ['Kubernetes','K8s'], 'deep learning': ['deep learning','neural networks'],
 'test automation': ['test automation','automation testing'], 'google analytics': ['Google Analytics','GA4'],
 'aws': ['AWS','Amazon Web Services'], 'sql': ['SQL','SQL'], 'seo': ['SEO','search engine optimization'],
 'hris': ['HRIS','HR information systems'], 'siem': ['SIEM','SIEM tools'], 'etl': ['ETL','ETL pipelines'],
}
def surface(skill):
    opts = SYNONYMS.get(skill)
    if opts:
        return rng.choice(opts)
    return skill.title() if len(skill) > 4 and ' ' in skill else skill

EDU_LEVELS = {'Diploma': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
DEGREES = {
  'tech': ["B.Tech in Computer Science","B.E. in Information Technology","B.Sc in Computer Science","BCA","M.Tech in Computer Science","M.Sc in Data Science","MCA","MBA (Analytics)","PhD in Computer Science"],
  'biz':  ["BBA","B.Com","MBA (HR)","MBA (Marketing)","BA in Psychology","B.Des","M.Des","B.A. in Mass Communication"],
}
COMPANIES = ['Nexora Systems','BlueOrbit Technologies','Quantiva Labs','Zenith Analytics','Pixelwave Solutions','Cloudspire','Infranova','DataBridge Consulting',
             'Skyline Digital','Orion Software','BrightPath Services','Vertex Innovations','Lumina Tech','Apex Dynamics','CoreStack','Trident Infotech']
FIRST = ['Aarav','Priya','Rohan','Ananya','Vikram','Sneha','Arjun','Kavya','Rahul','Meera','Karan','Isha','Aditya','Neha','Siddharth','Pooja','Manish','Divya','Nikhil','Riya',
         'James','Emily','Daniel','Sophia','Michael','Olivia','David','Emma','Liam','Ava']
LAST = ['Sharma','Verma','Gupta','Singh','Patel','Reddy','Nair','Iyer','Mehta','Joshi','Kapoor','Malhotra','Bose','Chopra','Agarwal','Khan','Das','Rao','Smith','Johnson','Brown','Taylor','Wilson','Clark']
PROJECT_NOUNS = ['Recommendation Engine','Customer Churn Dashboard','Inventory Tracker','Sentiment Analyzer','Sales Forecasting Tool','Chat Support Bot','Fraud Detection Pipeline',
                 'Portfolio Website','Task Management App','Log Monitoring System','Attendance Portal','Survey Analytics Tool','Booking Platform','Campaign Performance Report']
CERTS = ['AWS Certified Cloud Practitioner','Google Data Analytics Certificate','Microsoft Azure Fundamentals','Certified Scrum Master','IBM Data Science Professional Certificate',
         'Coursera Machine Learning Specialization','CompTIA Security+','Google UX Design Certificate','Meta Front-End Developer Certificate','HubSpot Inbound Marketing','PMI-PBA','SHRM-CP']
ROLE_CERTS = {
 'Data Scientist': ['IBM Data Science Professional Certificate','Coursera Machine Learning Specialization','Google Data Analytics Certificate'],
 'Machine Learning Engineer': ['Coursera Machine Learning Specialization','IBM Data Science Professional Certificate','AWS Certified Cloud Practitioner'],
 'Data Analyst': ['Google Data Analytics Certificate','IBM Data Science Professional Certificate','Microsoft Azure Fundamentals'],
 'Data Engineer': ['AWS Certified Cloud Practitioner','Microsoft Azure Fundamentals','IBM Data Science Professional Certificate'],
 'Backend Developer': ['AWS Certified Cloud Practitioner','Microsoft Azure Fundamentals','Certified Scrum Master'],
 'Frontend Developer': ['Meta Front-End Developer Certificate','Google UX Design Certificate','Certified Scrum Master'],
 'Full Stack Developer': ['Meta Front-End Developer Certificate','AWS Certified Cloud Practitioner','Certified Scrum Master'],
 'DevOps Engineer': ['AWS Certified Cloud Practitioner','Microsoft Azure Fundamentals','CompTIA Security+'],
 'Cloud Engineer': ['AWS Certified Cloud Practitioner','Microsoft Azure Fundamentals','CompTIA Security+'],
 'QA Engineer': ['Certified Scrum Master','AWS Certified Cloud Practitioner','Microsoft Azure Fundamentals'],
 'Cybersecurity Analyst': ['CompTIA Security+','AWS Certified Cloud Practitioner','Microsoft Azure Fundamentals'],
 'UI/UX Designer': ['Google UX Design Certificate','Meta Front-End Developer Certificate','Certified Scrum Master'],
 'Business Analyst': ['PMI-PBA','Certified Scrum Master','Google Data Analytics Certificate'],
 'Digital Marketing Specialist': ['HubSpot Inbound Marketing','Google Data Analytics Certificate','Certified Scrum Master'],
 'HR Specialist': ['SHRM-CP','Certified Scrum Master','HubSpot Inbound Marketing'],
}
TECH_ROLES = {r for r in ROLES if r not in ('UI/UX Designer','Digital Marketing Specialist','HR Specialist','Business Analyst')}

# ----------------------------------------------------------------------------- job descriptions
JD_TEMPLATES = [
 "We are hiring a {role} to join our team. You will work with {req_txt}. Preferred: {pref_txt}. Minimum {yrs} of relevant experience and a {edu} degree (or equivalent) are required.",
 "{role} opening: we are looking for a candidate with hands-on experience in {req_txt}. Familiarity with {pref_txt} is a plus. Requirements: {yrs} of experience, {edu} degree.",
 "Join us as a {role}. Key skills: {req_txt}. Nice to have: {pref_txt}. You should bring {yrs} of industry experience and hold at least a {edu} degree.",
 "Our growing company needs a {role} skilled in {req_txt}. Bonus points for {pref_txt}. Candidate must have {yrs} of experience and a {edu} qualification.",
]
def make_jd(role):
    spec = ROLES[role]
    k = rng.randint(4, min(7, len(spec['core'])))
    req = rng.sample(spec['core'], k)
    pref = rng.sample(spec['nice'], rng.randint(1, 3))
    min_years = rng.choice([0, 1, 2, 3, 3, 4, 5])
    edu = rng.choice(["Bachelor's", "Bachelor's", "Bachelor's", "Master's", 'Diploma'])
    yrs_txt = 'fresher-level (0-1 years)' if min_years == 0 else f"{min_years}+ years"
    text = rng.choice(JD_TEMPLATES).format(role=role, req_txt=', '.join(surface(s) for s in req),
                                           pref_txt=', '.join(surface(s) for s in pref), yrs=yrs_txt, edu=edu)
    return dict(role=role, req=req, pref=pref, min_years=min_years, edu=edu, text=text)

# ----------------------------------------------------------------------------- candidates
def make_candidate(jd):
    # candidate usually applies in the same domain; sometimes a different one
    same = rng.random() < 0.8
    target = jd['role'] if same else rng.choice(list(ROLES))
    spec = ROLES[target]
    ability = float(np.clip(nprng.beta(2.2, 2.0), 0.05, 0.98))       # latent quality
    years = int(np.clip(round(nprng.gamma(2.2, 1.9) * (0.6 + ability)), 0, 15))
    skills = set()
    for s in spec['core']:
        if rng.random() < 0.15 + 0.8 * ability: skills.add(s)
    for s in spec['nice']:
        if rng.random() < 0.10 + 0.6 * ability: skills.add(s)
    # a few skills from the JD itself even if domain differs (transferable skills)
    for s in jd['req']:
        if s not in skills and rng.random() < 0.08 + 0.3 * ability: skills.add(s)
    if not skills: skills.add(rng.choice(spec['core']))
    tech = target in TECH_ROLES
    edu_pool = DEGREES['tech'] if tech else DEGREES['biz']
    degree = rng.choice(edu_pool)
    if degree.startswith(('PhD',)): lvl = 'PhD'
    elif degree.startswith(('M.','MBA','MCA')): lvl = "Master's"
    elif rng.random() < 0.05: lvl, degree = 'Diploma', 'Diploma in ' + ('Computer Engineering' if tech else 'Business Studies')
    else: lvl = "Bachelor's"
    projects = int(np.clip(round(nprng.normal(1 + 3 * ability, 1.1)), 0, 6))
    n_cert = int(np.clip(round(nprng.normal(2 * ability, 0.9)), 0, 3))
    return dict(target=target, ability=ability, years=years, skills=sorted(skills), degree=degree, edu=lvl,
                projects=projects, n_cert=n_cert)

# ----------------------------------------------------------------------------- resume text
BULLETS = [
 "Used {s} to deliver measurable improvements, raising team efficiency by {n}%.",
 "Designed and maintained solutions with {s}, supporting {n}+ internal users.",
 "Led a module built on {s}, reducing turnaround time by {n}%.",
 "Collaborated with cross-functional teams applying {s} across {n} projects.",
 "Automated recurring tasks using {s}, saving about {n} hours per week.",
]
SUMMARY = [
 "{role} with {yrs} of experience delivering results using {top}.",
 "Detail-oriented {role} bringing {yrs} of experience in {top}.",
 "Results-driven {role} skilled in {top}, with {yrs} of professional experience.",
]
def make_resume(c, name):
    role = c['target']
    yrs_txt = 'fresher-level' if c['years'] == 0 else (f"{c['years']} year" + ('s' if c['years'] != 1 else ''))
    skills_s = [surface(s) for s in c['skills']]
    rng.shuffle(skills_s)
    top = ', '.join(skills_s[:3])
    email = f"{name.lower().replace(' ', '.')}@example.com"
    lines = [name, role, f"Email: {email} | Phone: +91-9{rng.randint(100000000, 999999999)} | linkedin.com/in/{name.lower().replace(' ','-')}", '',
             'SUMMARY', rng.choice(SUMMARY).format(role=role, yrs=yrs_txt, top=top), '', 'SKILLS', ', '.join(skills_s), '']
    if c['years'] > 0:
        lines.append('EXPERIENCE')
        n_jobs = 1 if c['years'] < 4 else 2
        left = c['years']
        for j in range(n_jobs):
            d = left if j == n_jobs - 1 else max(1, left // 2)
            left -= d
            title = role if j == 0 else 'Junior ' + role
            lines.append(f"{title}, {rng.choice(COMPANIES)} ({d} year{'s' if d != 1 else ''})")
            for s in rng.sample(skills_s, min(len(skills_s), rng.randint(2, 3))):
                lines.append('- ' + rng.choice(BULLETS).format(s=s, n=rng.randint(10, 60)))
        lines.append('')
    else:
        lines += ['EXPERIENCE', 'Fresher - no full-time professional experience.', '']
    lines += ['EDUCATION', f"{c['degree']} ({2024 - c['years'] - rng.randint(0, 1)})", '']
    if c['projects']:
        lines.append('PROJECTS')
        for _ in range(c['projects']):
            sk = rng.choice(skills_s)
            lines.append(f"- {rng.choice(PROJECT_NOUNS)}: built using {sk}.")
        lines.append('')
    if c['n_cert']:
        lines.append('CERTIFICATIONS')
        pool = ROLE_CERTS.get(role, CERTS)
        for cert in rng.sample(pool, min(c['n_cert'], len(pool))): lines.append('- ' + cert)
    return '\n'.join(lines).strip()

# ----------------------------------------------------------------------------- hidden scoring
def canon(skills):
    return set(skills)

def match_score(jd, c):
    cs = canon(c['skills'])
    req = len(cs & set(jd['req'])) / len(jd['req'])
    pref = len(cs & set(jd['pref'])) / len(jd['pref'])
    exp = 1.0 if c['years'] >= jd['min_years'] else c['years'] / max(jd['min_years'], 1) * 0.8
    edu = 1.0 if EDU_LEVELS[c['edu']] >= EDU_LEVELS[jd['edu']] else 0.4
    proj = min(c['projects'], 4) / 4
    cert = min(c['n_cert'], 2) / 2
    role_fit = 1.0 if c['target'] == jd['role'] else 0.0
    score = 0.42*req + 0.08*pref + 0.18*exp + 0.08*edu + 0.07*proj + 0.04*cert + 0.13*role_fit
    return score, dict(req=req, pref=pref, exp=exp, edu=edu, proj=proj, cert=cert, role=role_fit)

def reason(decision, jd, c, parts):
    miss = [s for s in jd['req'] if s not in c['skills']]
    hit = [s for s in jd['req'] if s in c['skills']]
    if decision == 'select':
        bits = [f"Matches {len(hit)} of {len(jd['req'])} required skills"]
        if c['years'] >= jd['min_years']: bits.append('meets the experience requirement')
        if parts['role'] == 1: bits.append('relevant role background')
        if c['projects'] >= 3: bits.append('strong project portfolio')
        return '; '.join(bits).capitalize() + '.'
    bits = []
    if miss: bits.append('Missing required skills: ' + ', '.join(miss[:3]))
    if c['years'] < jd['min_years']: bits.append(f"below the {jd['min_years']}+ years experience requirement")
    if parts['role'] == 0: bits.append('background not aligned with the role')
    if parts['edu'] < 1: bits.append('education below requirement')
    if not bits: bits.append('Overall profile weaker than other shortlisted candidates')
    return '; '.join(bits).capitalize() + '.'

# ----------------------------------------------------------------------------- build
def main():
    jds = [make_jd(rng.choice(list(ROLES))) for _ in range(N_JDS)]
    rows = []
    for i in range(N_ROWS):
        jd = jds[i % N_JDS]
        c = make_candidate(jd)
        name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        score, parts = match_score(jd, c)
        rows.append(dict(Candidate_ID=f"C{i+1:05d}", Role=jd['role'], Resume=make_resume(c, name), Job_Description=jd['text'],
                         Years_Experience=c['years'], Education_Level=c['edu'], Num_Projects=c['projects'], Num_Certifications=c['n_cert'],
                         _score=score + nprng.normal(0, NOISE_SD), _jd=jd, _c=c, _parts=parts))
    cut = np.quantile([r['_score'] for r in rows], 1 - SELECT_RATE)
    out = []
    for r in rows:
        dec = 'select' if r['_score'] >= cut else 'reject'
        out.append({k: v for k, v in r.items() if not k.startswith('_')} |
                   {'Decision': dec, 'Reason_for_decision': reason(dec, r['_jd'], r['_c'], r['_parts'])})
    df = pd.DataFrame(out)
    df = df[['Candidate_ID','Role','Resume','Job_Description','Years_Experience','Education_Level','Num_Projects','Num_Certifications','Decision','Reason_for_decision']]
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv('resume_screening_dataset.csv', index=False)
    print(df.shape, df.Decision.value_counts(normalize=True).round(3).to_dict())

if __name__ == '__main__':
    main()
