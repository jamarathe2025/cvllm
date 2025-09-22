EXTRACTION_PROMPT = """
You are an expert HR assistant. Extract the following JSON fields from the provided resume text.
Fields:
- name
- contact: email, phone, location, linkedin, github, website
- summary
- skills (list)
- education (list of {institution, degree, field, start_date, end_date, gpa})
- experience (list of {title, company, start_date, end_date, bullets, technologies})
- projects (list of {name, description, bullets, technologies, link})
- certifications (list)
- awards (list)
- publications (list)

Return ONLY valid JSON. If unsure, use nulls or empty arrays. Do not invent facts.
Resume Text:
----------------
{resume_text}
""".strip()


JD_PARSING_PROMPT = """
You are an expert recruiter. Parse this job description and return JSON with:
- title
- company
- location
- responsibilities (list)
- requirements (list)
- nice_to_have (list)
- keywords (list)  # list of role-specific keywords and skills
Return ONLY JSON.

Job Description:
----------------
{job_text}
""".strip()


TAILORING_PROMPT = """
You are an elite resume writer optimizing for ATS. Given structured resume data and a parsed job description, produce a tailored resume in Markdown.
Instructions:
- Emphasize impact and quantified achievements.
- Match role keywords naturally; avoid keyword stuffing.
- Keep concise, strong bullet points (STAR style where possible).
- Reorder experience/projects to highlight role fit.
- Tone: {tone}. Target seniority: {seniority}. Target role: {role}.
- Length: {length} (prefer 1 page unless critical experience requires otherwise).
- Sections: Header (name + contact), Summary, Skills, Experience, Projects (if relevant), Education, Certifications.

Return ONLY the Markdown for the final resume.

Structured Resume JSON:
{resume_json}

Parsed Job Description JSON:
{jd_json}
""".strip()
