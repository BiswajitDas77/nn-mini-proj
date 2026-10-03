import re
from pypdf import PdfReader

TECH_SKILLS = ["python", "java", "c++", "c#", "javascript", "sql", "react", "node", "aws", "docker", "machine learning", "data science", "numpy", "pandas", "git"]
SOFT_SKILLS = ["communication", "leadership", "teamwork", "problem solving", "time management", "adaptability", "critical thinking", "agile", "collaboration", "management"]

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text.lower()

def extract_experience(text):
    text_lower = text.lower()
    
    # 1. Look for explicit months (e.g., "1 month", "2 months")
    month_matches = re.findall(r'(\d+)\s*(?:months?|mos?)', text_lower)
    total_months = sum([float(m) for m in month_matches])
    
    # 2. Look for explicit years (e.g., "2 years", "1.5 yrs")
    year_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?)', text_lower)
    total_years = sum([float(y) for y in year_matches])
    
    # Combine (convert months to fractional years)
    calculated_exp = total_years + (total_months / 12.0)
    
    # If nothing found, return entry-level 0.2 (~2 months), not 2.0
    return round(calculated_exp, 2) if calculated_exp > 0 else 0.2

def parse_resume(pdf_file):
    """
    Parses a resume PDF and extracts:
    - Experience (Years)
    - Tech Score (0-100)
    - Soft Score (0-100)
    - Projects (Count)
    Returns a dictionary of these features.
    """
    text = extract_text_from_pdf(pdf_file)
    
    # 1. Experience Extraction
    exp = min(extract_experience(text), 10.0)
        
    # 2. Tech Score Extraction
    tech_count = 0
    for skill in TECH_SKILLS:
        if skill in text:
            tech_count += 1
    # Max tech skills ~10. Scale to 100
    tech_score = min(tech_count * 10, 100)
    if tech_score == 0:
        tech_score = 30.0 # Fallback
        
    # 3. Soft Skills Extraction
    soft_count = 0
    for skill in SOFT_SKILLS:
        if skill in text:
            soft_count += 1
    soft_score = min(soft_count * 15, 100)
    if soft_score == 0:
        soft_score = 40.0 # Fallback
        
    # 4. Projects Extraction
    # Count occurrences of words like "project", "developed", "built", "created"
    proj_keywords = ["project", "developed", "built", "created", "designed", "led"]
    proj_count = 0
    for word in proj_keywords:
        proj_count += len(re.findall(r'\b' + word + r'\b', text))
    
    # Estimate projects based on keywords (roughly divide by 2)
    projects = min(proj_count // 2, 6)
    if projects == 0:
        projects = 1 # Fallback
        
    return {
        "Experience": float(exp),
        "Tech_Score": float(tech_score),
        "Soft_Score": float(soft_score),
        "Projects": int(projects)
    }

if __name__ == '__main__':
    # Simple test (assuming you have a test.pdf)
    import sys
    if len(sys.argv) > 1:
        print(parse_resume(sys.argv[1]))
    else:
        print("Run with a PDF file to test parsing: python resume_parser.py <file.pdf>")
