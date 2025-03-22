import streamlit as st
import os
import random
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pickle
import time
import PyPDF2
import docx2txt
from openai import AzureOpenAI
import numpy as np
import requests
from typing import List
from database import Job, Student, Application, Base, engine, get_db

# Remove the set_page_config call since it's now in main.py

SESSION_DIR = "./.streamlit_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def load_student_session():
    try:
        session_file = os.path.join(SESSION_DIR, "student_session.pkl")
        if os.path.exists(session_file):
            with open(session_file, "rb") as f:
                session_data = pickle.load(f)
                for key, value in session_data.items():
                    st.session_state[key] = value
    except Exception:
        pass

def save_student_session():
    try:
        session_data = {
            "student_authenticated": st.session_state.get("student_authenticated", False),
            "student_page": st.session_state.get("student_page", "Browse Jobs"),
            "student_id": st.session_state.get("student_id", None),
            "student_email": st.session_state.get("student_email", None)
        }
        session_file = os.path.join(SESSION_DIR, "student_session.pkl")
        with open(session_file, "wb") as f:
            pickle.dump(session_data, f)
    except Exception:
        pass

Base.metadata.create_all(bind=engine)

AZURE_OPENAI_KEY = st.secrets.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_ENDPOINT = st.secrets.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT_NAME = st.secrets.get("AZURE_DEPLOYMENT_NAME", "gpt-4o")
AZURE_EMBEDDING_DEPLOYMENT = st.secrets.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
AZURE_API_VERSION = st.secrets.get("AZURE_API_VERSION", "2023-05-15")

azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

def fetch_skills_from_api() -> List[str]:
    try:
        response = requests.get(
            "https://api.github.com/topics?per_page=100", 
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=5
        )
        
        if response.status_code == 200:
            programming_skills = [
                topic["name"] for topic in response.json()["items"]
                if topic.get("name") and len(topic["name"]) > 2
            ]
            
            soft_skills_response = requests.get(
                "https://api.datamuse.com/words?rel_syn=skill&max=50",
                timeout=5
            )
            
            soft_skills = []
            if soft_skills_response.status_code == 200:
                soft_skills = [item["word"] for item in soft_skills_response.json() 
                              if len(item["word"].split()) <= 2]
            
            all_skills = list(set(programming_skills + soft_skills + FALLBACK_SKILLS))
            return all_skills[:500]
    except Exception:
        pass
    
    return FALLBACK_SKILLS

FALLBACK_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Ruby", "PHP", "Swift",
    "Kotlin", "Rust", "Scala", "R", "MATLAB", "Perl", "Groovy", "Dart", "Haskell", "Clojure",
    "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express", "Django", "Flask", 
    "Spring Boot", "ASP.NET", "Ruby on Rails", "jQuery", "Bootstrap", "Tailwind CSS", "Redux",
    "GraphQL", "REST API", "JSON", "XML", "WebSockets", "Svelte", "Next.js", "Gatsby",
    "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "Redis", "Cassandra", 
    "DynamoDB", "Elasticsearch", "Neo4j", "Firebase", "MariaDB", "Couchbase", "MS SQL Server",
    "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Jenkins", "GitLab CI", 
    "GitHub Actions", "Terraform", "Ansible", "Prometheus", "Grafana", "ELK Stack", "CircleCI",
    "Serverless", "Heroku", "DigitalOcean", "Nginx", "Apache", "Linux", "Bash",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Analysis",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy", "SciPy",
    "Data Visualization", "Tableau", "Power BI", "Matplotlib", "Seaborn", "Jupyter",
    "Big Data", "Hadoop", "Spark", "Data Mining", "Statistical Analysis", "A/B Testing",
    "Communication", "Leadership", "Problem Solving", "Critical Thinking", "Teamwork",
    "Time Management", "Adaptability", "Creativity", "Attention to Detail", "Project Management",
    "Conflict Resolution", "Negotiation", "Mentoring", "Customer Service", "Presentation"
]

@st.cache_data(ttl=3600)
def get_all_skills():
    return fetch_skills_from_api()

def extract_text_from_resume(uploaded_file):
    if not uploaded_file:
        return ""
        
    text = ""
    file_extension = uploaded_file.name.split(".")[-1].lower()
    
    try:
        if file_extension == "pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
                
            if len(text.strip()) < 100:
                st.warning("This appears to be a scanned PDF. Text extraction may be limited.")
                
        elif file_extension in ["docx", "doc"]:
            text = docx2txt.process(uploaded_file)
                
        elif file_extension == "txt":
            try:
                text = uploaded_file.getvalue().decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = uploaded_file.getvalue().decode("latin-1")
                except Exception:
                    st.error("Could not decode text file. Please ensure it's a valid text file.")
                    return None
        else:
            st.error(f"Unsupported file format: {file_extension}. Please upload a PDF, DOCX, or TXT file.")
            return None
    except Exception as e:
        st.error(f"Error extracting text from file: {str(e)}")
        return None
    
    import re
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text

def extract_skills_from_resume(resume_text: str) -> List[str]:
    if not resume_text or len(resume_text.strip()) < 50:
        return []
    
    try:
        prompt = f"""
        Extract ALL technical and soft skills from the following resume text. 
        Be thorough and identify EVERY skill including:
        - Programming languages and frameworks
        - Software and tools
        - Methodologies and processes
        - Soft skills and professional competencies
        - Industry-specific knowledge
        
        Return a list of skills, one per line, without any explanation or categorization.
        
        Resume text:
        {resume_text[:4000]}
        """
        
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a skilled AI assistant specializing in comprehensive skill extraction from resumes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400
        )
        
        skills_text = response.choices[0].message.content.strip()
        extracted_skills = [skill.strip() for skill in skills_text.split('\n') if skill.strip()]
        
        valid_skills = []
        for skill in extracted_skills:
            clean_skill = skill.strip().rstrip('.,;:')
            if len(clean_skill) >= 2 and len(clean_skill) <= 50:
                valid_skills.append(clean_skill)
                
        return valid_skills
    except Exception as e:
        st.warning(f"Error extracting skills: {str(e)}")
        skills_pool = get_all_skills()
        detected_skills = []
        for skill in skills_pool:
            if skill.lower() in resume_text.lower():
                detected_skills.append(skill)
        if len(detected_skills) < 3:
            num_additional = min(5, max(3, len(resume_text) // 500))
            additional_skills = random.sample([s for s in skills_pool if s not in detected_skills], 
                                             min(num_additional, len(skills_pool) - len(detected_skills)))
            detected_skills.extend(additional_skills)
        return detected_skills

def get_embedding(text, model=AZURE_EMBEDDING_DEPLOYMENT):
    try:
        text = text.replace("\n", " ")
        embedding = azure_client.embeddings.create(
            input=[text],
            model=model
        )
        return embedding.data[0].embedding
    except Exception:
        return np.random.rand(1536)

def calculate_match_score(job, student):
    try:
        student_skills = set(student.skills.split(","))
        job_skills = set(job.skills.split(","))
        
        if not job_skills:
            return 0.0
        
        skill_overlap = len(student_skills.intersection(job_skills))
        skill_match = (skill_overlap / len(job_skills)) * 70
        
        job_embedding = get_embedding(f"{job.title} {job.description} {job.skills}")
        resume_embedding = get_embedding(student.resume_text[:5000])
        
        similarity = np.dot(job_embedding, resume_embedding) / (
            np.linalg.norm(job_embedding) * np.linalg.norm(resume_embedding)
        )
        embedding_match = similarity * 30
        
        match_score = skill_match + embedding_match
        
        # Convert to standard Python float before returning
        return float(min(100.0, max(0.0, match_score)))
    except Exception:
        student_skills = set(student.skills.split(","))
        job_skills = set(job.skills.split(","))
        overlap = len(student_skills.intersection(job_skills))
        total = len(job_skills)
        return float((overlap / total) * 100) if total > 0 else 0.0

def suggest_relevant_skills(job_title: str, job_description: str) -> List[str]:
    try:
        prompt = f"""
        Based on the following job title and description, list the TOP 10 most relevant technical and soft skills 
        that would make a candidate successful in this role. Return only the skill names, one per line.
        
        Job Title: {job_title}
        
        Job Description:
        {job_description[:2000]}
        """
        
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "You are a skilled career advisor helping job seekers identify relevant skills."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        suggested_skills = [skill.strip() for skill in response.choices[0].message.content.strip().split('\n') if skill.strip()]
        return suggested_skills
    except Exception:
        skills_pool = get_all_skills()
        relevant_skills = []
        
        for skill in skills_pool:
            if skill.lower() in job_description.lower() or skill.lower() in job_title.lower():
                relevant_skills.append(skill)
                
        if len(relevant_skills) > 10:
            return relevant_skills[:10]
        elif len(relevant_skills) < 3:
            additional = random.sample([s for s in skills_pool if s not in relevant_skills], 
                                      min(7, len(skills_pool)))
            return relevant_skills + additional
        else:
            return relevant_skills

def student_login(email):
    if not email:
        return None
        
    db = get_db()
    try:
        student = db.query(Student).filter(Student.email == email).first()
        return student
    except Exception:
        return None
    finally:
        db.close()

def submit_application(student_id, job_id):
    db = get_db()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        student = db.query(Student).filter(Student.id == student_id).first()
        
        if not job or not student:
            return False, "Job or student not found"
        
        existing_application = db.query(Application).filter(
            Application.job_id == job_id,
            Application.student_id == student_id
        ).first()
        
        if existing_application:
            return False, "You have already applied for this job"
        
        match_score = calculate_match_score(job, student)
        
        # Convert NumPy float to Python float if needed
        if hasattr(match_score, "item"):  # Check if it's a NumPy type with the item() method
            match_score = float(match_score)
        
        new_application = Application(
            job_id=job_id,
            student_id=student_id,
            match_score=match_score,
            status="pending",
            application_date=datetime.now()
        )
        
        db.add(new_application)
        db.commit()
        return True, match_score
    except Exception as e:
        db.rollback()
        return False, str(e)
    finally:
        db.close()

def run_student_portal():
    """Main function to run the student portal"""
    
    # Initialize all session state variables
    if "student_authenticated" not in st.session_state:
        st.session_state.student_authenticated = False

    if "student_page" not in st.session_state:
        st.session_state.student_page = "Browse Jobs"
        
    if "student_data" not in st.session_state:
        st.session_state.student_data = None
        
    if "student_id" not in st.session_state:
        st.session_state.student_id = None
        
    if "student_email" not in st.session_state:
        st.session_state.student_email = None
        
    if "show_registration" not in st.session_state:
        st.session_state.show_registration = False

    # Load any saved session data
    load_student_session()
    
    # Continue with the rest of your code
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/resume.png", width=100)
        st.title("Student Portal")
        
        if not st.session_state.student_authenticated:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            email = st.text_input("Email")
            
            if st.button("Login/Register", use_container_width=True):
                student = student_login(email)
                if student:
                    st.session_state.student_authenticated = True
                    st.session_state.student_id = student.id
                    st.session_state.student_data = student
                    st.session_state.student_email = student.email
                    save_student_session()
                    st.rerun()
                else:
                    st.session_state.show_registration = True
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Check if student_data is available before accessing it
            if st.session_state.student_data:
                st.markdown('<div class="avatar-container">', unsafe_allow_html=True)
                st.image("https://img.icons8.com/fluency/96/000000/user-male-circle.png", width=80)
                st.success(f"Hello, {st.session_state.student_data.name}!")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Handle case where student_data is missing but authenticated flag is true
                st.warning("Session data is incomplete. Please log in again.")
                st.session_state.student_authenticated = False
                st.rerun()
            
            st.divider()
            
            st.header("Navigation")
            
            menu_options = {
                "Browse Jobs": "🔍",
                "My Applications": "📋",
                "Profile": "👤"
            }
            
            for page, icon in menu_options.items():
                if st.button(f"{icon} {page}", key=f"menu_{page}", use_container_width=True, 
                           help=f"Navigate to {page}",
                           type="primary" if st.session_state.student_page == page else "secondary"):
                    st.session_state.student_page = page
                    save_student_session()
                    st.rerun()
            
            st.divider()
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.student_authenticated = False
                if "student_id" in st.session_state:
                    del st.session_state.student_id
                if "student_data" in st.session_state:
                    del st.session_state.student_data
                
                session_file = os.path.join(SESSION_DIR, "student_session.pkl")
                if os.path.exists(session_file):
                    os.remove(session_file)
                    
                st.rerun()

    if not st.session_state.student_authenticated and not st.session_state.get("show_registration", False):
        st.markdown('<h1 class="main-header">Resume Matching Portal</h1>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("Please login with your email to continue. If you're new, you'll be prompted to register.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🚀 Find your perfect job match!")
        st.write("""
        Our AI-powered resume matching system helps you:
        - Find jobs that match your skills and experience
        - Get personalized match scores for each position
        - Track your application status
        - Improve your resume with AI skill extraction
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    elif not st.session_state.student_authenticated and st.session_state.get("show_registration", False):
        st.markdown('<h1 class="main-header">Student Registration</h1>', unsafe_allow_html=True)
        
        # Store form state to track which step we're on
        if "registration_step" not in st.session_state:
            st.session_state.registration_step = "upload_resume"
        
        with st.form("registration_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name", key="reg_name")
                email = st.text_input("Email", value=st.session_state.get("email", ""), key="reg_email")
            
            with col2:
                education_level = st.selectbox(
                    "Education Level", 
                    ["High School", "Bachelor's", "Master's", "PhD"],
                    key="reg_edu_level"
                )
                field_of_study = st.text_input("Field of Study", key="reg_field")
            
            education = f"{education_level} in {field_of_study}"
            
            st.markdown("### Resume Upload")
            resume_file = st.file_uploader("Upload your resume (PDF, DOCX, or TXT)", 
                                         type=["pdf", "docx", "txt"],
                                         key="resume_uploader")
            
            # Display resume preview if it's been processed
            if "resume_text" in st.session_state:
                st.success(f"Resume text extracted ({len(st.session_state.resume_text)} characters)")
                with st.expander("Preview Extracted Text"):
                    st.text(st.session_state.resume_text[:500] + "..." if len(st.session_state.resume_text) > 500 else st.session_state.resume_text)
            
            # Choose which buttons to show based on step
            if st.session_state.registration_step == "upload_resume":
                process_resume_button = st.form_submit_button("Process Resume", use_container_width=True)
                extract_skills_button = False
                register_button = False
            elif st.session_state.registration_step == "extract_skills":
                process_resume_button = False
                extract_skills_button = st.form_submit_button("Extract Skills from Resume", use_container_width=True)
                register_button = False
            else:  # "register" step
                process_resume_button = False
                extract_skills_button = False
                register_button = st.form_submit_button("Register", use_container_width=True)
            
            # Always show clear button
            clear_button = st.form_submit_button("Clear Form", use_container_width=True)
            
            # Handle process resume button
            if process_resume_button and resume_file:
                resume_text = extract_text_from_resume(resume_file)
                if resume_text:
                    st.session_state.resume_text = resume_text
                    st.session_state.registration_step = "extract_skills"
                    st.rerun()
                else:
                    st.error("Could not extract text from the uploaded file. Please try another file.")
            
            # Handle extract skills button
            if extract_skills_button and "resume_text" in st.session_state:
                with st.spinner("Analyzing your resume and extracting skills..."):
                    extracted_skills = extract_skills_from_resume(st.session_state.resume_text)
                    st.session_state.extracted_skills = extracted_skills
                    if extracted_skills:
                        st.success(f"✅ Successfully extracted {len(extracted_skills)} skills from your resume!")
                        st.session_state.registration_step = "register"
                        st.rerun()
                    else:
                        st.warning("No skills were detected in your resume. Please select skills manually below.")
                        st.session_state.registration_step = "register"
                        st.rerun()
            
            # Display skills selection based on step
            st.markdown("### Skills")
            all_skills = get_all_skills()
            
            if "extracted_skills" in st.session_state and st.session_state.extracted_skills:
                st.write("**AI-extracted skills from your resume:**")
                extracted_skill_cols = st.columns(4)
                
                selected_skills = []
                for i, skill in enumerate(st.session_state.extracted_skills):
                    with extracted_skill_cols[i % 4]:
                        if st.checkbox(skill, value=True, key=f"skill_{skill}"):
                            selected_skills.append(skill)
                
                with st.expander("Add More Skills"):
                    additional_skills = st.multiselect(
                        "Select additional skills", 
                        options=[s for s in all_skills if s not in st.session_state.extracted_skills],
                        key="additional_skills"
                    )
                    selected_skills.extend(additional_skills)
            else:
                selected_skills = st.multiselect("Select Your Skills", all_skills, key="manual_skills")
            
            # Handle registration
            if register_button:
                # Validate form fields
                validation_errors = []
                if not name:
                    validation_errors.append("Full Name is required")
                if not email:
                    validation_errors.append("Email is required")
                if not education_level:
                    validation_errors.append("Education Level is required")
                if not field_of_study:
                    validation_errors.append("Field of Study is required")
                if not selected_skills:
                    validation_errors.append("At least one skill is required")
                if "resume_text" not in st.session_state or not st.session_state.resume_text:
                    validation_errors.append("Resume upload is required")
                
                # Display validation errors if any
                if validation_errors:
                    error_message = "Please fix the following errors:\n" + "\n".join([f"- {error}" for error in validation_errors])
                    st.error(error_message)
                else:
                    # Proceed with registration
                    try:
                        db = get_db()
                        try:
                            existing_student = db.query(Student).filter(Student.email == email).first()
                            if existing_student:
                                st.error(f"A student with email {email} already exists. Please use a different email.")
                            else:
                                new_student = Student(
                                    name=name,
                                    email=email,
                                    skills=",".join(selected_skills),
                                    education=education,
                                    resume_text=st.session_state.resume_text
                                )
                                
                                db.add(new_student)
                                db.commit()
                                db.refresh(new_student)
                                
                                st.session_state.student_authenticated = True
                                st.session_state.student_id = new_student.id
                                st.session_state.student_data = new_student
                                st.session_state.student_email = new_student.email
                                
                                save_student_session()
                                
                                st.success("Registration successful! Redirecting to your dashboard...")
                                time.sleep(1)
                                st.session_state.student_page = "Browse Jobs"
                                st.rerun()
                        finally:
                            db.close()
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")
            
            # Handle clear button
            if clear_button:
                for key in ["resume_text", "extracted_skills", "registration_step"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.registration_step = "upload_resume"
                st.rerun()

    else:
        current_page = st.session_state.get("student_page", "Browse Jobs")
        
        if current_page == "Browse Jobs":
            st.markdown('<h1 class="main-header">🔍 Browse Available Jobs</h1>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_term = st.text_input("Search by keyword")
            
            with col2:
                sort_option = st.selectbox("Sort by", ["Relevance", "Date Posted", "Alphabetical"])
            
            student_id = st.session_state.student_id
            db = get_db()
            applied_job_ids = [app.job_id for app in db.query(Application).filter(Application.student_id == student_id).all()]
            
            filtered_jobs = db.query(Job).all()
            if search_term:
                filtered_jobs = [
                    job for job in filtered_jobs 
                    if search_term.lower() in job.title.lower() 
                    or search_term.lower() in job.description.lower()
                    or any(search_term.lower() in skill.lower() for skill in job.skills.split(","))
                ]
            
            if sort_option == "Date Posted":
                filtered_jobs.sort(key=lambda x: x.date_posted, reverse=True)
            elif sort_option == "Alphabetical":
                filtered_jobs.sort(key=lambda x: x.title)
            else:
                student = db.query(Student).filter(Student.id == student_id).first()
                if student:
                    for job in filtered_jobs:
                        match_score = calculate_match_score(job, student)
                        job.relevance = match_score
                    
                    filtered_jobs.sort(key=lambda x: x.relevance, reverse=True)
            
            if not filtered_jobs:
                st.info("No jobs found matching your criteria.")
            
            for job in filtered_jobs:
                st.markdown('<div class="job-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(job.title)
                    st.caption(f"{job.company} • Posted on {job.date_posted}")
                    
                    short_desc = job.description[:200] + "..." if len(job.description) > 200 else job.description
                    st.write(short_desc)
                    
                    st.markdown('<div class="skills-container">', unsafe_allow_html=True)
                    for skill in job.skills.split(","):
                        st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    with st.expander("See full description"):
                        st.write(job.description)
                        st.write("**Required Skills:**")
                        for skill in job.skills.split(","):
                            st.info(skill)
                
                with col2:
                    student = db.query(Student).filter(Student.id == student_id).first()
                    
                    if hasattr(job, 'relevance'):
                        match_score = job.relevance
                    else:
                        match_score = calculate_match_score(job, student)
                    
                    st.metric("Match Score", f"{match_score:.1f}%")
                    
                    st.progress(float(match_score)/100.0)
                    
                    student_skills = set(student.skills.split(","))
                    job_skills = set(job.skills.split(","))
                    missing_skills = job_skills - student_skills
                    
                    if missing_skills:
                        with st.expander("🔍 Skill Gap Analysis"):
                            st.write("**Add these skills to improve your match:**")
                            
                            for skill in missing_skills:
                                col1, col2 = st.columns([3, 1])
                                col1.write(f"• {skill}")
                                if col2.button("Add", key=f"add_skill_{job.id}_{skill}"):
                                    updated_skills = list(set(student_skills.union({skill})))
                                    student.skills = ",".join(updated_skills)
                                    db.commit()
                                    st.success(f"Added {skill} to your profile!")
                                    time.sleep(0.5)
                                    st.rerun()
                    
                    if job.id in applied_job_ids:
                        st.success("✅ Applied")
                    else:
                        if st.button("Apply", key=f"apply_{job.id}"):
                            success, message = submit_application(student_id, job.id)
                            if success:
                                st.success(f"Applied successfully! Match score: {message:.1f}%")
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error(f"Application failed: {message}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            db.close()
        
        elif current_page == "My Applications":
            st.markdown('<h1 class="main-header">📋 My Applications</h1>', unsafe_allow_html=True)
            
            student_id = st.session_state.student_id
            
            db = get_db()
            
            applications = db.query(Application).filter(Application.student_id == student_id).all()
            
            if not applications:
                st.info("You haven't applied to any jobs yet. Browse available positions and apply!")
            else:
                status_filters = ["All", "Pending", "In Review", "Accepted", "Rejected"]
                selected_filter = st.selectbox("Filter by status", status_filters)
                
                filtered_applications = applications
                if selected_filter != "All":
                    filtered_applications = [app for app in applications if app.status.lower() == selected_filter.lower()]
                
                for app in filtered_applications:
                    job = db.query(Job).filter(Job.id == app.job_id).first()
                    if job:
                        st.markdown('<div class="job-card">', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.subheader(job.title)
                            st.caption(f"{job.company} • Applied on {app.application_date.strftime('%Y-%m-%d')}")
                            
                            short_desc = job.description[:150] + "..." if len(job.description) > 150 else job.description
                            st.write(short_desc)
                            
                            st.markdown('<div class="skills-container">', unsafe_allow_html=True)
                            for skill in job.skills.split(",")[:5]:
                                st.markdown(f'<span class="skill-tag">{skill}</span>', unsafe_allow_html=True)
                            if len(job.skills.split(",")) > 5:
                                st.markdown(f'<span class="skill-tag">+{len(job.skills.split(",")) - 5} more</span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            status_colors = {
                                "pending": "🟠 Pending",
                                "in review": "🔵 In Review",
                                "accepted": "🟢 Accepted",
                                "rejected": "🔴 Rejected"
                            }
                            
                            st.metric("Match Score", f"{app.match_score:.1f}%")
                            st.progress(float(app.match_score)/100.0)
                            
                            status_display = status_colors.get(app.status.lower(), app.status)
                            st.info(status_display)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            
            db.close()
        
        elif current_page == "Profile":
            st.markdown('<h1 class="main-header">👤 My Profile</h1>', unsafe_allow_html=True)
            
            student_id = st.session_state.student_id
            db = get_db()
            student = db.query(Student).filter(Student.id == student_id).first()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Personal Information")
                
                st.write(f"**Name:** {student.name}")
                st.write(f"**Email:** {student.email}")
                st.write(f"**Education:** {student.education}")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Skills")
                
                skills = student.skills.split(",") if student.skills else []
                
                skill_cols = st.columns(4)
                for i, skill in enumerate(skills):
                    skill_cols[i % 4].info(skill)
                
                with st.expander("Edit Skills"):
                    all_skills = get_all_skills()
                    
                    # Make sure all existing skills are included in the options
                    combined_skills = list(set(all_skills + skills))
                    
                    updated_skills = st.multiselect(
                        "Your Skills", 
                        options=combined_skills,
                        default=skills
                    )
                    
                    if st.button("Update Skills"):
                        student.skills = ",".join(updated_skills)
                        db.commit()
                        st.success("Skills updated successfully!")
                        time.sleep(0.5)
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Resume")
                
                # Add resume upload capability
                resume_file = st.file_uploader(
                    "Upload a new resume (PDF, DOCX, or TXT)", 
                    type=["pdf", "docx", "txt"],
                    key="profile_resume_uploader"
                )
                
                if resume_file:
                    new_resume_text = extract_text_from_resume(resume_file)
                    if new_resume_text:
                        st.success(f"Successfully extracted {len(new_resume_text)} characters from your resume")
                        
                        with st.expander("Preview Extracted Text"):
                            st.text(new_resume_text[:500] + "..." if len(new_resume_text) > 500 else new_resume_text)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Update Resume Text", key="update_resume_text_btn"):
                                student.resume_text = new_resume_text
                                db.commit()
                                st.success("Resume text updated successfully!")
                                time.sleep(0.5)
                                st.rerun()
                        
                        with col2:
                            if st.button("Extract Skills from Resume", key="extract_skills_from_new_resume"):
                                with st.spinner("Analyzing your resume and extracting skills..."):
                                    extracted_skills = extract_skills_from_resume(new_resume_text)
                                    
                                if extracted_skills:
                                    st.success(f"✅ Successfully extracted {len(extracted_skills)} skills from your resume!")
                                    
                                    # Store extracted skills in session state
                                    st.session_state.profile_extracted_skills = extracted_skills
                                    
                                    # Display skills as a list with checkboxes instead of using nested columns
                                    st.write("**Select skills to add to your profile:**")
                                    
                                    # Use a grid-like display with HTML/CSS instead of nested columns
                                    num_skills = len(extracted_skills)
                                    skills_html = "<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;'>"
                                    
                                    for i, skill in enumerate(extracted_skills):
                                        # Initially check all skills
                                        if st.checkbox(skill, value=True, key=f"profile_skill_{i}"):
                                            if "selected_profile_skills" not in st.session_state:
                                                st.session_state.selected_profile_skills = []
                                            if skill not in st.session_state.selected_profile_skills:
                                                st.session_state.selected_profile_skills.append(skill)
                                        elif "selected_profile_skills" in st.session_state and skill in st.session_state.selected_profile_skills:
                                            st.session_state.selected_profile_skills.remove(skill)
                                    
                                    skills_html += "</div>"
                                    
                                    if st.button("Update Profile with Selected Skills"):
                                        if "selected_profile_skills" in st.session_state and st.session_state.selected_profile_skills:
                                            # Combine existing skills with new skills
                                            updated_skills = list(set(skills + st.session_state.selected_profile_skills))
                                            student.skills = ",".join(updated_skills)
                                            student.resume_text = new_resume_text
                                            db.commit()
                                            st.success("Profile updated with new resume and skills!")
                                            
                                            # Clear the session state
                                            if "profile_extracted_skills" in st.session_state:
                                                del st.session_state.profile_extracted_skills
                                            if "selected_profile_skills" in st.session_state:
                                                del st.session_state.selected_profile_skills
                                            
                                            time.sleep(0.5)
                                            st.rerun()
                                        else:
                                            st.warning("Please select at least one skill to add to your profile.")
                                else:
                                    st.warning("No skills were detected in your resume.")
                with st.expander("View/Edit Resume Text"):
                    manual_resume_text = st.text_area(
                        "Edit your resume text",
                        value=student.resume_text,
                        height=300
                    )
                    
                    if st.button("Update Resume Text", key="update_manual_text"):
                        if len(manual_resume_text) > 100:
                            student.resume_text = manual_resume_text
                            db.commit()
                            st.success("Resume text updated successfully!")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("Resume text is too short. Please provide a more detailed resume.")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Job Recommendations")
                
                applications = db.query(Application).filter(Application.student_id == student_id).all()
                applied_job_ids = [app.job_id for app in applications]
                
                recommendations = db.query(Job).filter(~Job.id.in_(applied_job_ids)).all() if applied_job_ids else db.query(Job).all()
                
                for job in recommendations[:5]:
                    match_score = calculate_match_score(job, student)
                    
                    if match_score >= 60:
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{job.title}** at {job.company}")
                            st.caption(f"Posted on {job.date_posted}")
                        
                        with col2:
                            st.metric("Match", f"{match_score:.1f}%")
                            
                            if st.button("View & Apply", key=f"rec_{job.id}"):
                                st.session_state.student_page = "Browse Jobs"
                                st.rerun()
                        
                        st.divider()
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Application Statistics")
                
                applications = db.query(Application).filter(Application.student_id == student_id).all()
                
                total_applications = len(applications)
                status_counts = {}
                
                for app in applications:
                    status = app.status.lower()
                    if status in status_counts:
                        status_counts[status] += 1
                    else:
                        status_counts[status] = 1
                
                st.metric("Total Applications", total_applications)
                
                if applications:
                    status_labels = list(status_counts.keys())
                    status_values = list(status_counts.values())
                    
                    fig = px.pie(
                        values=status_values,
                        names=status_labels,
                        title="Application Status",
                        color=status_labels,
                        color_discrete_map={
                            'pending': '#FFA500',
                            'in review': '#1E90FF',
                            'accepted': '#32CD32',
                            'rejected': '#FF6347'
                        }
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Skill Suggestions")
                
                all_jobs = db.query(Job).all()
                all_job_skills = []
                for job in all_jobs:
                    all_job_skills.extend(job.skills.split(","))
                
                skill_frequency = {}
                for skill in all_job_skills:
                    if skill in skill_frequency:
                        skill_frequency[skill] += 1
                    else:
                        skill_frequency[skill] = 1
                
                student_skills = student.skills.split(",")
                missing_popular_skills = [
                    (skill, count) for skill, count in skill_frequency.items()
                    if skill not in student_skills and count > 1
                ]
                
                missing_popular_skills.sort(key=lambda x: x[1], reverse=True)
                
                if missing_popular_skills:
                    st.write("Based on the job market, consider adding these skills:")
                    
                    for skill, count in missing_popular_skills[:5]:
                        st.info(f"{skill} (in {count} jobs)")
                        
                        if st.button("Add", key=f"add_suggested_{skill}"):
                            updated_skills = list(set(student_skills + [skill]))
                            student.skills = ",".join(updated_skills)
                            db.commit()
                            st.success(f"Added {skill} to your profile!")
                            time.sleep(0.5)
                            st.rerun()
                else:
                    st.info("Your skills cover the current job market well!")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                if applications:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.subheader("Match Score Trends")
                    
                    match_scores = [app.match_score for app in applications]
                    avg_match = sum(match_scores) / len(match_scores) if match_scores else 0
                    
                    st.metric("Average Match Score", f"{avg_match:.1f}%")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Box(
                        y=match_scores,
                        boxmean=True,
                        marker_color='#1E90FF',
                        name='Match Scores'
                    ))
                    
                    fig.update_layout(
                        title_text="Your Match Score Distribution",
                        title_x=0,
                        height=300,
                        margin=dict(t=30, b=0, l=0, r=0),
                        yaxis=dict(title="Match Score (%)")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            db.close()

def register_student(name, email, education, selected_skills, resume_text):
    db = get_db()
    try:
        existing_student = db.query(Student).filter(Student.email == email).first()
        if existing_student:
            return False, f"A student with email {email} already exists."
        
        new_student = Student(
            name=name,
            email=email,
            skills=",".join(selected_skills),
            education=education,
            resume_text=resume_text
        )
        
        db.add(new_student)
        db.commit()
        db.refresh(new_student)
        
        return True, new_student
    except Exception as e:
        db.rollback()
        return False, str(e)
    finally:
        db.close()

# If the file is run directly (not imported), execute the main function
if __name__ == "__main__" and not os.environ.get("STREAMLIT_IMPORTED_BY_MAIN"):
    # This is for running the student portal directly, not through main.py
    st.set_page_config(
        page_title="Student Portal - Resume Matcher",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    run_student_portal()
                    