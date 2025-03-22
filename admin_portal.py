import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import pickle
from database import SessionLocal, Job, Student, Application, Base, engine
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

SESSION_DIR = "./.streamlit_sessions"
os.makedirs(SESSION_DIR, exist_ok=True)

def load_session():
    try:
        session_file = os.path.join(SESSION_DIR, "admin_session.pkl")
        if os.path.exists(session_file):
            with open(session_file, "rb") as f:
                session_data = pickle.load(f)
                # Debug info
                st.sidebar.markdown("**Debug: Session loaded**")
                for key, value in session_data.items():
                    st.session_state[key] = value
    except Exception as e:
        st.sidebar.error(f"Error loading session: {str(e)}")

def save_session():
    try:
        session_data = {
            "admin_authenticated": st.session_state.admin_authenticated,
            "admin_page": st.session_state.admin_page,
        }
        session_file = os.path.join(SESSION_DIR, "admin_session.pkl")
        with open(session_file, "wb") as f:
            pickle.dump(session_data, f)
        st.sidebar.markdown("**Debug: Session saved**")
    except Exception as e:
        st.sidebar.error(f"Error saving session: {str(e)}")

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if "admin_page" not in st.session_state:
    st.session_state.admin_page = "Dashboard"

load_session()

try:
    session_file = os.path.join(SESSION_DIR, "admin_session.pkl")
    if os.path.exists(session_file):
        with open(session_file, "rb") as f:
            session_data = pickle.load(f)
            if "admin_authenticated" in session_data and session_data["admin_authenticated"]:
                st.session_state.admin_authenticated = True
                if "admin_page" in session_data:
                    st.session_state.admin_page = session_data["admin_page"]
except Exception as e:
    st.sidebar.error(f"Error loading authentication: {str(e)}")

def run_admin_portal():
    """Main function to run the admin portal"""
    
    # Initialize all session state variables
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if "admin_page" not in st.session_state:
        st.session_state.admin_page = "Dashboard"
    
    # Load saved session data
    load_session()
    
    # Try to load authentication from session file
    try:
        session_file = os.path.join(SESSION_DIR, "admin_session.pkl")
        if os.path.exists(session_file):
            with open(session_file, "rb") as f:
                session_data = pickle.load(f)
                if "admin_authenticated" in session_data and session_data["admin_authenticated"]:
                    st.session_state.admin_authenticated = True
                    if "admin_page" in session_data:
                        st.session_state.admin_page = session_data["admin_page"]
    except Exception as e:
        st.sidebar.error(f"Error loading authentication: {str(e)}")
    
    # Admin login flow
    if not st.session_state.admin_authenticated:
        st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: auto;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 50px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.title("Admin Portal")
        st.markdown("#### Sign in to access the dashboard")
        
        admin_username = st.text_input("Username")
        admin_password = st.text_input("Password", type="password")
        
        col1, col2 = st.columns([1, 1])
        with col2:
            login_button = st.button("Login", use_container_width=True)
        
        if login_button:
            if admin_username == "admin" and admin_password == "password":
                st.session_state.admin_authenticated = True
                save_session()
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Admin authenticated pages
    else:
        st.markdown("""
        <style>
        .sidebar-content {
            padding-top: 20px;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .card {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .metric-card {
            text-align: center;
            background-color: #f8f9fa;
        }
        .chart-container {
            margin-top: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with st.sidebar:
            st.title("Admin Portal")
            st.divider()
            
            menu_options = {
                "Dashboard": "📊",
                "Manage Jobs": "💼",
                "Applications": "📝",
                "Students": "👨‍🎓"
            }
            
            for page, icon in menu_options.items():
                if st.button(f"{icon} {page}", key=f"menu_{page}", use_container_width=True, 
                           help=f"Navigate to {page}",
                           type="primary" if st.session_state.admin_page == page else "secondary"):
                    st.session_state.admin_page = page
                    save_session()
                    st.rerun()
            
            st.divider()
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.admin_authenticated = False
                session_file = os.path.join(SESSION_DIR, "admin_session.pkl")
                if os.path.exists(session_file):
                    os.remove(session_file)
                st.rerun()
        
        admin_page = st.session_state.admin_page
        
        # Dashboard page
        if admin_page == "Dashboard":
            st.title("📊 Dashboard")
            
            db = get_db()
            jobs_count = db.query(Job).count()
            students_count = db.query(Student).count()
            applications_count = db.query(Application).count()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                st.metric("Total Jobs", jobs_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                st.metric("Registered Students", students_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
                st.metric("Applications", applications_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
            st.subheader("Applications by Status")
            applications = db.query(Application).all()
            if applications:
                status_counts = {}
                for app in applications:
                    status = app.status
                    if status in status_counts:
                        status_counts[status] += 1
                    else:
                        status_counts[status] = 1
                
                status_df = pd.DataFrame({
                    'Status': list(status_counts.keys()),
                    'Count': list(status_counts.values())
                })
                
                fig = px.pie(status_df, values='Count', names='Status', hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Bold)
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No applications data yet")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="card chart-container">', unsafe_allow_html=True)
            st.subheader("Top Jobs by Applications")
            if applications:
                job_apps = {}
                for app in applications:
                    job_id = app.job_id
                    if job_id in job_apps:
                        job_apps[job_id] += 1
                    else:
                        job_apps[job_id] = 1
                
                sorted_jobs = sorted(job_apps.items(), key=lambda x: x[1], reverse=True)[:5]
                job_ids = [job_id for job_id, _ in sorted_jobs]
                counts = [count for _, count in sorted_jobs]
                
                job_titles = []
                for job_id in job_ids:
                    job = db.query(Job).filter(Job.id == job_id).first()
                    job_titles.append(job.title if job else f"Job {job_id}")
                
                jobs_df = pd.DataFrame({
                    'Job': job_titles,
                    'Applications': counts
                })
                
                fig = px.bar(jobs_df, x='Job', y='Applications', color='Applications',
                            color_continuous_scale='Viridis')
                fig.update_layout(xaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No job applications data yet")
            st.markdown('</div>', unsafe_allow_html=True)

        elif admin_page == "Manage Jobs":
            st.title("💼 Manage Jobs")
            
            with st.expander("➕ Add New Job", expanded=False):
                with st.form("new_job_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        job_title = st.text_input("Job Title")
                    with col2:
                        company = st.text_input("Company")
                    
                    description = st.text_area("Description")
                    
                    all_skills = ["Python", "Java", "JavaScript", "SQL", "Machine Learning", 
                                 "Data Analysis", "React", "Node.js", "AWS", "Docker", 
                                 "Communication", "Leadership", "Problem Solving", 
                                 "Project Management", "UX Design", "UI Design"]
                    
                    required_skills = st.multiselect("Required Skills", all_skills)
                    
                    col1, col2 = st.columns([4, 1])
                    with col2:
                        if st.form_submit_button("Add Job", use_container_width=True):
                            if job_title and company and description and required_skills:
                                db = get_db()
                                new_job = Job(
                                    title=job_title,
                                    company=company,
                                    description=description,
                                    skills=",".join(required_skills),
                                    date_posted=datetime.now(),
                                    status="active"
                                )
                                
                                db.add(new_job)
                                db.commit()
                                st.success("Job added successfully!")
                            else:
                                st.error("All fields are required")
            
            st.markdown("### Current Jobs")
            db = get_db()
            jobs = db.query(Job).all()
            
            if not jobs:
                st.info("No jobs available. Add your first job using the form above.")
            
            for job in jobs:
                with st.expander(f"{job.title} - {job.company}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Description:** {job.description}")
                        st.write(f"**Skills:** {job.skills}")
                        st.write(f"**Posted:** {job.date_posted}")
                        st.write(f"**Status:** {job.status}")
                    
                    with col2:
                        st.button("✏️ Edit", key=f"edit_{job.id}", 
                                 use_container_width=True,
                                 on_click=lambda jid=job.id: setattr(st.session_state, 'edit_job_id', jid))
                        
                        if st.button("🗑️ Delete", key=f"delete_{job.id}", use_container_width=True):
                            st.session_state[f"show_confirm_{job.id}"] = True
                        
                        if st.session_state.get(f"show_confirm_{job.id}", False):
                            delete_confirm = st.checkbox("Confirm deletion", key=f"confirm_{job.id}")
                            
                            if delete_confirm:
                                try:
                                    db = get_db()
                                    applications = db.query(Application).filter(Application.job_id == job.id).all()
                                    for app in applications:
                                        db.delete(app)
                                    
                                    job_to_delete = db.query(Job).filter(Job.id == job.id).first()
                                    if job_to_delete:
                                        db.delete(job_to_delete)
                                        db.commit()
                                        st.success("Job deleted successfully!")
                                        
                                        if f"show_confirm_{job.id}" in st.session_state:
                                            del st.session_state[f"show_confirm_{job.id}"]
                                        
                                        time.sleep(0.5)
                                        st.rerun()
                                    else:
                                        st.error("Job not found in database")
                                except Exception as e:
                                    db.rollback()
                                    st.error(f"Error deleting job: {str(e)}")
                                finally:
                                    db.close()
                                    
                            if st.button("Cancel", key=f"cancel_delete_{job.id}"):
                                del st.session_state[f"show_confirm_{job.id}"]
                                st.rerun()
                    
                    applications = db.query(Application).filter(Application.job_id == job.id).all()
                    if applications:
                        st.markdown("---")
                        st.write(f"**Applications:** {len(applications)}")
                        
                        status_counts = {}
                        for app in applications:
                            status = app.status
                            if status in status_counts:
                                status_counts[status] += 1
                            else:
                                status_counts[status] = 1
                        
                        for status, count in status_counts.items():
                            st.write(f"- {status.capitalize()}: {count}")
                    else:
                        st.markdown("---")
                        st.write("**Applications:** None yet")
            
            if "edit_job_id" in st.session_state:
                job_id = st.session_state.edit_job_id
                job = db.query(Job).filter(Job.id == job_id).first()
                
                if job:
                    st.markdown("---")
                    st.subheader(f"Edit Job: {job.title}")
                    
                    with st.form("edit_job_form"):
                        col1, col2 = st.columns(2)
                        with col1:
                            updated_title = st.text_input("Job Title", value=job.title)
                        with col2:
                            updated_company = st.text_input("Company", value=job.company)
                        
                        updated_description = st.text_area("Description", value=job.description)
                        
                        all_skills = ["Python", "Java", "JavaScript", "SQL", "Machine Learning", 
                                    "Data Analysis", "React", "Node.js", "AWS", "Docker", 
                                    "Communication", "Leadership", "Problem Solving", 
                                    "Project Management", "UX Design", "UI Design"]
                        
                        current_skills = job.skills.split(",") if job.skills else []
                        valid_current_skills = [skill for skill in current_skills if skill in all_skills]
                        updated_skills = st.multiselect("Required Skills", all_skills, default=valid_current_skills)
                        
                        updated_status = st.selectbox("Status", ["active", "closed"], index=0 if job.status == "active" else 1)
                        
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col2:
                            st.form_submit_button("Cancel", on_click=lambda: st.session_state.pop('edit_job_id', None))
                        with col3:
                            if st.form_submit_button("Update"):
                                job.title = updated_title
                                job.company = updated_company
                                job.description = updated_description
                                job.skills = ",".join(updated_skills)
                                job.status = updated_status
                                
                                db.commit()
                                st.success("Job updated successfully!")
                                
                                del st.session_state.edit_job_id
                                st.rerun()

        elif admin_page == "Applications":
            st.title("📝 Application Management")
            
            db = get_db()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.selectbox("Filter by Status", ["All", "pending", "reviewed", "shortlisted", "rejected"])
            
            with col2:
                jobs = db.query(Job).all()
                job_options = ["All"] + [job.title for job in jobs]
                job_filter = st.selectbox("Filter by Job", job_options)
            
            with col3:
                match_score_filter = st.slider("Minimum Match Score", 0, 100, 0)
            
            query = db.query(Application)
            
            if status_filter != "All":
                query = query.filter(Application.status == status_filter)
            
            if job_filter != "All":
                job_id = next((job.id for job in jobs if job.title == job_filter), None)
                if job_id:
                    query = query.filter(Application.job_id == job_id)
            
            if match_score_filter > 0:
                query = query.filter(Application.match_score >= match_score_filter)
            
            applications = query.all()
            
            if applications:
                st.markdown(f"### Found {len(applications)} applications")
                
                for app in applications:
                    job = db.query(Job).filter(Job.id == app.job_id).first()
                    student = db.query(Student).filter(Student.id == app.student_id).first()
                    
                    if job and student:
                        status_colors = {
                            "pending": "🟠",
                            "reviewed": "🔵",
                            "shortlisted": "🟢",
                            "rejected": "🔴"
                        }
                        
                        with st.expander(f"{status_colors.get(app.status, '⚪')} {student.name} - {job.title}"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**Student:** {student.name} ({student.email})")
                                st.write(f"**Job:** {job.title} at {job.company}")
                                st.write(f"**Applied on:** {app.application_date}")
                                st.write(f"**Match Score:** {app.match_score}%")
                                st.write(f"**Status:** {app.status.capitalize()}")
                                
                                st.progress(app.match_score/100)
                                
                                st.markdown("**Student Skills:**")
                                student_skills = student.skills.split(",") if student.skills else []
                                job_skills = job.skills.split(",") if job.skills else []
                                
                                for skill in student_skills:
                                    if skill in job_skills:
                                        st.success(skill)
                                    else:
                                        st.info(skill)
                                
                                missing_skills = [skill for skill in job_skills if skill not in student_skills]
                                if missing_skills:
                                    st.markdown("**Missing Skills:**")
                                    for skill in missing_skills:
                                        st.error(skill)
                            
                            with col2:
                                new_status = st.selectbox(
                                    "Update Status", 
                                    ["pending", "reviewed", "shortlisted", "rejected"],
                                    index=["pending", "reviewed", "shortlisted", "rejected"].index(app.status),
                                    key=f"status_{app.id}"
                                )
                                
                                if st.button("Update", key=f"update_{app.id}", use_container_width=True):
                                    app.status = new_status
                                    db.commit()
                                    st.success(f"Status updated to {new_status}!")
                                    st.info("Status updates would trigger email notifications in a real implementation.")
            else:
                st.info("No applications match your filters.")

        elif admin_page == "Students":
            st.title("👨‍🎓 Student Management")
            
            db = get_db()
            students = db.query(Student).all()
            
            if students:
                search_term = st.text_input("🔍 Search by name or email")
                
                filtered_students = students
                if search_term:
                    filtered_students = [
                        s for s in students 
                        if search_term.lower() in s.name.lower() 
                        or search_term.lower() in s.email.lower()
                    ]
                
                st.markdown(f"### Found {len(filtered_students)} students")
                
                for student in filtered_students:
                    with st.expander(f"{student.name} - {student.email}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**Name:** {student.name}")
                            st.write(f"**Email:** {student.email}")
                            st.write(f"**Education:** {student.education}")
                            
                            st.markdown("**Skills:**")
                            skills = student.skills.split(",") if student.skills else []
                            for skill in skills:
                                st.info(skill)
                            
                            st.markdown("**Resume Text:**")
                            st.markdown(
                                f"""
                                <div style="max-height: 200px; overflow-y: auto; padding: 10px; 
                                            border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
                                    {student.resume_text.replace('\n', '<br>')}
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        with col2:
                            applications = db.query(Application).filter(Application.student_id == student.id).all()
                            st.write(f"**Applications:** {len(applications)}")
                            
                            if applications:
                                status_counts = {}
                                for app in applications:
                                    status = app.status
                                    if status in status_counts:
                                        status_counts[status] += 1
                                    else:
                                        status_counts[status] = 1
                            
                                for status, count in status_counts.items():
                                    st.write(f"- {status.capitalize()}: {count}")
                            
                                avg_score = sum(app.match_score for app in applications) / len(applications)
                                st.metric("Avg Match Score", f"{avg_score:.1f}%")
                        
                            if applications:
                                st.markdown("**Application History:**")
                                status_colors = {
                                    "pending": "🟠",
                                    "reviewed": "🔵",
                                    "shortlisted": "🟢",
                                    "rejected": "🔴"
                                }
                                
                                for app in applications:
                                    job = db.query(Job).filter(Job.id == app.job_id).first()
                                    if job:
                                        st.write(f"{status_colors.get(app.status, '⚪')} {job.title} - {job.company} | Match: {app.match_score}% | Status: {app.status.capitalize()} | Applied: {app.application_date}")
            else:
                st.info("No students registered yet")

        # Close the database connection
        db.close()

# If the file is run directly (not imported), execute the main function
if __name__ == "__main__" and not os.environ.get("STREAMLIT_IMPORTED_BY_MAIN"):
    # This is for running the admin portal directly, not through main.py
    st.set_page_config(
        page_title="Admin Portal - Resume Matcher",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    run_admin_portal()