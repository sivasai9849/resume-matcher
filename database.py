from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "postgresql://postgres.juwjnoezycsgkhetdcsi:ig7e3pg7Fiz1yqqN@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        raise e

class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    company = Column(String)
    description = Column(String)
    skills = Column(String)
    date_posted = Column(Date, default=datetime.now)
    status = Column(String)

    applications = relationship("Application", back_populates="job")

class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True)
    skills = Column(String)
    education = Column(String)
    resume_text = Column(String)

    applications = relationship("Application", back_populates="student")

class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    student_id = Column(Integer, ForeignKey("students.id"))
    match_score = Column(Float)
    status = Column(String)
    application_date = Column(Date, default=datetime.now)

    job = relationship("Job", back_populates="applications")
    student = relationship("Student", back_populates="applications")

# Create tables
Base.metadata.create_all(bind=engine) 