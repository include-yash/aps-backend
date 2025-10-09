from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from database import Base

class Teacher(Base):
    __tablename__ = "teachers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    # simple setup, no auth or password

    tests = relationship("Test", back_populates="teacher")

class Test(Base):
    __tablename__ = "test"
    id = Column(Integer, primary_key=True, index=True)
    teacher_id = Column(Integer, ForeignKey("teachers.id"), nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    schema_sql = Column(Text, nullable=False)
    table_name = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    teacher = relationship("Teacher", back_populates="tests")
    questions = relationship("Question", back_populates="test")
    submissions = relationship("QuizSubmission", back_populates="test")

class Question(Base):
    __tablename__ = "question"
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer, ForeignKey("tests.id", ondelete="CASCADE"))
    question_text = Column(Text, nullable=False)
    difficulty = Column(String(20))
    expected_sql = Column(Text, nullable=False)

    test = relationship("Test", back_populates="questions")

class QuizSubmission(Base):
    __tablename__ = "quiz_submission"
    id = Column(Integer, primary_key=True, index=True)
    student_name = Column(String(100), nullable=False)
    student_usn = Column(String(20), nullable=False)
    test_id = Column(Integer, ForeignKey("tests.id", ondelete="CASCADE"))
    total_marks = Column(Integer, nullable=False)
    max_marks = Column(Integer, nullable=False)
    time_taken = Column(Integer)
    submitted_at = Column(DateTime(timezone=True), server_default=func.now())

    test = relationship("Test", back_populates="submissions")
