from typing import Optional, List, Any, Dict
from pydantic import BaseModel, Field
from pydantic import ConfigDict

class PredictSalaryRequest(BaseModel):
    role: Optional[str] = None
    skills: Optional[str] = None
    work_type: Optional[Any] = None
    qualification: Optional[str] = None
    preference: Optional[str] = None
    experience_mid: Optional[Any] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    embedding_1024: Optional[List[float]] = Field(default=None, description="Raw 1024-D embedding if already computed")

class PredictSalaryResponse(BaseModel):
    predicted_salary: float

class RecommendJobsRequest(BaseModel):
    role: Optional[str] = None
    skills: Optional[str] = None
    education: Optional[str] = None
    work_type: Optional[Any] = None
    experience: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    gender: Optional[str] = None
    max_recommendations: int = 10

class JobRecommendation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # Dataset-mapped fields with exact alias names
    job_id: Any = Field(alias="Job_Id")
    job_title: str = Field(alias="Job Title")
    role: str = Field(alias="Role")
    job_description: Any = Field(alias="Job Description")
    benefits: Any = Field(alias="Benefits")
    skills: Any = Field(alias="skills")
    responsibilities: Any = Field(alias="Responsibilities")
    company: str = Field(alias="Company")
    company_size: Any = Field(alias="Company Size")
    company_profile: Any = Field(alias="Company Profile")
    experience: Any = Field(alias="Experience")
    salary_range: Any = Field(alias="Salary Range")
    qualifications: Any = Field(alias="Qualifications")
    location: Any = Field(alias="location")
    country: str = Field(alias="Country")
    work_type: Any = Field(alias="Work_Type")
    preference: Any = Field(alias="Preference")
    latitude: Any = Field(alias="latitude")
    longitude: Any = Field(alias="longitude")

    # Scoring metadata
    total_score: float
    individual_scores: Dict[str, float] = {}

class RecommendJobsResponse(BaseModel):
    recommendations: List[JobRecommendation]

class PredictAndRecommendRequest(BaseModel):
    role: Optional[str] = None
    skills: Optional[str] = None
    qualification: Optional[str] = None
    work_type: Optional[Any] = None
    experience: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    preference: Optional[str] = None
    gender: Optional[str] = None
    max_recommendations: int = 10

class PredictAndRecommendResponse(BaseModel):
    predicted_salary: float
    recommendations: List[JobRecommendation]