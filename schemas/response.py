from pydantic import BaseModel, Field

class HealthCheckResponse(BaseModel):
    status: str = Field(default="healthy", description="The overall operational status of the API.")
    version: str = Field(..., description="The current running version of the application.")
    debug: bool = Field(..., description="Indicates if the application is running in debug mode.")



class ChatResponse(BaseModel):
    response: str = Field(..., description="The AI Coding Mentor's response to the user.")
    current_phase: str = Field(..., description="The current phase of the mentoring process (e.g., 'IDEATION', 'CODING', 'REVIEW').")