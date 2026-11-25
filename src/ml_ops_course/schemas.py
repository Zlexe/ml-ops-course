from pydantic import BaseModel, Field


class ClientFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, examples=[41])
    job: str = Field(..., examples=["technician"])
    marital: str = Field(..., examples=["married"])
    education: str = Field(..., examples=["secondary"])
    default: str = Field(..., examples=["no"])
    balance: int = Field(..., examples=[1270])
    housing: str = Field(..., examples=["yes"])
    loan: str = Field(..., examples=["no"])
    contact: str = Field(..., examples=["unknown"])
    day: int = Field(..., ge=1, le=31, examples=[5])
    month: str = Field(..., examples=["may"])
    campaign: int = Field(..., ge=1, examples=[1])
    pdays: int = Field(..., examples=[-1])
    previous: int = Field(..., ge=0, examples=[0])
    poutcome: str = Field(..., examples=["unknown"])


class PredictionResponse(BaseModel):
    probability: float
    label: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
