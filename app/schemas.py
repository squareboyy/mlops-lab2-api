from typing import Dict, List, Annotated
from pydantic import BaseModel, Field, conlist


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    probability: float


class DriftRequest(BaseModel):
    samples: Annotated[
        List[Annotated[List[float], Field(min_length=4, max_length=4)]], 
        Field(min_length=10)
    ]
    alpha: float = Field(
        default=0.05, 
        ge=0.001, 
        le=0.5, 
        description="Поріг значущості для KS-тесту"
    )

class FeatureDriftInfo(BaseModel):
    statistic: float
    p_value: float
    drift_detected: bool

class DriftResponse(BaseModel):
    drift_detected: bool
    n_drifted_features: int
    drifted_features: List[str]
    per_feature: Dict[str, FeatureDriftInfo]
    n_samples: int
    alpha: float