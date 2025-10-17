from pydantic import BaseSettings, Field
from typing import List


class Settings(BaseSettings):
    mlflow_tracking_uri: str = Field(default="http://mlflow:5000", alias="MLFLOW_TRACKING_URI")
    model_names: List[str] = Field(default_factory=lambda: ["model_lgbm", "model_cat"], alias="MODEL_NAMES")
    poll_seconds: int = Field(default=30, alias="POLL_SECONDS")
    meta_state_path: str = Field(default="/state/meta.pkl", alias="META_STATE_PATH")


class Config:
    env_file = ".env"
    env_file_encoding = "utf-8"
    populate_by_name = True


settings = Settings()