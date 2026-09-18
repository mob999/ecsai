"""Host execution settings, deliberately separate from simulated resources."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EDGE_SIM_", extra="forbid", frozen=True)

    workers: int = Field(default=1, ge=1)
    timeout_s: float = Field(default=30, gt=0, allow_inf_nan=False)
    output_dir: str = "outputs"
