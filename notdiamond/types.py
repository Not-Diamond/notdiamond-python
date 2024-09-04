from typing import Any, Dict, List

from pydantic.v1 import BaseModel, validator

from notdiamond.exceptions import InvalidApiKey, MissingApiKey


class NDApiKeyValidator(BaseModel):
    api_key: str

    @validator("api_key", pre=True)
    @classmethod
    def api_key_must_be_a_string(cls, v) -> str:
        if not isinstance(v, str):
            raise InvalidApiKey("ND API key should be a string")
        return v

    @validator("api_key", pre=False)
    @classmethod
    def string_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise MissingApiKey("ND API key should be longer than 0")
        return v


class ModelSelectRequestPayload(BaseModel):
    prompt_template: str
    formatted_prompt: str
    components: Dict[str, Dict]
    llm_configs: List[Dict]
    metric: str
    max_model_depth: int


class FeedbackRequestPayload(BaseModel):
    session_id: str
    provider: Dict[str, Any]
    feedback: Dict[str, int]
