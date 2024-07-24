from typing import Optional

from notdiamond import settings
from notdiamond.exceptions import ApiError
from notdiamond.llms.config import LLMConfig
from notdiamond.metrics.request import feedback_request
from notdiamond.types import NDApiKeyValidator


class Metric:
    def __init__(self, metric: Optional[str] = "accuracy"):
        self.metric = metric

    def __call__(self):
        return self.metric

    def feedback(
        self,
        session_id: str,
        llm_config: LLMConfig,
        value: int,
        notdiamond_api_key: Optional[str] = None,
    ):
        if notdiamond_api_key is None:
            notdiamond_api_key = settings.NOTDIAMOND_API_KEY
        NDApiKeyValidator(api_key=notdiamond_api_key)
        if value not in [0, 1]:
            raise ApiError("Invalid feedback value. It must be 0 or 1.")

        return feedback_request(
            session_id=session_id,
            llm_config=llm_config,
            feedback_payload=self.request_payload(value),
            notdiamond_api_key=notdiamond_api_key,
        )

    def request_payload(self, value: int):
        return {self.metric: value}
