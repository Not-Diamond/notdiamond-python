import logging
from typing import Dict

import requests

from notdiamond import settings
from notdiamond._utils import _default_headers
from notdiamond.exceptions import ApiError
from notdiamond.llms.config import LLMConfig
from notdiamond.types import FeedbackRequestPayload

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def feedback_request(
    session_id: str,
    llm_config: LLMConfig,
    feedback_payload: Dict[str, int],
    notdiamond_api_key: str,
    nd_api_url: str = settings.NOTDIAMOND_API_URL,
    _user_agent: str = settings.DEFAULT_USER_AGENT,
) -> bool:
    url = f"{nd_api_url}/v2/report/metrics/feedback"

    payload: FeedbackRequestPayload = {
        "session_id": session_id,
        "provider": llm_config.prepare_for_request(),
        "feedback": feedback_payload,
    }

    headers = _default_headers(notdiamond_api_key, _user_agent)

    try:
        response = requests.post(url, json=payload, headers=headers)
    except Exception as e:
        raise ApiError(f"ND API error for feedback: {e}")

    if response.status_code != 200:
        LOGGER.error(
            f"ND API feedback error: failed to report feedback with status {response.status_code}. {response.text}"
        )
        return False

    return True
