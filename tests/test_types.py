"""Tests for the types defined in the Python library"""


import sys

import pytest

from notdiamond.exceptions import InvalidApiKey, MissingApiKey
from notdiamond.types import NDApiKeyValidator

sys.path.append("../")


def test_api_key_not_string_fails():
    """When the API key is not a string, the validation fails."""
    with pytest.raises(InvalidApiKey):
        NDApiKeyValidator(api_key=1)

    with pytest.raises(InvalidApiKey):
        NDApiKeyValidator(api_key=None)


def test_api_key_empty_string_fails():
    """When the API key is an empty string, the validation fails."""
    with pytest.raises(MissingApiKey):
        NDApiKeyValidator(api_key="")


def test_api_key_passes():
    """When the API key is a valid string and not empty, validation passes successfully."""
    notdiamond_api_key = NDApiKeyValidator(api_key="NDAPIKEY")
    assert notdiamond_api_key.api_key == "NDAPIKEY"
