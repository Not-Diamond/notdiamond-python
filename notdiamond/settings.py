import os
from importlib import metadata

from dotenv import load_dotenv

load_dotenv(os.getcwd() + "/.env")

VERSION = metadata.version("notdiamond")

NOTDIAMOND_API_KEY = os.getenv("NOTDIAMOND_API_KEY", default="")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default="")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", default="")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", default="")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", default="")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", default="")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", default="")
PPLX_API_KEY = os.getenv("PPLX_API_KEY", default="")
REPLICATE_API_KEY = os.getenv("REPLICATE_API_KEY", default="")


ND_BASE_URL = "https://not-diamond-server.onrender.com"

PROVIDERS = {
    "openai": {
        "models": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
        ],
        "api_key": OPENAI_API_KEY,
        "support_tools": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
        ],
        "support_response_model": [
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini",
            "gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-4-1106-preview",
        ],
        "openrouter_identifier": {
            "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
            "gpt-4": "openai/gpt-4",
            "gpt-4-turbo": "openai/gpt-4-turbo",
            "gpt-4o": "openai/gpt-4o",
            "gpt-4o-2024-05-13": "openai/gpt-4o-2024-05-13",
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "gpt-4o-mini-2024-07-18": "openai/gpt-4o-mini-2024-07-18",
            "gpt-4-turbo-preview": "openai/gpt-4-turbo-preview",
            "gpt-4-1106-preview": "openai/gpt-4-1106-preview",
        },
    },
    "anthropic": {
        "models": [
            "claude-2.1",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
        ],
        "api_key": ANTHROPIC_API_KEY,
        "support_tools": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
        ],
        "support_response_model": [
            "claude-2.1",
            "claude-3-opus-20240229",
        ],
        "openrouter_identifier": {
            "claude-2.1": "anthropic/claude-2.1",
            "claude-3-opus-20240229": "anthropic/claude-3-opus",
            "claude-3-sonnet-20240229": "anthropic/claude-3-sonnet",
            "claude-3-haiku-20240307": "anthropic/claude-3-haiku",
            "claude-3-5-sonnet-20240620": "anthropic/claude-3.5-sonnet",
        },
    },
    "google": {
        "models": [
            "gemini-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
        ],
        "api_key": GOOGLE_API_KEY,
        "support_tools": [
            "gemini-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
        ],
        "support_response_model": [
            "gemini-pro",
            "gemini-1.0-pro-latest",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
        ],
        "openrouter_identifier": {
            "gemini-pro": "google/gemini-pro",
            "gemini-1.0-pro-latest": "google/gemini-pro",
            "gemini-1.5-pro-latest": "google/gemini-pro-1.5",  #
            "gemini-1.5-flash-latest": "google/gemini-flash-1.5",  #
        },
    },
    "cohere": {
        "models": ["command-r", "command-r-plus"],
        "api_key": COHERE_API_KEY,
        "support_tools": ["command-r", "command-r-plus"],
        "support_response_model": ["command-r", "command-r-plus"],
        "openrouter_identifier": {
            "command-r": "cohere/command-r",
            "command-r-plus": "cohere/command-r-plus",
        },
    },
    "mistral": {
        "models": [
            "mistral-large-latest",
            "mistral-large-2402",
            "mistral-large-2407",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
            "codestral-latest",
        ],
        "api_key": MISTRAL_API_KEY,
        "support_tools": [
            "mistral-large-latest",
            "mistral-large-2402",
            "mistral-large-2407",
            "mistral-small-latest",
            "open-mixtral-8x22b",
        ],
        "support_response_model": [
            "mistral-large-latest",
            "mistral-large-2402",
            "mistral-large-2407",
            "mistral-medium-latest",
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
            "codestral-latest",
        ],
        "openrouter_identifier": {
            "mistral-large-latest": "mistralai/mistral-large",
            "mistral-large-2407": "mistralai/mistral-large",
            "mistral-medium-latest": "mistralai/mistral-medium",
            "mistral-small-latest": "mistralai/mistral-small",
            "open-mistral-7b": "mistralai/mistral-7b-instruct",
            "open-mixtral-8x7b": "mistralai/mixtral-8x7b",
            "open-mixtral-8x22b": "mistralai/mixtral-8x22b-instruct",
        },
    },
    "togetherai": {
        "models": [
            "Phind-CodeLlama-34B-v2",
            "Mistral-7B-Instruct-v0.2",
            "Mixtral-8x7B-Instruct-v0.1",
            "Mixtral-8x22B-Instruct-v0.1",
            "Llama-3-70b-chat-hf",
            "Llama-3-8b-chat-hf",
            "Qwen2-72B-Instruct",
            "Meta-Llama-3.1-8B-Instruct-Turbo",
            "Meta-Llama-3.1-70B-Instruct-Turbo",
            "Meta-Llama-3.1-405B-Instruct-Turbo",
        ],
        "api_key": TOGETHER_API_KEY,
        "model_prefix": {
            "Phind-CodeLlama-34B-v2": "Phind",
            "Mistral-7B-Instruct-v0.2": "mistralai",
            "Mixtral-8x7B-Instruct-v0.1": "mistralai",
            "Mixtral-8x22B-Instruct-v0.1": "mistralai",
            "Llama-3-70b-chat-hf": "meta-llama",
            "Llama-3-8b-chat-hf": "meta-llama",
            "Qwen2-72B-Instruct": "Qwen",
            "Meta-Llama-3.1-8B-Instruct-Turbo": "meta-llama",
            "Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama",
            "Meta-Llama-3.1-405B-Instruct-Turbo": "meta-llama",
        },
        "openrouter_identifier": {
            "Phind-CodeLlama-34B-v2": "phind/phind-codellama-34b",
            "Llama-3-70b-chat-hf": "meta-llama/llama-3-70b-instruct",
            "Llama-3-8b-chat-hf": "meta-llama/llama-3-8b-instruct",
            "Qwen2-72B-Instruct": "qwen/qwen-2-72b-instruct",
            "Mistral-7B-Instruct-v0.2": "mistralai/mistral-7b-instruct",
            "Mixtral-8x7B-Instruct-v0.1": "mistralai/mixtral-8x7b",
            "Mixtral-8x22B-Instruct-v0.1": "mistralai/mixtral-8x22b-instruct",
            "Meta-Llama-3.1-8B-Instruct-Turbo": "meta-llama/llama-3.1-8b-instruct",
            "Meta-Llama-3.1-70B-Instruct-Turbo": "meta-llama/llama-3.1-70b-instruct",
            "Meta-Llama-3.1-405B-Instruct-Turbo": "meta-llama/llama-3.1-405b-instruct",
        },
    },
    "perplexity": {
        "models": [
            "llama-3-sonar-large-32k-online",
        ],
        "api_key": PPLX_API_KEY,
        "openrouter_identifier": {
            "llama-3-sonar-large-32k-online": "perplexity/llama-3-sonar-large-32k-online",
        },
    },
    "replicate": {
        "models": [
            "mistral-7b-instruct-v0.2",
            "mixtral-8x7b-instruct-v0.1",
            "meta-llama-3-70b-instruct",
            "meta-llama-3-8b-instruct",
            "meta-llama-3.1-405b-instruct",
        ],
        "api_key": REPLICATE_API_KEY,
        "model_prefix": {
            "mistral-7b-instruct-v0.2": "mistralai",
            "mixtral-8x7b-instruct-v0.1": "mistralai",
            "meta-llama-3-70b-instruct": "meta",
            "meta-llama-3-8b-instruct": "meta",
            "meta-llama-3.1-405b-instruct": "meta",
        },
        "openrouter_identifier": {
            "mistral-7b-instruct-v0.2": "mistralai/mistral-7b-instruct",
            "mixtral-8x7b-instruct-v0.1": "mistralai/mixtral-8x7b",
            "meta-llama-3-70b-instruct": "meta-llama/llama-3-70b-instruct",
            "meta-llama-3-8b-instruct": "meta-llama/llama-3-8b-instruct",
            "meta-llama-3.1-405b-instruct": "meta-llama/llama-3.1-405b-instruct",
        },
    },
}
