import logging
from typing import Optional

from notdiamond import settings
from notdiamond.exceptions import UnsupportedLLMProvider

POSSIBLE_PROVIDERS = list(settings.PROVIDERS.keys())
POSSIBLE_MODELS = list(
    model
    for provider_values in settings.PROVIDERS.values()
    for values in provider_values.values()
    if isinstance(values, list)
    for model in values
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class LLMConfig:
    """
    A NotDiamond LLM provider config (or LLMConfig) is represented by a combination of provider and model.
    Provider refers to the company of the foundational model, such as openai, anthropic, google.
    The model represents the model name as defined by the owner company, such as gpt-3.5-turbo
    Beside this you can also specify the API key for each provider, specify extra arguments
    that are also supported by Langchain (eg. temperature), and a system prmopt to be used
    with the provider. If the provider is selected during routing, then the system prompt will
    be used, replacing the one in the message array if there are any.

    All supported providers and models can be found in our docs.

    If the API key it's not specified, it will try to pick it up from an .env file before failing.
    As example for OpenAI it will look for OPENAI_API_KEY.

    Attributes:
        provider (str): The name of the LLM provider (e.g., "openai", "anthropic"). Must be one of the
                        predefined providers in `POSSIBLE_PROVIDERS`.
        model (str): The name of the LLM model to use (e.g., "gpt-3.5-turbo").
                        Must be one of the predefined models in `POSSIBLE_MODELS`.
        system_prompt (Optional[str], optional): The system prompt to use for the provider. Defaults to None.
        api_key (Optional[str], optional): The API key for accessing the LLM provider's services.
                                            Defaults to None, in which case it tries to fetch from the settings.
        openrouter_model (str): The OpenRouter model equivalent for this provider / model
        **kwargs: Additional keyword arguments that might be necessary for specific providers or models.

    Raises:
        UnsupportedLLMProvider: If the `provider` or `model` specified is not supported.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """_summary_

        Args:
            provider (str): The name of the LLM provider (e.g., "openai", "anthropic").
            model (str): The name of the LLM model to use (e.g., "gpt-3.5-turbo").
            system_prompt (Optional[str], optional): The system prompt to use for the provider. Defaults to None.
            api_key (Optional[str], optional): The API key for accessing the LLM provider's services.
                                                Defaults to None.
            **kwargs: Additional keyword arguments that might be necessary for specific providers or models.

        Raises:
            UnsupportedLLMProvider: If the `provider` or `model` specified is not supported.
        """
        if provider not in POSSIBLE_PROVIDERS:
            raise UnsupportedLLMProvider(
                f"Given LLM provider {provider} is not in the list of supported providers."
            )
        if model not in POSSIBLE_MODELS:
            raise UnsupportedLLMProvider(
                f"Given LLM model {model} is not in the list of supported models."
            )

        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self._openrouter_model = settings.PROVIDERS[provider][
            "openrouter_identifier"
        ].get(model, None)
        self.api_key = (
            api_key
            if api_key is not None
            else settings.PROVIDERS[provider]["api_key"]
        )
        self.kwargs = kwargs

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"

    def __repr__(self) -> str:
        return f"LLMConfig({self.provider}/{self.model})"

    def __eq__(self, other):
        if isinstance(other, LLMConfig):
            return (
                self.provider == other.provider and self.model == other.model
            )
        return False

    @property
    def openrouter_model(self):
        if self._openrouter_model is None:
            LOGGER.warning(
                f"Configured model {str(self)} is not available via OpenRouter. Please try another model."
            )
        return self._openrouter_model

    def prepare_for_request(self):
        """
        Converts the LLMConfig object to a dict in the format accepted by
        the NotDiamond API.

        Returns:
            dict
        """
        return {"provider": self.provider, "model": self.model}

    def set_api_key(self, api_key: str) -> "LLMConfig":
        self.api_key = api_key

        return self

    @classmethod
    def from_string(cls, llm_provider: str):
        """
        We allow our users to specify LLM providers for NotDiamond in the string format 'provider_name/model_name',
        as example 'openai/gpt-3.5-turbo'. Underlying our workflows we want to ensure we use LLMConfig as
        the base type, so this class method converts a string specification of an LLM provider into an
        LLMConfig object.

        Args:
            llm_provider (str): this is the string definition of the LLM provider

        Returns:
            LLMConfig: initialized object with correct provider and model
        """
        split_items = llm_provider.split("/")
        provider = split_items[0]
        model = split_items[1]
        return cls(provider=provider, model=model)
