from enum import Enum

from notdiamond.llms.config import LLMConfig


class NDLLMProviders(Enum):
    """
    NDLLMProviders serves as a registry for the supported LLM models by NotDiamond.
    It allows developers to easily specify available LLM providers for the router.

    Attributes:
        GPT_3_5_TURBO (NDLLMProvider): refers to 'gpt-3.5-turbo' model by OpenAI
        GPT_3_5_TURBO_0125 (NDLLMProvider): refers to 'gpt-3.5-turbo-0125' model by OpenAI
        GPT_4 (NDLLMProvider): refers to 'gpt-4' model by OpenAI
        GPT_4_0613 (NDLLMProvider): refers to 'gpt-4-0613' model by OpenAI
        GPT_4_1106_PREVIEW (NDLLMProvider): refers to 'gpt-4-1106-preview' model by OpenAI
        GPT_4_TURBO (NDLLMProvider): refers to 'gpt-4-turbo' model by OpenAI
        GPT_4_TURBO_PREVIEW (NDLLMProvider): refers to 'gpt-4-turbo-preview' model by OpenAI
        GPT_4_TURBO_2024_04_09 (NDLLMProvider): refers to 'gpt-4-turbo-2024-04-09' model by OpenAI
        GPT_4o_2024_05_13 (NDLLMProvider): refers to 'gpt-4o-2024-05-13' model by OpenAI
        GPT_4o (NDLLMProvider): refers to 'gpt-4o' model by OpenAI
        GPT_4o_MINI_2024_07_18 (NDLLMProvider): refers to 'gpt-4o-mini-2024-07-18' model by OpenAI
        GPT_4o_MINI (NDLLMProvider): refers to 'gpt-4o-mini' model by OpenAI
        GPT_4_0125_PREVIEW (NDLLMProvider): refers to 'gpt-4-0125-preview' model by OpenAI

        CLAUDE_2_1 (NDLLMProvider): refers to 'claude-2.1' model by Anthropic
        CLAUDE_3_OPUS_20240229 (NDLLMProvider): refers to 'claude-3-opus-20240229' model by Anthropic
        CLAUDE_3_SONNET_20240229 (NDLLMProvider): refers to 'claude-3-sonnet-20240229' model by Anthropic
        CLAUDE_3_5_SONNET_20240620 (NDLLMProvider): refers to 'claude-3-5-sonnet-20240620' model by Anthropic
        CLAUDE_3_HAIKU_20240307 (NDLLMProvider): refers to 'claude-3-haiku-20240307' model by Anthropic

        GEMINI_PRO (NDLLMProvider): refers to 'gemini-pro' model by Google
        GEMINI_1_PRO_LATEST (NDLLMProvider): refers to 'gemini-1.0-pro-latest' model by Google
        GEMINI_15_PRO_LATEST (NDLLMProvider): refers to 'gemini-1.5-pro-latest' model by Google
        GEMINI_15_FLASH_LATEST (NDLLMProvider): refers to 'gemini-1.5-flash-latest' model by Google

        COMMAND_R (NDLLMProvider): refers to 'command-r' model by Cohere
        COMMAND_R_PLUS (NDLLMProvider): refers to 'command-r-plus' model by Cohere

        MISTRAL_LARGE_LATEST (NDLLMProvider): refers to 'mistral-large-latest' model by Mistral AI
        MISTRAL_LARGE_2407 (NDLLMProvider): refers to 'mistral-large-2407' model by Mistral AI
        MISTRAL_LARGE_2402 (NDLLMProvider): refers to 'mistral-large-2402' model by Mistral AI
        MISTRAL_MEDIUM_LATEST (NDLLMProvider): refers to 'mistral-medium-latest' model by Mistral AI
        MISTRAL_SMALL_LATEST (NDLLMProvider): refers to 'mistral-small-latest' model by Mistral AI
        OPEN_MISTRAL_7B (NDLLMProvider): refers to 'open-mistral-7b' model by Mistral AI
        OPEN_MIXTRAL_8X7B (NDLLMProvider): refers to 'open-mixtral-8x7b' model by Mistral AI
        OPEN_MIXTRAL_8X22B (NDLLMProvider): refers to 'open-mixtral-8x22b' model by Mistral AI

        TOGETHER_PHIND_CODELLAMA_34B_V2 (NDLLMProvider): refers to 'Phind-CodeLlama-34B-v2' model served via TogetherAI
        TOGETHER_MISTRAL_7B_INSTRUCT_V0_2 (NDLLMProvider): refers to 'Mistral-7B-Instruct-v0.2' model served via TogetherAI
        TOGETHER_MIXTRAL_8X7B_INSTRUCT_V0_1 (NDLLMProvider): refers to 'Mixtral-8x7B-Instruct-v0.1' model served via TogetherAI
        TOGETHER_MIXTRAL_8X22B_INSTRUCT_V0_1 (NDLLMProvider): refers to 'Mixtral-8x22B-Instruct-v0.1' model served via TogetherAI
        TOGETHER_LLAMA_3_70B_CHAT_HF (NDLLMProvider): refers to 'Llama-3-70b-chat-hf' model served via TogetherAI
        TOGETHER_LLAMA_3_8B_CHAT_HF (NDLLMProvider): refers to 'Llama-3-8b-chat-hf' model served via TogetherAI
        TOGETHER_QWEN2_72B_INSTRUCT (NDLLMProvider): refers to 'Qwen2-72B-Instruct' model served via TogetherAI
        TOGETHER_LLAMA_3_1_8B_INSTRUCT_TURBO (NDLLMProvider): refers to 'Meta-Llama-3.1-8B-Instruct-Turbo' model served via TogetherAI
        TOGETHER_LLAMA_3_1_70B_INSTRUCT_TURBO (NDLLMProvider): refers to 'Meta-Llama-3.1-70B-Instruct-Turbo' model served via TogetherAI
        TOGETHER_LLAMA_3_1_405B_INSTRUCT_TURBO (NDLLMProvider): refers to 'Meta-Llama-3.1-405B-Instruct-Turbo' model served via TogetherAI

        REPLICATE_MISTRAL_7B_INSTRUCT_V0_2 (NDLLMProvider): refers to "mistral-7b-instruct-v0.2" model served via Replicate
        REPLICATE_MIXTRAL_8X7B_INSTRUCT_V0_1 (NDLLMProvider): refers to "mixtral-8x7b-instruct-v0.1" model served via Replicate
        REPLICATE_META_LLAMA_3_70B_INSTRUCT (NDLLMProvider): refers to "meta-llama-3-70b-instruct" model served via Replicate
        REPLICATE_META_LLAMA_3_8B_INSTRUCT (NDLLMProvider): refers to "meta-llama-3-8b-instruct" model served via Replicate
        REPLICATE_META_LLAMA_3_1_405B_INSTRUCT (NDLLMProvider): refers to "meta-llama-3.1-405b-instruct" model served via Replicate

    Note:
        This class is static and designed to be used without instantiation.
        Access its attributes directly to obtain configurations for specific LLM providers.
    """

    GPT_3_5_TURBO = ("openai", "gpt-3.5-turbo")
    GPT_3_5_TURBO_0125 = ("openai", "gpt-3.5-turbo-0125")
    GPT_4 = ("openai", "gpt-4")
    GPT_4_0613 = ("openai", "gpt-4-0613")
    GPT_4_1106_PREVIEW = ("openai", "gpt-4-1106-preview")
    GPT_4_TURBO = ("openai", "gpt-4-turbo")
    GPT_4_TURBO_PREVIEW = ("openai", "gpt-4-turbo-preview")
    GPT_4_TURBO_2024_04_09 = ("openai", "gpt-4-turbo-2024-04-09")
    GPT_4o_2024_05_13 = ("openai", "gpt-4o-2024-05-13")
    GPT_4o = ("openai", "gpt-4o")
    GPT_4o_MINI_2024_07_18 = ("openai", "gpt-4o-mini-2024-07-18")
    GPT_4o_MINI = ("openai", "gpt-4o-mini")
    GPT_4_0125_PREVIEW = ("openai", "gpt-4-0125-preview")

    CLAUDE_2_1 = ("anthropic", "claude-2.1")
    CLAUDE_3_OPUS_20240229 = ("anthropic", "claude-3-opus-20240229")
    CLAUDE_3_SONNET_20240229 = ("anthropic", "claude-3-sonnet-20240229")
    CLAUDE_3_5_SONNET_20240620 = ("anthropic", "claude-3-5-sonnet-20240620")
    CLAUDE_3_HAIKU_20240307 = ("anthropic", "claude-3-haiku-20240307")

    GEMINI_PRO = ("google", "gemini-pro")
    GEMINI_1_PRO_LATEST = ("google", "gemini-1.0-pro-latest")
    GEMINI_15_PRO_LATEST = ("google", "gemini-1.5-pro-latest")
    GEMINI_15_FLASH_LATEST = ("google", "gemini-1.5-flash-latest")

    COMMAND_R = ("cohere", "command-r")
    COMMAND_R_PLUS = ("cohere", "command-r-plus")

    MISTRAL_LARGE_LATEST = ("mistral", "mistral-large-latest")
    MISTRAL_LARGE_2407 = ("mistral", "mistral-large-2407")
    MISTRAL_LARGE_2402 = ("mistral", "mistral-large-2402")
    MISTRAL_MEDIUM_LATEST = ("mistral", "mistral-medium-latest")
    MISTRAL_SMALL_LATEST = ("mistral", "mistral-small-latest")
    CODESTRAL_LATEST = ("mistral", "codestral-latest")
    OPEN_MISTRAL_7B = ("mistral", "open-mistral-7b")
    OPEN_MIXTRAL_8X7B = ("mistral", "open-mixtral-8x7b")
    OPEN_MIXTRAL_8X22B = ("mistral", "open-mixtral-8x22b")

    TOGETHER_PHIND_CODELLAMA_34B_V2 = ("togetherai", "Phind-CodeLlama-34B-v2")
    TOGETHER_MISTRAL_7B_INSTRUCT_V0_2 = (
        "togetherai",
        "Mistral-7B-Instruct-v0.2",
    )
    TOGETHER_MIXTRAL_8X7B_INSTRUCT_V0_1 = (
        "togetherai",
        "Mixtral-8x7B-Instruct-v0.1",
    )
    TOGETHER_MIXTRAL_8X22B_INSTRUCT_V0_1 = (
        "togetherai",
        "Mixtral-8x22B-Instruct-v0.1",
    )
    TOGETHER_LLAMA_3_70B_CHAT_HF = ("togetherai", "Llama-3-70b-chat-hf")
    TOGETHER_LLAMA_3_8B_CHAT_HF = ("togetherai", "Llama-3-8b-chat-hf")
    TOGETHER_QWEN2_72B_INSTRUCT = ("togetherai", "Qwen2-72B-Instruct")
    TOGETHER_LLAMA_3_1_8B_INSTRUCT_TURBO = ("togetherai", "Meta-Llama-3.1-8B-Instruct-Turbo")
    TOGETHER_LLAMA_3_1_70B_INSTRUCT_TURBO = ("togetherai", "Meta-Llama-3.1-70B-Instruct-Turbo")
    TOGETHER_LLAMA_3_1_405B_INSTRUCT_TURBO = ("togetherai", "Meta-Llama-3.1-405B-Instruct-Turbo")

    LLAMA_3_SONAR_LARGE_32K_ONLINE = (
        "perplexity",
        "llama-3-sonar-large-32k-online",
    )

    REPLICATE_MISTRAL_7B_INSTRUCT_V0_2 = (
        "replicate",
        "mistral-7b-instruct-v0.2",
    )
    REPLICATE_MIXTRAL_8X7B_INSTRUCT_V0_1 = (
        "replicate",
        "mixtral-8x7b-instruct-v0.1",
    )
    REPLICATE_META_LLAMA_3_70B_INSTRUCT = (
        "replicate",
        "meta-llama-3-70b-instruct",
    )
    REPLICATE_META_LLAMA_3_8B_INSTRUCT = (
        "replicate",
        "meta-llama-3-8b-instruct",
    )
    REPLICATE_META_LLAMA_3_1_405B_INSTRUCT = (
        "replicate",
        "meta-llama-3.1-405b-instruct",
    )

    def __new__(cls, provider, model):
        return LLMConfig(provider=provider, model=model)
