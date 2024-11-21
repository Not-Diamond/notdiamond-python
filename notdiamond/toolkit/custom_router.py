import json
import tempfile
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from litellm import token_counter
from tqdm import tqdm

from notdiamond.exceptions import ApiError
from notdiamond.llms.client import NotDiamond
from notdiamond.llms.config import LLMConfig
from notdiamond.settings import NOTDIAMOND_API_KEY, NOTDIAMOND_API_URL, VERSION
from notdiamond.types import NDApiKeyValidator


class CustomRouter:
    """
    Implementation of CustomRouter class, used to train custom routers using custom datasets.

    Attributes:
        language (str): The language of the dataset in lowercase. Defaults to "english".
        maximize (bool): Whether higher score is better. Defaults to true.
        api_key (Optional[str], optional): The NotDiamond API key. If not specified, will try to
            find it in the environment variable NOTDIAMOND_API_KEY.
    """

    def __init__(
        self,
        language: str = "english",
        maximize: bool = True,
        api_key: Optional[str] = None,
    ):
        if api_key is None:
            api_key = NOTDIAMOND_API_KEY
        NDApiKeyValidator(api_key=api_key)

        self.api_key = api_key
        self.language = language
        self.maximize = maximize

    def _request_train_router(
        self,
        prompt_column: str,
        dataset_file: str,
        llm_configs: List[LLMConfig],
        preference_id: Optional[str],
        nd_api_url: str,
    ) -> str:
        url = f"{nd_api_url}/v2/pzn/trainCustomRouter"

        files = {"dataset_file": open(dataset_file, "rb")}

        payload = {
            "language": self.language,
            "llm_providers": json.dumps(
                [provider.prepare_for_request() for provider in llm_configs]
            ),
            "prompt_column": prompt_column,
            "maximize": self.maximize,
            "preference_id": preference_id,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": f"Python-SDK/{VERSION}",
        }

        response = requests.post(
            url=url, headers=headers, data=payload, files=files
        )
        if response.status_code != 200:
            raise ApiError(
                f"ND backend error status code: {response.status_code}, {response.text}"
            )

        preference_id = response.json()["preference_id"]
        return preference_id

    def _prepare_joint_dataset(
        self,
        dataset: Dict[Union[str, LLMConfig], pd.DataFrame],
        prompt_column: str,
        response_column: str,
        score_column: str,
    ) -> Tuple[pd.DataFrame, List[LLMConfig]]:
        a_provider = list(dataset.keys())[0]
        prompts = dataset[a_provider].get(prompt_column, None)
        if prompts is None:
            raise ValueError(f"Prompt column {prompt_column} not found in df.")
        prompts = prompts.to_list()

        llm_configs = []
        joint_dataset = {prompt_column: prompts}
        for provider, df in dataset.items():
            llm_configs.append(provider)

            responses = df.get(response_column, None)
            if responses is None:
                raise ValueError(
                    f"Response column {response_column} not found in df."
                )
            responses = responses.to_list()
            joint_dataset[f"{str(provider)}/response"] = responses

            scores = df.get(score_column, None)
            if scores is None:
                raise ValueError(
                    f"Score column {score_column} not found in df."
                )
            scores = scores.to_list()
            joint_dataset[f"{str(provider)}/score"] = scores

        joint_df = pd.DataFrame(joint_dataset)

        llm_configs = NotDiamond._parse_llm_configs_data(llm_configs)
        return joint_df, llm_configs

    def fit(
        self,
        dataset: Dict[Union[str, LLMConfig], pd.DataFrame],
        prompt_column: str,
        response_column: str,
        score_column: str,
        preference_id: Optional[str] = None,
        nd_api_url: Optional[str] = NOTDIAMOND_API_URL,
    ) -> str:
        """
        Method to train a custom router using provided dataset.

        Parameters:
            dataset (Dict[str, pandas.DataFrame]): The dataset to train a custom router.
                Each key in the dictionary should be in the form of <provider>/<model>.
            prompt_column (str): The column name in each DataFrame corresponding
                to the prompts used to evaluate the LLM.
            response_column (str): The column name in each DataFrame corresponding
                to the response given by the LLM for a given prompt.
            score_column (str): The column name in each DataFrame corresponding
                to the score given to the response from the LLM.
            preference_id (Optional[str], optional): If specified, the custom router
                associated with the preference_id will be updated with the provided dataset.
            nd_api_url (Optional[str], optional): The URL of the NotDiamond API. Defaults to prod.

        Raises:
            ApiError: When the NotDiamond API fails
            ValueError: When parsing the provided dataset fails
            UnsupportedLLMProvider: When a provider specified in the dataset is not supported.

        Returns:
            str:
                preference_id: the preference_id associated with the custom router.
                    Use this preference_id in your routing calls to use the custom router.
        """

        joint_df, llm_configs = self._prepare_joint_dataset(
            dataset, prompt_column, response_column, score_column
        )

        with tempfile.NamedTemporaryFile(suffix=".csv") as joint_csv:
            joint_df.to_csv(joint_csv.name, index=False)
            preference_id = self._request_train_router(
                prompt_column,
                joint_csv.name,
                llm_configs,
                preference_id,
                nd_api_url,
            )

        return preference_id

    def _get_latency(self, llm_config: LLMConfig, prompt: str) -> float:
        llm = NotDiamond._llm_from_config(llm_config)
        start_time = time.time()
        _ = llm.invoke([("human", prompt)])
        end_time = time.time()
        return (end_time - start_time) * 1000  # ms

    def _get_cost(
        self, llm_config: LLMConfig, prompt: str, response: str
    ) -> float:
        n_input_tokens = token_counter(model="gpt-4o", text=prompt)
        n_output_tokens = token_counter(model="gpt-4o", text=response)
        input_price = (
            llm_config.default_input_price
            if llm_config.input_price is None
            else llm_config.input_price
        )
        output_price = (
            llm_config.default_output_price
            if llm_config.output_price is None
            else llm_config.output_price
        )
        return (
            n_input_tokens * input_price + n_output_tokens * output_price
        ) / 1e6

    def _eval_custom_router(
        self,
        client: NotDiamond,
        llm_configs: List[LLMConfig],
        joint_df: pd.DataFrame,
        prompt_column: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        eval_results = OrderedDict()
        eval_results[prompt_column] = []
        eval_results["session_id"] = []
        eval_results["notdiamond/score"] = []
        eval_results["notdiamond/cost"] = []
        eval_results["notdiamond/latency"] = []
        eval_results["notdiamond/response"] = []
        eval_results["notdiamond/recommended_provider"] = []

        for provider in llm_configs:
            provider_score_column = (
                f"{provider.provider}/{provider.model}/score"
            )
            eval_results[provider_score_column] = []

            provider_response_column = (
                f"{provider.provider}/{provider.model}/response"
            )
            eval_results[provider_response_column] = []

            provider_cost_column = f"{provider.provider}/{provider.model}/cost"
            eval_results[provider_cost_column] = []

            provider_latency_column = (
                f"{provider.provider}/{provider.model}/latency"
            )
            eval_results[provider_latency_column] = []

        for _, row in tqdm(joint_df.iterrows(), total=len(joint_df)):
            prompt = row[prompt_column]
            eval_results[prompt_column].append(prompt)

            session_id, nd_provider = client.chat.completions.model_select(
                messages=[{"role": "user", "content": prompt}], timeout=60
            )
            if nd_provider is None:
                continue

            eval_results["session_id"].append(session_id)

            provider_matched = False
            for provider in llm_configs:
                provider_score = row[
                    f"{provider.provider}/{provider.model}/score"
                ]
                eval_results[
                    f"{provider.provider}/{provider.model}/score"
                ].append(provider_score)

                provider_response = row[
                    f"{provider.provider}/{provider.model}/response"
                ]
                eval_results[
                    f"{provider.provider}/{provider.model}/response"
                ].append(provider_response)

                provider_cost = self._get_cost(
                    provider, prompt, provider_response
                )
                eval_results[
                    f"{provider.provider}/{provider.model}/cost"
                ].append(provider_cost)

                provider_latency = self._get_latency(provider, prompt)
                eval_results[
                    f"{provider.provider}/{provider.model}/latency"
                ].append(provider_latency)

                if (
                    not provider_matched
                    and provider.provider == nd_provider.provider
                    and provider.model == nd_provider.model
                ):
                    provider_matched = True
                    eval_results["notdiamond/score"].append(provider_score)
                    eval_results["notdiamond/cost"].append(provider_cost)
                    eval_results["notdiamond/latency"].append(provider_latency)
                    eval_results["notdiamond/response"].append(
                        provider_response
                    )
                    eval_results["notdiamond/recommended_provider"].append(
                        f"{nd_provider.provider}/{nd_provider.model}"
                    )

            if not provider_matched:
                raise ValueError(
                    f"""
                    Custom router returned {nd_provider.provider}/{nd_provider.model}
                    which is not in the set of models in the test dataset
                    """
                )

        eval_results_df = pd.DataFrame(eval_results)

        eval_stats = OrderedDict()
        best_average_provider = None
        best_average_score = -(2 * int(self.maximize) - 1) * np.inf

        nd_average_score = eval_results_df["notdiamond/score"].mean()
        eval_stats["Not Diamond Average Score"] = [nd_average_score]

        nd_average_cost = eval_results_df["notdiamond/cost"].mean()
        eval_stats["Not Diamond Average Cost"] = [nd_average_cost]

        nd_average_latency = eval_results_df["notdiamond/latency"].mean()
        eval_stats["Not Diamond Average Latency"] = [nd_average_latency]

        for provider in llm_configs:
            provider_avg_score = eval_results_df[
                f"{provider.provider}/{provider.model}/score"
            ].mean()
            eval_stats[f"{provider.provider}/{provider.model}/avg_score"] = [
                provider_avg_score
            ]

            provider_avg_cost = eval_results_df[
                f"{provider.provider}/{provider.model}/cost"
            ].mean()
            eval_stats[f"{provider.provider}/{provider.model}/avg_cost"] = [
                provider_avg_cost
            ]

            provider_avg_latency = eval_results_df[
                f"{provider.provider}/{provider.model}/latency"
            ].mean()
            eval_stats[f"{provider.provider}/{provider.model}/avg_latency"] = [
                provider_avg_latency
            ]

            if self.maximize:
                if provider_avg_score > best_average_score:
                    best_average_score = provider_avg_score
                    best_average_cost = provider_avg_cost
                    best_average_latency = provider_avg_latency
                    best_average_provider = (
                        f"{provider.provider}/{provider.model}"
                    )
            else:
                if provider_avg_score < best_average_score:
                    best_average_score = provider_avg_score
                    best_average_cost = provider_avg_cost
                    best_average_latency = provider_avg_latency
                    best_average_provider = (
                        f"{provider.provider}/{provider.model}"
                    )

        eval_stats["Best Average Provider"] = [best_average_provider]
        eval_stats["Best Provider Average Score"] = [best_average_score]
        eval_stats["Best Provider Average Cost"] = [best_average_cost]
        eval_stats["Best Provider Average Latency"] = [best_average_latency]

        first_columns = [
            "Best Average Provider",
            "Best Provider Average Score",
            "Best Provider Average Cost",
            "Best Provider Average Latency",
            "Not Diamond Average Score",
            "Not Diamond Average Cost",
            "Not Diamond Average Latency",
        ]
        column_order = first_columns + [
            col for col in eval_stats.keys() if col not in first_columns
        ]
        ordered_eval_stats = OrderedDict()
        for col in column_order:
            ordered_eval_stats[col] = eval_stats[col]

        eval_stats_df = pd.DataFrame(ordered_eval_stats)
        return eval_results_df, eval_stats_df

    def eval(
        self,
        dataset: Dict[Union[str, LLMConfig], pd.DataFrame],
        prompt_column: str,
        response_column: str,
        score_column: str,
        preference_id: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Method to evaluate a custom router using provided dataset.

        Parameters:
            dataset (Dict[str, pandas.DataFrame]): The dataset to train a custom router.
                Each key in the dictionary should be in the form of <provider>/<model>.
            prompt_column (str): The column name in each DataFrame corresponding
                to the prompts used to evaluate the LLM.
            response_column (str): The column name in each DataFrame corresponding
                to the response given by the LLM for a given prompt.
            score_column (str): The column name in each DataFrame corresponding
                to the score given to the response from the LLM.
            preference_id (str): The preference_id associated with the custom router
                returned from .fit().

        Raises:
            ApiError: When the NotDiamond API fails
            ValueError: When parsing the provided dataset fails
            UnsupportedLLMProvider: When a provider specified in the dataset is not supported.

        Returns:
            Tuple[pandas.DataFrame, pandas.DataFrame]:
                eval_results_df: A DataFrame containing all the prompts, responses of each provider
                    (indicated by column <provider>/<model>/response), scores of each provider
                    (indicated by column <provider>/<model>/score), and notdiamond custom router
                    response and score (indicated by column notdiamond/response and notdiamond/score).
                eval_stats_df: A DataFrame containing the "Best Average Provider" computed from the
                    provided dataset, the "Best Provider Average Score" achieved by the "Best Average Provider",
                    and the "Not Diamond Average Score" achieved through custom router.
        """

        joint_df, llm_configs = self._prepare_joint_dataset(
            dataset, prompt_column, response_column, score_column
        )

        client = NotDiamond(
            llm_configs=llm_configs,
            api_key=self.api_key,
            preference_id=preference_id,
        )

        eval_results_df, eval_stats_df = self._eval_custom_router(
            client, llm_configs, joint_df, prompt_column
        )
        return eval_results_df, eval_stats_df
