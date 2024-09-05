import uuid
from unittest.mock import Mock, patch

import pytest

from notdiamond.exceptions import UnsupportedLLMProvider
from notdiamond.llms.config import LLMConfig
from notdiamond.toolkit import CustomRouter


class Test_CustomRouter:
    def test_custom_router(self, custom_router_dataset):
        (
            dataset,
            prompt_column,
            response_column,
            score_column,
        ) = custom_router_dataset
        custom_router = CustomRouter()

        preference_id = custom_router.fit(
            dataset=dataset,
            prompt_column=prompt_column,
            response_column=response_column,
            score_column=score_column,
        )
        assert isinstance(preference_id, str)

    def test_custom_router_and_model(self, custom_router_and_model_dataset):
        (
            dataset,
            prompt_column,
            response_column,
            score_column,
        ) = custom_router_and_model_dataset
        custom_router = CustomRouter()

        preference_id = custom_router.fit(
            dataset=dataset,
            prompt_column=prompt_column,
            response_column=response_column,
            score_column=score_column,
        )
        assert isinstance(preference_id, str)

    def test_custom_router_score_column_error(self, custom_router_dataset):
        dataset, prompt_column, response_column, _ = custom_router_dataset
        score_column = " "
        custom_router = CustomRouter()

        with pytest.raises(ValueError):
            _ = custom_router.fit(
                dataset=dataset,
                prompt_column=prompt_column,
                response_column=response_column,
                score_column=score_column,
            )

    def test_custom_router_response_column_error(self, custom_router_dataset):
        dataset, prompt_column, _, score_column = custom_router_dataset
        response_column = " "
        custom_router = CustomRouter()

        with pytest.raises(ValueError):
            _ = custom_router.fit(
                dataset=dataset,
                prompt_column=prompt_column,
                response_column=response_column,
                score_column=score_column,
            )

    def test_custom_router_prompt_column_error(self, custom_router_dataset):
        dataset, _, response_column, score_column = custom_router_dataset
        prompt_column = " "
        custom_router = CustomRouter()

        with pytest.raises(ValueError):
            _ = custom_router.fit(
                dataset=dataset,
                prompt_column=prompt_column,
                response_column=response_column,
                score_column=score_column,
            )

    def test_custom_router_invalid_provider(self, custom_router_dataset):
        (
            dataset,
            prompt_column,
            response_column,
            score_column,
        ) = custom_router_dataset
        custom_router = CustomRouter()
        a_provider = list(dataset.keys())[0]
        invalid_provider = "anthropic/haiku-20240307"
        dataset[invalid_provider] = dataset[a_provider]

        with pytest.raises(UnsupportedLLMProvider):
            _ = custom_router.fit(
                dataset=dataset,
                prompt_column=prompt_column,
                response_column=response_column,
                score_column=score_column,
            )

    def test_eval_custom_router(self, custom_router_dataset, mocker):
        mock_NDLLM = mocker.patch(
            "notdiamond.toolkit.custom_router.NotDiamond",
            autospec=True,
        )
        mock_NDLLM._parse_llm_configs_data.return_value = [
            LLMConfig(provider="openai", model="gpt-3.5-turbo"),
            LLMConfig(provider="anthropic", model="claude-3-haiku-20240307"),
        ]

        mock_chat = mocker.Mock()
        mock_completions = mocker.Mock()
        mock_model_select = mocker.Mock()

        mock_instance = mock_NDLLM.return_value
        mock_instance.chat = mock_chat
        mock_chat.completions = mock_completions
        mock_completions.model_select = mock_model_select
        mock_model_select.return_value = (
            str(uuid.uuid4()),
            LLMConfig(provider="openai", model="gpt-3.5-turbo"),
        )
        (
            dataset,
            prompt_column,
            response_column,
            score_column,
        ) = custom_router_dataset

        with patch.multiple(
            "notdiamond.llms.client", NDApiKeyValidator=Mock(return_value=True)
        ):
            custom_router = CustomRouter()

            eval_results_df, eval_stats_df = custom_router.eval(
                dataset=dataset,
                prompt_column=prompt_column,
                response_column=response_column,
                score_column=score_column,
                preference_id="abc",
            )

            assert eval_results_df.shape[0] == 15
            assert "notdiamond/score" in eval_results_df.columns
            assert "notdiamond/response" in eval_results_df.columns
            for provider in dataset.keys():
                assert f"{provider}/score" in eval_results_df.columns
                assert f"{provider}/response" in eval_results_df.columns

            assert "Best Average Provider" in eval_stats_df.columns
            assert len(eval_stats_df["Best Average Provider"]) == 1
            assert eval_stats_df["Best Average Provider"][0] in list(
                dataset.keys()
            )

            assert "Best Provider Average Score" in eval_stats_df.columns
            assert len(eval_stats_df["Best Provider Average Score"]) == 1
            assert isinstance(
                eval_stats_df["Best Provider Average Score"][0], float
            )

            assert "Not Diamond Average Score" in eval_stats_df.columns
            assert len(eval_stats_df["Not Diamond Average Score"]) == 1
            assert isinstance(
                eval_stats_df["Not Diamond Average Score"][0], float
            )
