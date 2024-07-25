# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Not-Diamond/notdiamond-python/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                     |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| notdiamond/\_\_init\_\_.py                               |        3 |        0 |        0 |        0 |    100% |           |
| notdiamond/\_utils.py                                    |       43 |       12 |       18 |        4 |     67% |29-30, 48-49, 59->exit, 62-63, 70, 75-85 |
| notdiamond/callbacks.py                                  |       19 |        4 |        0 |        0 |     79% |     16-22 |
| notdiamond/exceptions.py                                 |        6 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/\_\_init\_\_.py                          |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/client.py                                |      488 |       88 |      212 |       30 |     79% |90->92, 108, 178->181, 182-183, 208-211, 259->262, 333, 336-339, 349, 386->385, 396-397, 439->444, 445, 474-481, 538-540, 544, 564-566, 692-712, 769->772, 829-880, 937->940, 955-959, 962, 997-1050, 1103->1106, 1120-1123, 1126, 1206->1209, 1210-1213, 1216, 1252-1257, 1297-1302, 1360-1365, 1405->1407, 1440->1442, 1453, 1464, 1547-1552 |
| notdiamond/llms/config.py                                |       44 |        5 |       14 |        2 |     88% |72, 97, 104, 125-127 |
| notdiamond/llms/providers.py                             |       54 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/request.py                               |       71 |        9 |       16 |        0 |     90% |177-179, 241-243, 293-297 |
| notdiamond/metrics/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/metrics/metric.py                             |       20 |        2 |        4 |        2 |     83% |15, 24->26, 28 |
| notdiamond/metrics/request.py                            |       21 |        3 |        2 |        1 |     83% | 37-38, 46 |
| notdiamond/prompts.py                                    |       18 |        2 |        6 |        2 |     83% |20-21, 24->26 |
| notdiamond/settings.py                                   |       16 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/\_\_init\_\_.py                       |        2 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/custom\_router.py                     |      122 |        6 |       34 |        5 |     92% |42->44, 80, 205, 240, 262-264 |
| notdiamond/types.py                                      |       28 |        0 |       16 |        0 |    100% |           |
| tests/conftest.py                                        |       73 |        8 |       26 |        0 |     92% |55, 73, 134-149, 159 |
| tests/helpers.py                                         |       22 |        0 |        8 |        2 |     93% |4->11, 17->24 |
| tests/test\_components/test\_llms/test\_callbacks.py     |       16 |        0 |        2 |        0 |    100% |           |
| tests/test\_components/test\_llms/test\_llm.py           |      467 |       24 |      190 |        2 |     95% |28-32, 225-240, 273, 310, 348, 391, 405, 433, 546, 683->exit, 702, 705->exit, 940-959 |
| tests/test\_components/test\_llms/test\_llm\_request.py  |       31 |        0 |        2 |        0 |    100% |           |
| tests/test\_components/test\_llms/test\_provider.py      |       31 |        0 |        2 |        0 |    100% |           |
| tests/test\_documentation/test\_fallback\_and\_custom.py |       26 |        3 |        4 |        2 |     83% |69->exit, 70->exit, 82-84 |
| tests/test\_documentation/test\_function\_calling.py     |       34 |        1 |        6 |        0 |     98% |        13 |
| tests/test\_documentation/test\_getting\_started.py      |       33 |        0 |        2 |        0 |    100% |           |
| tests/test\_documentation/test\_langchain.py             |        6 |        0 |        2 |        0 |    100% |           |
| tests/test\_documentation/test\_openrouter.py            |       14 |        0 |        0 |        0 |    100% |           |
| tests/test\_documentation/test\_personalization.py       |       10 |        0 |        0 |        0 |    100% |           |
| tests/test\_documentation/test\_rag.py                   |       21 |       15 |        4 |        0 |     32% |     12-54 |
| tests/test\_documentation/test\_structured\_output.py    |       43 |        2 |        4 |        0 |     96% |     55-56 |
| tests/test\_llm\_calls/test\_anthropic.py                |       87 |        0 |        4 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_cohere.py                   |       37 |        0 |        2 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_google.py                   |      128 |      105 |        5 |        0 |     21% |12-24, 27-39, 42-53, 56-67, 70-82, 85-97, 100-114, 119-133, 136-149, 154-167, 170-184, 187-201, 206-220, 223-236, 241-254, 257-271 |
| tests/test\_llm\_calls/test\_mistral.py                  |      141 |        0 |        2 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_openai.py                   |      354 |        0 |       10 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_perplexity.py               |       13 |        0 |        2 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_replicate.py                |       41 |        0 |        2 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_togetherai.py               |       70 |        6 |        4 |        0 |     92% |     42-55 |
| tests/test\_toolkit/test\_custom\_router.py              |       68 |        0 |       12 |        0 |    100% |           |
| tests/test\_types.py                                     |       16 |        0 |        6 |        0 |    100% |           |
|                                                **TOTAL** | **2737** |  **295** |  **623** |   **52** | **89%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/Not-Diamond/notdiamond-python/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/Not-Diamond/notdiamond-python/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Not-Diamond/notdiamond-python/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/Not-Diamond/notdiamond-python/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2FNot-Diamond%2Fnotdiamond-python%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/Not-Diamond/notdiamond-python/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.