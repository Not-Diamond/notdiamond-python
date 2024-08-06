# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Not-Diamond/notdiamond-python/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                     |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|--------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| notdiamond/\_\_init\_\_.py                               |        3 |        0 |        0 |        0 |    100% |           |
| notdiamond/\_utils.py                                    |       43 |       12 |       18 |        4 |     67% |29-30, 48-49, 59->exit, 62-63, 70, 75-85 |
| notdiamond/callbacks.py                                  |       19 |        4 |        0 |        0 |     79% |     16-22 |
| notdiamond/exceptions.py                                 |        6 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/\_\_init\_\_.py                          |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/client.py                                |      491 |       90 |      212 |       31 |     79% |90->92, 108, 181->184, 185-186, 211-214, 262->265, 336, 339-342, 352, 389->388, 399-400, 442->447, 448, 477-484, 541-543, 547, 567-569, 690-704, 788->791, 848-899, 964-965, 983->986, 1001-1005, 1008, 1043-1096, 1153-1154, 1186-1189, 1192, 1267-1268, 1300-1303, 1306, 1342-1347, 1387-1392, 1450-1455, 1495->1497, 1530->1532, 1543, 1554, 1637-1642 |
| notdiamond/llms/config.py                                |       44 |        5 |       14 |        2 |     88% |72, 97, 104, 125-127 |
| notdiamond/llms/providers.py                             |       56 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/request.py                               |       81 |       10 |       18 |        1 |     89% |173-175, 237-239, 285-289, 308 |
| notdiamond/metrics/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/metrics/metric.py                             |       20 |        2 |        4 |        2 |     83% |15, 24->26, 28 |
| notdiamond/metrics/request.py                            |       21 |        3 |        2 |        1 |     83% | 37-38, 46 |
| notdiamond/prompts.py                                    |       18 |        2 |        6 |        2 |     83% |20-21, 24->26 |
| notdiamond/settings.py                                   |       16 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/\_\_init\_\_.py                       |        2 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/custom\_router.py                     |      116 |        6 |       34 |        5 |     91% |35->37, 73, 198, 233, 255-257 |
| notdiamond/types.py                                      |       28 |        0 |       16 |        0 |    100% |           |
| tests/conftest.py                                        |       73 |        8 |       26 |        0 |     92% |55, 73, 134-149, 159 |
| tests/helpers.py                                         |       22 |        0 |        8 |        2 |     93% |4->11, 17->24 |
| tests/test\_components/test\_llms/test\_callbacks.py     |       16 |        0 |        2 |        0 |    100% |           |
| tests/test\_components/test\_llms/test\_llm.py           |      474 |       24 |      192 |        2 |     95% |28-32, 225-240, 273, 310, 348, 391, 405, 433, 546, 698->exit, 717, 720->exit, 955-974 |
| tests/test\_components/test\_llms/test\_llm\_request.py  |       31 |        0 |        2 |        0 |    100% |           |
| tests/test\_components/test\_llms/test\_provider.py      |       31 |        0 |        2 |        0 |    100% |           |
| tests/test\_documentation/test\_fallback\_and\_custom.py |       26 |        3 |        4 |        2 |     83% |69->exit, 70->exit, 82-84 |
| tests/test\_documentation/test\_function\_calling.py     |       34 |        1 |        6 |        0 |     98% |        13 |
| tests/test\_documentation/test\_getting\_started.py      |       33 |        0 |        2 |        0 |    100% |           |
| tests/test\_documentation/test\_langchain.py             |        6 |        0 |        2 |        0 |    100% |           |
| tests/test\_documentation/test\_openrouter.py            |       14 |        0 |        0 |        0 |    100% |           |
| tests/test\_documentation/test\_personalization.py       |       10 |        0 |        0 |        0 |    100% |           |
| tests/test\_documentation/test\_structured\_output.py    |       43 |       19 |        4 |        0 |     55% |46-56, 62-69 |
| tests/test\_llm\_calls/test\_anthropic.py                |       87 |        0 |        4 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_cohere.py                   |       37 |        0 |        2 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_google.py                   |      166 |      138 |        5 |        0 |     19% |12-24, 27-39, 42-53, 56-67, 70-82, 85-97, 100-114, 119-133, 136-149, 154-167, 170-184, 187-201, 206-220, 223-236, 241-254, 257-271, 276-290, 295-309, 312-325, 330-343, 346-360 |
| tests/test\_llm\_calls/test\_mistral.py                  |      141 |        0 |        2 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_openai.py                   |       40 |        0 |        9 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_perplexity.py               |       13 |        0 |        2 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_replicate.py                |       41 |        0 |        2 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_togetherai.py               |       70 |        6 |        4 |        0 |     92% |     42-55 |
| tests/test\_toolkit/test\_custom\_router.py              |       68 |        0 |       12 |        0 |    100% |           |
| tests/test\_types.py                                     |       16 |        0 |        6 |        0 |    100% |           |
|                                                **TOTAL** | **2456** |  **333** |  **622** |   **54** | **86%** |           |


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