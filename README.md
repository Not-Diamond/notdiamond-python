# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Not-Diamond/notdiamond-python/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| notdiamond/\_\_init\_\_.py                                   |        3 |        0 |        0 |        0 |    100% |           |
| notdiamond/\_utils.py                                        |       45 |       12 |       16 |        3 |     69% |29-30, 48-49, 62-63, 70, 75-85 |
| notdiamond/callbacks.py                                      |       19 |        4 |        0 |        0 |     79% |     16-22 |
| notdiamond/exceptions.py                                     |        7 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/\_\_init\_\_.py                              |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/client.py                                    |      519 |      102 |      206 |       26 |     79% |126, 215->218, 301->304, 378, 381-384, 394, 431->430, 441-442, 484->489, 490, 519-526, 604-606, 610, 630-632, 759-774, 860->863, 924-975, 1042-1043, 1061->1064, 1081-1085, 1088, 1125-1178, 1237-1238, 1278, 1355-1356, 1396, 1415->exit, 1432-1437, 1479-1484, 1547-1552, 1592->1594, 1618-1640, 1651, 1667, 1674, 1677, 1690, 1698, 1704, 1710, 1723, 1726, 1729, 1759-1764 |
| notdiamond/llms/config.py                                    |       97 |       18 |       24 |        7 |     78% |108, 148, 155, 187-189, 207, 211, 264, 287, 290, 293-297, 300, 303-305, 323, 327 |
| notdiamond/llms/providers.py                                 |       66 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/request.py                                   |       81 |        4 |       14 |        1 |     95% |307-311, 332 |
| notdiamond/metrics/\_\_init\_\_.py                           |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/metrics/metric.py                                 |       20 |        2 |        4 |        2 |     83% |15, 25->27, 29 |
| notdiamond/metrics/request.py                                |       22 |        3 |        2 |        1 |     83% | 36-37, 45 |
| notdiamond/prompts.py                                        |       29 |        2 |       12 |        2 |     90% |23-24, 27->29 |
| notdiamond/settings.py                                       |       18 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/\_\_init\_\_.py                           |        2 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/custom\_router.py                         |      162 |        8 |       34 |        5 |     92% |37->39, 76, 243, 292, 345-349 |
| notdiamond/toolkit/langchain.py                              |      118 |       13 |       30 |        8 |     86% |63, 71->82, 86-87, 116, 123, 131, 139, 147, 155, 189->191, 235, 250->exit, 281, 313, 319 |
| notdiamond/toolkit/litellm/\_\_init\_\_.py                   |       63 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/litellm/litellm\_notdiamond.py            |       81 |       11 |       30 |        6 |     81% |114-120, 143-146, 170, 199->201, 201->203, 203->205, 227->226, 276 |
| notdiamond/toolkit/litellm/main.py                           |      785 |      600 |      452 |       54 |     20% |102, 105, 110, 113, 120, 125, 128, 137-148, 174-181, 188-190, 193, 196, 208-209, 212-213, 216-217, 219-220, 223-224, 227-228, 231-234, 237-238, 241-242, 245-248, 255-256, 259-272, 281-282, 284-285, 292-297, 302->312, 313, 321, 327-519, 642->646, 697-699, 703, 709-715, 905, 911, 913-918, 920, 928, 930-931, 943->954, 959, 960->964, 968, 985-986, 1009-1023, 1030, 1035->1038, 1071-1074, 1112, 1128-1197, 1249-3062 |
| notdiamond/toolkit/openai.py                                 |       85 |        8 |       16 |        4 |     88% |55, 58, 61, 94->97, 98, 149, 152, 155, 188->191, 192 |
| notdiamond/toolkit/rag/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/rag/document\_loaders.py                  |        1 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/rag/evaluation.py                         |       87 |       16 |       16 |        0 |     79% |     50-80 |
| notdiamond/toolkit/rag/evaluation\_dataset.py                |       35 |       10 |        6 |        2 |     66% |51, 55, 62-65, 68, 71-72, 81, 87 |
| notdiamond/toolkit/rag/llms.py                               |       29 |        7 |       12 |        1 |     66% |     54-68 |
| notdiamond/toolkit/rag/metrics.py                            |        3 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/rag/testset.py                            |       50 |        5 |       18 |        7 |     82% |135, 158, 164, 170->193, 175, 179, 195->194 |
| notdiamond/toolkit/rag/workflow.py                           |       96 |       11 |       30 |        6 |     87% |104, 117, 123, 135, 143, 155, 162, 186, 232, 235, 238 |
| notdiamond/types.py                                          |       28 |        0 |        4 |        0 |    100% |           |
| tests/conftest.py                                            |      109 |       12 |       22 |        2 |     89% |55, 73, 170-185, 195, 211-212, 218-219 |
| tests/helpers.py                                             |       22 |        0 |        8 |        2 |     93% |4->11, 17->24 |
| tests/test\_components/test\_llms/test\_callbacks.py         |       19 |        0 |        0 |        0 |    100% |           |
| tests/test\_components/test\_llms/test\_embedding\_config.py |       20 |        0 |        0 |        0 |    100% |           |
| tests/test\_components/test\_llms/test\_llm.py               |      501 |       21 |       68 |        4 |     95% |28-32, 237-252, 287, 325, 364, 408, 425->432, 451, 572->578, 726->exit, 748->exit, 983-1002 |
| tests/test\_components/test\_llms/test\_llm\_request.py      |       59 |        0 |        0 |        0 |    100% |           |
| tests/test\_components/test\_llms/test\_provider.py          |       31 |        0 |        0 |        0 |    100% |           |
| tests/test\_documentation/test\_fallback\_and\_custom.py     |       30 |        3 |        4 |        2 |     85% |73->exit, 74->exit, 86-88 |
| tests/test\_documentation/test\_function\_calling.py         |       37 |        1 |        2 |        0 |     97% |        15 |
| tests/test\_documentation/test\_getting\_started.py          |       37 |        0 |        0 |        0 |    100% |           |
| tests/test\_documentation/test\_langchain.py                 |        8 |        0 |        2 |        0 |    100% |           |
| tests/test\_documentation/test\_openrouter.py                |       16 |        0 |        0 |        0 |    100% |           |
| tests/test\_documentation/test\_personalization.py           |       12 |        0 |        0 |        0 |    100% |           |
| tests/test\_documentation/test\_structured\_output.py        |       46 |       19 |        4 |        0 |     58% |48-58, 64-71 |
| tests/test\_llm\_calls/test\_anthropic.py                    |      120 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_cohere.py                       |       38 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_google.py                       |      167 |      138 |        0 |        0 |     17% |13-25, 28-40, 43-54, 57-68, 71-83, 86-98, 101-115, 120-134, 137-150, 155-168, 171-185, 188-202, 207-221, 224-237, 242-255, 258-272, 277-291, 296-310, 313-326, 331-344, 347-361 |
| tests/test\_llm\_calls/test\_mistral.py                      |      166 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_openai.py                       |       45 |        0 |        6 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_openai\_o1.py                   |       12 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_perplexity.py                   |       14 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_replicate.py                    |       43 |       30 |        0 |        0 |     30% |13-24, 27-38, 41-54, 57-70, 73-86 |
| tests/test\_llm\_calls/test\_togetherai.py                   |       72 |       12 |        0 |        0 |     83% |43-56, 60-73 |
| tests/test\_toolkit/langchain/test\_integration.py           |       54 |        0 |        0 |        0 |    100% |           |
| tests/test\_toolkit/langchain/test\_unit.py                  |      130 |        1 |        8 |        2 |     98% |142, 200->202 |
| tests/test\_toolkit/rag/conftest.py                          |       47 |        0 |        0 |        0 |    100% |           |
| tests/test\_toolkit/rag/test\_data\_gen.py                   |       43 |        0 |        0 |        0 |    100% |           |
| tests/test\_toolkit/rag/test\_evaluation.py                  |       24 |        0 |        0 |        0 |    100% |           |
| tests/test\_toolkit/rag/test\_example\_workflow.py           |       53 |        1 |        2 |        0 |     98% |        53 |
| tests/test\_toolkit/rag/test\_workflow.py                    |       28 |        2 |        0 |        0 |     93% |    14, 40 |
| tests/test\_toolkit/test\_custom\_router.py                  |       93 |        0 |        4 |        0 |    100% |           |
| tests/test\_toolkit/test\_litellm.py                         |       61 |       10 |        6 |        0 |     85% |113-114, 133-134, 170-171, 189-190, 216-217 |
| tests/test\_toolkit/test\_openai\_client.py                  |       83 |        0 |       12 |        4 |     96% |104->108, 105->104, 210->214, 211->210 |
| tests/test\_types.py                                         |       16 |        0 |        0 |        0 |    100% |           |
|                                                    **TOTAL** | **4807** | **1086** | **1074** |  **151** | **72%** |           |


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