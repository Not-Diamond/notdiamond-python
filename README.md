# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/Not-Diamond/notdiamond-python/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                         |    Stmts |     Miss |   Branch |   BrPart |   Cover |   Missing |
|------------------------------------------------------------- | -------: | -------: | -------: | -------: | ------: | --------: |
| notdiamond/\_\_init\_\_.py                                   |       13 |        3 |        2 |        1 |     73% |     15-17 |
| notdiamond/\_init.py                                         |       16 |        1 |        8 |        1 |     92% |       115 |
| notdiamond/\_utils.py                                        |       45 |       12 |       16 |        3 |     69% |29-30, 48-49, 62-63, 70, 75-85 |
| notdiamond/callbacks.py                                      |       19 |        4 |        0 |        0 |     79% |     16-22 |
| notdiamond/exceptions.py                                     |        7 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/\_\_init\_\_.py                              |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/client.py                                    |      525 |      110 |      206 |       29 |     78% |131, 223->226, 257-260, 311->314, 389, 392-395, 405, 442->441, 452-453, 495->500, 501, 530-537, 619-621, 625, 645-647, 777-793, 880->883, 945-996, 1064-1065, 1083->1086, 1104-1108, 1111, 1148-1201, 1261-1262, 1297-1300, 1303, 1381-1382, 1417-1420, 1423, 1442->exit, 1459-1464, 1506-1511, 1574-1579, 1619->1621, 1645-1667, 1678, 1694, 1701, 1704, 1717, 1725, 1731, 1737, 1750, 1753, 1756, 1761, 1764, 1792-1797 |
| notdiamond/llms/config.py                                    |       97 |       18 |       24 |        7 |     78% |108, 148, 155, 187-189, 207, 211, 264, 287, 290, 293-297, 300, 303-305, 323, 327 |
| notdiamond/llms/providers.py                                 |       73 |        0 |        0 |        0 |    100% |           |
| notdiamond/llms/request.py                                   |       87 |       12 |       22 |        3 |     83% |183->199, 191-197, 261->281, 273-279, 327-331, 352 |
| notdiamond/metrics/\_\_init\_\_.py                           |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/metrics/metric.py                                 |       20 |        2 |        4 |        2 |     83% |15, 25->27, 29 |
| notdiamond/metrics/request.py                                |       22 |        3 |        2 |        1 |     83% | 36-37, 45 |
| notdiamond/prompts.py                                        |       29 |        2 |       12 |        2 |     90% |23-24, 27->29 |
| notdiamond/settings.py                                       |       18 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/\_\_init\_\_.py                           |        2 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/\_retry.py                                |      240 |       24 |       64 |       11 |     87% |72, 167->172, 182->exit, 191, 202, 215, 228, 255-259, 271->exit, 293->exit, 321, 324-330, 378, 420, 438, 489, 498-499, 531-534 |
| notdiamond/toolkit/custom\_router.py                         |      172 |        9 |       52 |        6 |     92% |37->39, 76, 247, 300, 347->323, 356-363 |
| notdiamond/toolkit/langchain.py                              |      118 |       13 |       30 |        8 |     86% |63, 71->82, 86-87, 116, 123, 131, 139, 147, 155, 189->191, 235, 250->exit, 281, 313, 319 |
| notdiamond/toolkit/litellm/\_\_init\_\_.py                   |       79 |        0 |        2 |        0 |    100% |           |
| notdiamond/toolkit/litellm/litellm\_notdiamond.py            |       85 |       13 |       30 |        6 |     80% |116-122, 145-155, 179, 208->210, 210->212, 212->214, 237->236, 286 |
| notdiamond/toolkit/litellm/main.py                           |      787 |      605 |      452 |       55 |     19% |102, 105, 110, 113, 120, 125, 128, 137-148, 174-181, 188-190, 193, 196, 208-209, 212-213, 216-217, 219-220, 223-224, 227-228, 231-234, 237-238, 241-242, 245-248, 255-256, 259-272, 281-282, 284-285, 292-297, 302->312, 313, 321, 327-519, 643->647, 698-700, 704, 710-716, 722-724, 907, 908->910, 913, 915-920, 922, 930, 932-933, 945->956, 961, 962->966, 970, 987-988, 1011-1025, 1032, 1037->1040, 1073-1076, 1114, 1130-1199, 1251-3064, 3068-3070 |
| notdiamond/toolkit/openai.py                                 |       64 |        4 |        8 |        2 |     92% |56, 59, 62, 91->94, 95 |
| notdiamond/toolkit/rag/\_\_init\_\_.py                       |        0 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/rag/document\_loaders.py                  |        1 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/rag/evaluation.py                         |       87 |       16 |       16 |        0 |     79% |     50-80 |
| notdiamond/toolkit/rag/evaluation\_dataset.py                |       35 |       10 |        6 |        2 |     66% |51, 55, 62-65, 68, 71-72, 81, 87 |
| notdiamond/toolkit/rag/llms.py                               |       29 |        7 |       12 |        1 |     66% |     53-67 |
| notdiamond/toolkit/rag/metrics.py                            |        3 |        0 |        0 |        0 |    100% |           |
| notdiamond/toolkit/rag/testset.py                            |       50 |        5 |       18 |        7 |     82% |134, 159, 165, 171->194, 176, 180, 196->195 |
| notdiamond/toolkit/rag/workflow.py                           |       96 |       11 |       30 |        6 |     87% |104, 117, 123, 135, 143, 155, 162, 186, 232, 235, 238 |
| notdiamond/types.py                                          |       28 |        0 |        4 |        0 |    100% |           |
| tests/conftest.py                                            |      109 |       13 |       22 |        3 |     88% |55, 73, 170-185, 195, 204, 211-212, 218-219 |
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
| tests/test\_documentation/test\_structured\_output.py        |       46 |        4 |        4 |        1 |     90% |49, 57-58, 110 |
| tests/test\_init.py                                          |      128 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_anthropic.py                    |      136 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_cohere.py                       |       38 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_google.py                       |      197 |      164 |        0 |        0 |     17% |13-25, 28-40, 43-54, 57-68, 71-83, 86-98, 101-115, 120-134, 137-150, 155-168, 171-185, 188-202, 207-221, 224-237, 242-255, 258-272, 277-291, 296-310, 313-326, 331-344, 347-361, 364-378, 383-398, 401-414, 419-432 |
| tests/test\_llm\_calls/test\_mistral.py                      |      170 |        2 |        0 |        0 |     99% |  181, 200 |
| tests/test\_llm\_calls/test\_openai.py                       |       45 |        0 |        6 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_openai\_o1.py                   |       12 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_perplexity.py                   |       14 |        0 |        0 |        0 |    100% |           |
| tests/test\_llm\_calls/test\_replicate.py                    |       43 |       30 |        0 |        0 |     30% |13-24, 27-38, 41-54, 57-70, 73-86 |
| tests/test\_llm\_calls/test\_togetherai.py                   |       79 |       17 |        0 |        0 |     78% |43-56, 60-73, 159-171 |
| tests/test\_toolkit/langchain/test\_integration.py           |       54 |        0 |        0 |        0 |    100% |           |
| tests/test\_toolkit/langchain/test\_unit.py                  |      130 |        1 |        8 |        2 |     98% |142, 200->202 |
| tests/test\_toolkit/rag/conftest.py                          |       47 |        0 |        0 |        0 |    100% |           |
| tests/test\_toolkit/rag/test\_data\_gen.py                   |       43 |        0 |        0 |        0 |    100% |           |
| tests/test\_toolkit/rag/test\_evaluation.py                  |       24 |        0 |        0 |        0 |    100% |           |
| tests/test\_toolkit/rag/test\_example\_workflow.py           |       53 |        1 |        2 |        0 |     98% |        53 |
| tests/test\_toolkit/rag/test\_workflow.py                    |       28 |        2 |        0 |        0 |     93% |    14, 40 |
| tests/test\_toolkit/test\_custom\_router.py                  |      139 |        0 |        8 |        0 |    100% |           |
| tests/test\_toolkit/test\_litellm.py                         |       62 |       15 |        6 |        0 |     75% |100-117, 138-139, 177-178, 196-197, 223-224 |
| tests/test\_toolkit/test\_openai\_client.py                  |       83 |        0 |       12 |        4 |     96% |104->108, 105->104, 210->214, 211->210 |
| tests/test\_toolkit/test\_retry.py                           |      326 |        0 |        0 |        0 |    100% |           |
| tests/test\_types.py                                         |       16 |        0 |        0 |        0 |    100% |           |
|                                                    **TOTAL** | **5661** | **1158** | **1172** |  **171** | **75%** |           |


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