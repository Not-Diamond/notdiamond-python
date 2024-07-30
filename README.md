# Getting started with Not Diamond

Not Diamond is an AI model router that automatically determines which LLM is best-suited to respond to any query, improving LLM output quality by combining multiple LLMs into a **meta-model** that learns when to call each LLM.

# Key features

- **[Maximize output quality](https://notdiamond.readme.io/docs/quickstart)**: Not Diamond [outperforms every foundation model](https://notdiamond.readme.io/docs/benchmark-performance) on major evaluation benchmarks by always calling the best model for every prompt.
- **[Reduce cost and latency](https://notdiamond.readme.io/docs/cost-and-latency-tradeoffs)**: Not Diamond lets you define intelligent cost and latency tradeoffs to efficiently leverage smaller and cheaper models without degrading quality.
- **[Train your own router](https://notdiamond.readme.io/docs/router-training-quickstart)**: Not Diamond lets you train your own custom routers optimized to your data and use case.
- **[Python](https://python.notdiamond.ai/), [TypeScript](https://www.npmjs.com/package/notdiamond), and [REST API](https://notdiamond.readme.io/reference/api-introduction) support**: Not Diamond works across a variety of stacks.

# Installation

**Python**: Requires **Python 3.10+**. It’s recommended that you create and activate a [virtualenv](https://virtualenv.pypa.io/en/latest/) prior to installing the package. For this example, we'll be installing the optional additional `create` dependencies, which you can learn more about [here](https://notdiamond.readme.io/docs/model_select-vs-create).

```shell
pip install notdiamond[create]
```

# Setting up

Create a `.env` file with your [Not Diamond API key](https://app.notdiamond.ai/keys) and the [API keys of the models](https://notdiamond.readme.io/docs/api-keys) you want to route between:

```shell
NOTDIAMOND_API_KEY = "YOUR_NOTDIAMOND_API_KEY"
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY"
```

# Sending your first Not Diamond API request

Create a new file in the same directory as your `.env` file and copy and run the code below (you can toggle between  Python and TypeScript in the top left of the code block):

```python
from notdiamond import NotDiamond

# Define the Not Diamond routing client
client = NotDiamond()

# The best LLM is determined by Not Diamond based on the messages and specified models
result, session_id, provider = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Concisely explain merge sort."}  # Adjust as desired
    ],
    model=['openai/gpt-3.5-turbo', 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet-20240620']
)

print("ND session ID: ", session_id)  # A unique ID of Not Diamond's recommendation
print("LLM called: ", provider.model)  # The LLM routed to
print("LLM output: ", result.content)  # The LLM response
```
