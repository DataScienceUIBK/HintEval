(get-started-evaluation)=
# Evaluating Your Hint Dataset

Once your hint dataset is ready—whether you've created your own or used the [synthetic dataset generation module](get-started-dataset-generation)—it's time to evaluate the hints. This guide will help you set up HintEval quickly so you can focus on improving your hint generation pipelines.

You'll need your TogetherAI API key to run some of the metrics.

```python
api_key = "your-api-key"
base_url = "https://api.together.xyz/v1"
```

:::{note}
By default, these metrics use TogetherAI's API to compute the score. If you prefer to run them locally, you can set `api_key` to `None`. You can also try other platforms by changing the `base_url` accordingly.
:::

Let's start by loading the data.

## The Data

For this tutorial, we'll use the synthetic dataset generated in the [synthetic dataset generation module](get-started-dataset-generation). Alternatively, you can load a preprocessed dataset using the [Dataset.download_and_load_dataset()](../references/dataset.rst#hinteval.cores.dataset.dataset.Dataset.download_and_load_dataset) function. Each subset of the dataset includes a number of `Instance` objects, each of which contains:
- **question**: The question to evaluate along with its answers and hints.
- **answers**: A list of correct answers for the question.
- **hints**: A list of hints to help arrive at the correct answers.

```python
from hinteval import Dataset

dataset = Dataset.load('./synthetic_hint_dataset.pickle')
```

## Metrics

HintEval provides several metrics to evaluate different aspects of the hints:

1. **Relevance**: Measures how relevant the hints are to the question.
2. **Readability**: Assesses the readability of the hints and questions.
3. **Convergence**: Evaluates how well the hints narrow down potential answers to the question.
4. **Familiarity**: Measures how common or well-known the information in the hints is.
5. **AnswerLeakage**: Determines how much the hints reveal the correct answers.

Each metric contains various methods, which you can explore in the [metrics guide](core-concepts-metric).

Let's import the metrics:

```python
from hinteval.evaluation.relevance import Rouge
from hinteval.evaluation.readability import MachineLearningBased
from hinteval.evaluation.convergence import LlmBased
from hinteval.evaluation.familiarity import Wikipedia
from hinteval.evaluation.answer_leakage import ContextualEmbeddings
```
Here we’re using five metrics, but what do they represent?

1. **Relevance (_Rouge_)**: Measures the relevance of the hints to the question using ROUGE-L algorithms.
2. **Readability (_MachineLearningBased_)**: Uses a Random Forest algorithm to measure the readability of the hints and questions.
3. **Convergence (_LLmBased_)**: Assesses how well the hints help eliminate incorrect answers using the Meta LLaMA-3.1-70b-Instruct-Turbo model.
4. **Familiarity (_Wikipedia_)**: Evaluates the familiarity of the information in the hints, questions, and answers based on Wikipedia view counts.
5. **AnswerLeakage (_ContextualEmbeddings_)**: Measures how much the hints reveal the answers by calculating the similarity between the hints and answers using contextual embeddings.

To explore other metrics, check the [metrics guide](core-concepts-metric).

## Evaluation

First, extract the `question`, `hints`, and `answers` from each instance in the dataset and store them in separate lists.

```python
instances = dataset['entire'].get_instances()
questions = [instance.question for instance in instances]
answers = []
[answers.extend(instance.answers) for instance in instances]
hints = []
[hints.extend(instance.hints) for instance in instances]
```

To evaluate the dataset, call the `evaluate` method for each metric.

```python
Rouge('rougeL', enable_tqdm=True).evaluate(instances)
MachineLearningBased('random_forest', enable_tqdm=True).evaluate(questions + hints)
LlmBased('llama-3-70b', together_ai_api_key=api_key, enable_tqdm=True).evaluate(instances)
Wikipedia(enable_tqdm=True).evaluate(questions + hints + answers)
ContextualEmbeddings(enable_tqdm=True).evaluate(instances)
```

There you have it—all the evaluation scores you need.

:::{note}
Depending on the LLM provider you're using, you may need to configure parameters such as `model`, `max_tokens`, and `batch_size` based on the provider’s rate limits.
:::

## Exporting the Results

If you want to further analyze the results, you can export the evaluated dataset to a JSON file, including the scores.

```python
dataset.store_json('./evaluated_synthetic_hint_dataset.json')
```

:::{note}
The evaluated scores and metrics are automatically stored in the dataset. Saving the dataset will also save the scores.
:::

That's it! If you have any feedback, questions, or suggestions, feel free to raise an issue in the [GitHub repository](https://github.com/DataScienceUIBK/HintEval/issues). We appreciate your input!