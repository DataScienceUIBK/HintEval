(core-concepts-metric-readability)=

# Readability

The **Readability** metric measures how easy or difficult it is to understand a hint or question. This metric is
important to ensure that hints guide the user effectively and are understandable without causing confusion.

The HintEval framework provides four different methods for computing the Readability metric.

- [**Traditional**](#traditional)
- [**Machine-Learning**](#machine-learning)
- [**Neural-Network**](#neural-network)
- [**LLM**](#llm)

:::{note}
The `evaluate` function takes a list of `Hint` or `Question` objects as its input, where each object contains the text
that needs to be evaluated for readability.
:::

## Traditional

The [Traditional](../../references/metrics/readability.rst#hinteval.cores.evaluation_metrics.readability.TraditionalIndexes) method evaluates readability using classic readability formulas that have been widely adopted in
educational and linguistic fields. These formulas compute readability based on factors like sentence length, word
length, and complexity. For more information, refer to the [📝original paper](https://aclanthology.org/2023.bea-1.1/).

Traditional methods available:

- **Gunning Fog Index** (G-Fox): Measures the number of years of formal education a person needs to understand the text.
- **Flesch Reading Ease**: Rates text on a 100-point scale, with higher scores indicating easier readability.
- **Coleman-Liau Index**: Analyzes sentence length and character count to determine the readability score.
- **SMOG Index**: Estimates the years of education a person needs to comprehend the text based on complex words.
- **Automated Readability Index (ARI)**: Focuses on the readability of technical documents and uses sentence and word
  length to determine the score.

### Example

```python
from hinteval.cores import Question, Hint
from hinteval.evaluation.readability import TraditionalIndexes

traditional_indexes = TraditionalIndexes(method='flesch_kincaid_reading_ease')
sentence_1 = Question('What is the capital of Austria?')
sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
sentences = [sentence_1, sentence_2]
results = traditional_indexes.evaluate(sentences)
print(results)
# [87.945, 69.994]
metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
           sent.metrics.items()]
print(metrics)
# ['readability-flesch_kincaid_reading_ease-sm: 87.945', 'readability-flesch_kincaid_reading_ease-sm: 69.994']
```

## Machine-Learning

The [Machine-Learning](../../references/metrics/readability.rst#hinteval.cores.evaluation_metrics.readability.MachineLearningBased) method evaluates readability using  classic trained machine learninig models to predict readability scores based on text
features. These models are trained using labeled datasets that map text to a specific readability score. For more
information, refer to the [📝original paper](https://aclanthology.org/2023.bea-1.37/).

Machine-Learning methods available:

- **XGBoost**: A gradient boosting algorithm known for its speed and accuracy in machine learning tasks.
- **Random-Forest**: A popular ensemble learning method that builds multiple decision trees to improve prediction
  accuracy.

### Example

```python
from hinteval.cores import Question, Hint
from hinteval.evaluation.readability import MachineLearningBased

machine_learning = MachineLearningBased(method='xgboost')
sentence_1 = Question('What is the capital of Austria?')
sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
sentences = [sentence_1, sentence_2]
results = machine_learning.evaluate(sentences)
print(results)
# [0, 0]
classes = [sent.metrics['readability-ml-xgboost-sm'].metadata['description'] for sent in sentences]
print(classes)
# ['beginner', 'beginner']
metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
           sent.metrics.items()]
print(metrics)
# ['readability-ml-xgboost-sm: 0', 'readability-ml-xgboost-sm: 0']
```

## Neural-Network

The [Neural-Network](../../references/metrics/readability.rst#hinteval.cores.evaluation_metrics.readability.NeuralNetworkBased) method uses pre-trained transformer models to evaluate readability. These models are highly
effective in understanding the deeper context and structure of the text, which allows for more nuanced readability
evaluations. For more information, refer to the [📝original paper](https://aclanthology.org/2023.bea-1.37/).

Neural-Network methods available:

- **BERT-base**: A popular model that can capture contextual relationships between words and is trained on a large
  corpus of English text.
- **RoBERTa-large**: A more advanced variant of BERT that is trained on more data, providing better accuracy for
  readability predictions.

### Example

```python
from hinteval.cores import Question, Hint
from hinteval.evaluation.readability import NeuralNetworkBased

neural_network = NeuralNetworkBased(model_name='bert-base')
sentence_1 = Question('What is the capital of Austria?')
sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
sentences = [sentence_1, sentence_2]
results = neural_network.evaluate(sentences)
print(results)
# [0, 0]
classes = [sent.metrics['readability-nn-bert-base'].metadata['description'] for sent in sentences]
print(classes)
# ['beginner', 'beginner']
metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
           sent.metrics.items()]
print(metrics)
# ['readability-nn-bert-base: 0', 'readability-nn-bert-base: 0']

```

## LLM

The [LLM](../../references/metrics/readability.rst#hinteval.cores.evaluation_metrics.readability.LlmBased) method leverages large language models to evaluate the readability of a hint or question. By using models
like GPT-4 or Meta-LLaMA, this approach provides highly accurate and context-aware readability scores, allowing for a
deeper understanding of how easily the text can be comprehended. For more information, refer to
the [📝original paper](https://arxiv.org/abs/2305.14463).

:::{note}
We assume you have an active API key for the TogetherAI platform and are using this platform for readability evaluation
using LLM. In this example, we use *meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo* as the model, which is available on
the TogetherAI platform. If you wish to use another platform, ensure the model name is valid for that platform.

For local execution, you can set `api_key` to `None`. HintEval supports running large language models (LLMs) locally
via [HuggingFace](https://huggingface.co/models).
:::

:::{Warning}
The output may vary from the example shown below due to the inherent non-deterministic nature of large language models.
:::

### Example

```python
from hinteval.cores import Question, Hint
from hinteval.evaluation.readability import LlmBased

llm = LlmBased(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
               api_key='your_api_key', batch_size=2)
sentence_1 = Question('What is the capital of Austria?')
sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
sentences = [sentence_1, sentence_2]
results = llm.evaluate(sentences)
print(results)
# [0, 0]
classes = [sent.metrics['readability-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo'].metadata['description'] for
           sent in sentences]
print(classes)
# ['beginner', 'beginner']
metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
           sent.metrics.items()]
print(metrics)
# ['readability-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 0', 
# 'readability-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 0']
```

## Comparison

For each method, we provide details on:

| Method               | Preferred Device | Cost-Effectiveness | Accuracy  | Execution Speed |
|----------------------|--------------|--------------------|-----------|-----------------|
| **Traditional**      | CPU          | High               | Low       | Very Fast       |
| **Machine-Learning** | CPU          | High               | Moderate  | Moderate            |
| **Neural-Network**   | GPU          | Moderate           | High      | Moderate        |
| **LLM**              | GPU          | Low                | Very High | Slow            |

- **Preferred Device**: Indicates whether the method works best on CPU or GPU.
- **Cost-Effectiveness**: Evaluates how computationally expensive the method is, considering the resources needed.
- **Accuracy**: Reflects how accurate the method is in assessing the metric.
- **Execution Speed**: How quickly the method executes (e.g., Fast, Moderate, Slow).