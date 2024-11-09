(core-concepts-metric-relevance)=

# Relevance

The **Relevance** metric measures how much a hint helps in finding the answer to a question. In other words, it assesses
how relevant the hint is to the question, ensuring that the hint effectively guides the user toward the correct answer.

The HintEval framework provides four different methods for computing the Relevance metric.

- [**Rouge**](#rouge)
- [**Non-Contextual**](#non-contextual)
- [**Contextual**](#contextual)
- [**LLM**](#llm)

:::{note}
The `evaluate` function takes a list of `Instance` objects as its input, where each instance contains a question and its
associated hints.
:::

## Rouge

The [Rouge](../../references/metrics/relevance.rst#hinteval.cores.evaluation_metrics.relevance.Rouge) (Recall-Oriented Understudy for Gisting Evaluation) metric evaluates the overlap between the generated hints
and the reference text. It is commonly used in NLP tasks to compare text outputs. For more details, refer to
the [📝original paper](https://aclanthology.org/W04-1013/).

Rouge is available in three main variants:

- **Rouge-1**: Measures unigram (individual word) overlap between the generated hint and the question, assessing basic
  word-level similarity.
- **Rouge-2**: Measures bigram (word pair) overlap, capturing some degree of contextual similarity by considering
  consecutive words.
- **Rouge-L**: Measures the longest common subsequence (LCS) of words, evaluating how well the word sequence of the hint
  matches the question.

### Example

```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.relevance import Rouge

rouge = Rouge(model='rouge1')
instance_1 = Instance(
    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven.'),
           Hint('This city is the best city for life in 2024.')])
instance_2 = Instance(
    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('He was the first African-American president in U.S. history.'),
           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
instances = [instance_1, instance_2]
results = rouge.evaluate(instances)
print(results)
# [[0.0, 0.25], [0.421, 0.353]]
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in
           hint.metrics.items()]
print(metrics)
# ['relevance-rouge1: 0.0', 'relevance-rouge1: 0.25', 'relevance-rouge1: 0.421', 'relevance-rouge1: 0.353']
```

## Non-Contextual

The [Non-Contextual](../../references/metrics/relevance.rst#hinteval.cores.evaluation_metrics.relevance.NonContextualEmbeddings) method computes relevance by assessing how much a hint resembles a potential answer to the question.
It calculates text similarity based on fixed word embeddings of the question and hint. For more information, refer to
the [📝original paper](https://arxiv.org/abs/1909.01059).

Non-Contextual is available in two main variants:

- **Glove 6B**: Uses 6 billion tokens trained on a variety of text sources.
- **Glove 42B**: Uses 42 billion tokens trained on a more extensive dataset, providing more robust embeddings.

### Example

```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.relevance import NonContextualEmbeddings

non_contextual = NonContextualEmbeddings(glove_version='glove.6B')
instance_1 = Instance(
    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven.'),
           Hint('This city is the best city for life in 2024.')])
instance_2 = Instance(
    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('He was the first African-American president in U.S. history.'),
           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
instances = [instance_1, instance_2]
results = non_contextual.evaluate(instances)
print(results)
# [[0.867, 0.889], [0.91, 0.891]]
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in
           hint.metrics.items()]
print(metrics)
# ['relevance-non-contextual-6B-sm: 0.867', 'relevance-non-contextual-6B-sm: 0.889', 'relevance-non-contextual-6B-sm: 0.91', 'relevance-non-contextual-6B-sm: 0.891']
```

## Contextual

The [Contextual](../../references/metrics/relevance.rst#hinteval.cores.evaluation_metrics.relevance.ContextualEmbeddings) method uses more advanced embeddings to evaluate the relevance between hints and questions. This method
captures deeper relationships between words by leveraging pre-trained transformer models, which better understand the
context in which words are used. For more information, refer to
the [📝original paper](https://aclanthology.org/2020.lrec-1.676/).

Contextual is available in two main variants:

- **BERT-base**: A widely-used model that is based on the BERT architecture, trained on lower-cased English text, and
  uses 110 million parameters.
- **RoBERTa-large**: A more robust variant of BERT, trained on a larger corpus, with 355 million parameters for more
  refined and accurate contextual embeddings.

### Example

```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.relevance import ContextualEmbeddings

contextual = ContextualEmbeddings(model_name='bert-base')
instance_1 = Instance(
    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven.'),
           Hint('This city is the best city for life in 2024.')])
instance_2 = Instance(
    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('He was the first African-American president in U.S. history.'),
           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
instances = [instance_1, instance_2]
results = contextual.evaluate(instances)
print(results)
# [[1.0, 1.0], [1.0, 1.0]]
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in
           hint.metrics.items()]
print(metrics)
# ['relevance-contextual-bert-base: 1.0', 'relevance-contextual-bert-base: 1.0', 'relevance-contextual-bert-base: 1.0', 'relevance-contextual-bert-base: 1.0']
```

## LLM

The [LLM](../../references/metrics/relevance.rst#hinteval.cores.evaluation_metrics.relevance.LlmBased) method measures how relevant the hint to the question using *Answer Relevancy* metric. We consider the hint as
*answer* and question as *prompt* in this metric. For more information, refer to
the [📝original paper](https://aclanthology.org/2024.eacl-demo.16/).

:::{note}
We assume you have an active API key for the TogetherAI platform and are using this platform for hint evaluation using
LLM. In this example, we use *meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo* as the model, which is available on the
TogetherAI platform. If you wish to use another platform, ensure the model name is valid for that platform.

For local execution, you can set `api_key` to `None`. HintEval supports running large language models (LLMs) locally
via [HuggingFace](https://huggingface.co/models).
:::

:::{Warning}
The output may vary from the example shown below due to the inherent non-deterministic nature of large language models.
:::

### Example
```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.relevance import LlmBased

llm = LlmBased(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key='your_api_key', enable_tqdm=True)
instance_1 = Instance(
    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven.'),
           Hint('This city is the best city for life in 2024.')])
instance_2 = Instance(
    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('He was the first African-American president in U.S. history.'),
           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
instances = [instance_1, instance_2]
results = llm.evaluate(instances)
print(results)
# [[1.00, 0.81], [1.00, 0.95]]
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in
           hint.metrics.items()]
print(metrics)
# ['relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 1.00', 'relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 0.81',
#  'relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 1.00', 'relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 0.95']
```

## Comparison

For each method, we provide details on:

| Method             | Preferred Device | Cost-Effectiveness | Accuracy  | Execution Speed |
|--------------------|------------------|--------------------|-----------|-----------------|
| **Rouge**          | CPU              | High               | Low       | Very Fast       |
| **Non-Contextual** | CPU/GPU          | High               | Moderate  | Fast            |
| **Contextual**     | GPU              | Moderate           | High      | Moderate        |
| **LLM**            | GPU              | Low                | Very High | Slow            |

- **Preferred Device**: Indicates whether the method works best on CPU or GPU.
- **Cost-Effectiveness**: Evaluates how computationally expensive the method is, considering the resources needed.
- **Accuracy**: Reflects how accurate the method is in assessing the metric.
- **Execution Speed**: How quickly the method executes (e.g., Fast, Moderate, Slow).

