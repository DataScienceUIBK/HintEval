(core-concepts-metric-convergence)=

# Convergence

The **Convergence** metric evaluates how well the hints narrow down potential answers to the question. In other words, it assesses how effectively the hints guide the user toward eliminating incorrect answers and focusing on the correct one.

The HintEval framework provides three different methods for computing the Convergence metric.

- [**Specificity**](#specificity)
- [**Neural-Network**](#neural-network)
- [**LLM**](#llm)

:::{note}
The `evaluate` function takes a list of `Instance` objects as its input, where each instance contains a question and its associated hints and answers.
:::

## Specificity

The [Specificity](../../references/metrics/convergence.rst#hinteval.cores.evaluation_metrics.convergence.Specificity) method measures how well a hint reduces the space of possible answers by focusing on specific details. For more information, refer to the [📝original paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531734).

Specificity methods available:

- **BERT-base**: A widely-used model that captures contextual relationships between words, helping to assess how specific a hint is.
- **RoBERTa-large**: A more advanced variant of BERT, trained on a larger corpus, offering better accuracy for measuring specificity.

### Example

```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.convergence import Specificity

specificity = Specificity(model_name='bert-base')
instance_1 = Instance(
    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')])
instance_2 = Instance(

    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('He was the first African-American president in U.S. history.')])
instances = [instance_1, instance_2]
results = specificity.evaluate(instances)
print(results)
# [[1], [1]]
classes = [sent.hints[0].metrics['convergence-specificity-bert-base'].metadata['description'] for sent in
           instances]
print(classes)
# ['specific', 'specific']
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in

           hint.metrics.items()]
print(metrics)
# ['convergence-specificity-bert-base: 1', 'convergence-specificity-bert-base: 1']
```

## Neural-Network

The [Neural-Network](../../references/metrics/convergence.rst#hinteval.cores.evaluation_metrics.convergence.NeuralNetworkBased) method uses pre-trained transformer models to evaluate how effectively the hints converge on the correct answer. These models understand deeper context and relationships between words, allowing them to provide a more nuanced assessment of convergence.

Neural-Network methods available:

- **BERT-base**: A model that can capture contextual relationships between words and is trained on a large corpus of English text.
- **RoBERTa-large**: A more robust variant of BERT, trained on more data, providing better accuracy for convergence predictions.

### Example

```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.convergence import NeuralNetworkBased

neural_network = NeuralNetworkBased(model_name='bert-base')
instance_1 = Instance(

    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')])
instance_2 = Instance(
    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('He was named the 2009 Nobel Peace Prize laureate')])
instances = [instance_1, instance_2]
results = neural_network.evaluate(instances)
print(results)
# [[1.0], [1.0]]
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in
           hint.metrics.items()]
print(metrics)
# ['convergence-nn-bert-base: 1.0', 'convergence-nn-bert-base: 1.0']
```

## LLM

The [LLM](../../references/metrics/convergence.rst#hinteval.cores.evaluation_metrics.convergence.LlmBased) method leverages large language models to assess how well hints converge on the correct answer. By using models like LLaMA, this approach provides highly accurate and context-aware convergence scores, allowing for a deeper understanding of how effectively a hint narrows down potential answers. For more information, refer to the [📝original paper](https://dl.acm.org/doi/10.1145/3626772.3657855).

LLM methods available:

- **LLaMA-3-8B**: A large language model designed for generating more specific, detailed responses.
- **LLaMA-3-70B**: A more powerful variant of LLaMA, trained on a larger corpus, providing highly accurate convergence scores.

:::{note}
We assume you have an active API key for the TogetherAI platform and are using this platform for convergence evaluation using LLM. In this example, we use *meta-llama/Meta-Llama-3-8B-Instruct-Turbo* as the model, which is available on the TogetherAI platform. 

For local execution, you can set `api_key` to `None`. HintEval supports running large language models (LLMs) locally via [HuggingFace](https://huggingface.co/models).
:::

:::{Warning}
- Please note that this method only supports TogetherAI platform for using remotely.

- The output may vary from the example shown below due to the inherent non-deterministic nature of large language models.
:::

### Example

```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.convergence import LlmBased

llm = LlmBased(model_name='llama-3-8b', together_ai_api_key='your_api_key')
instance_1 = Instance(
    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')])
instance_2 = Instance(
    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('He was the first African-American president in U.S. history.')])
instances = [instance_1, instance_2]
results = llm.evaluate(instances)
print(results)
# [[0.91], [1.0]]
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in
           hint.metrics.items()]
print(metrics)
# ['convergence-llm-llama-3-8b: 0.91', 'convergence-llm-llama-3-8b: 1.0']
scores = [hint.metrics['convergence-llm-llama-3-8b'].metadata['scores'] for inst in instances for hint in
          inst.hints]
print(scores[0])
# {'Salzburg': 1, 'Graz': 0, 'Innsbruck': 0, 'Linz': 0, 'Klagenfurt': 0, 'Bregenz': 0, 'Wels': 0, 'St. Pölten': 0, 'Eisenstadt': 0, 'Sankt Johann impong': 0, 'Vienna': 1}
print(scores[1])
# {'George W. Bush': 0, 'Bill Clinton': 0, 'Jimmy Carter': 0, 'Donald Trump': 0, 'Joe Biden': 0, 'Ronald Reagan': 0, 'Richard Nixon': 0, 'Gerald Ford': 0, 'Franklin D. Roosevelt': 0, 'Theodore Roosevelt': 0, 'Barack Obama': 1}
```

## Comparison

For each method, we provide details on:

| Method               | Preferred Device | Cost-Effectiveness | Accuracy  | Execution Speed |
|----------------------|------------------|--------------------|-----------|-----------------|
| **Specificity**       | GPU              | Moderate           | Low       | Moderate        |
| **Neural-Network**    | GPU              | Moderate           | Moderate  | Moderate        |
| **LLM**              | GPU              | Low                | High | Slow            |

- **Preferred Device**: Indicates whether the method works best on CPU or GPU.
- **Cost-Effectiveness**: Evaluates how computationally expensive the method is, considering the resources needed.
- **Accuracy**: Reflects how accurate the method is in assessing the convergence metric.
- **Execution Speed**: How quickly the method executes (e.g., Fast, Moderate, Slow).