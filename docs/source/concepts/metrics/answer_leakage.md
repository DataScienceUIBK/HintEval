(core-concepts-metric-answerleakage)=

# AnswerLeakage

The **AnswerLeakage** metric measures how much a hint reveals the correct answer. It ensures that hints guide the user without directly revealing the solution, maintaining the educational value of the hint. This metric is essential for evaluating whether hints are subtle enough to encourage problem-solving rather than giving away the answer.

The HintEval framework provides two different methods for computing the AnswerLeakage metric:

- [**Lexical**](#lexical)
- [**Contextual**](#contextual)

:::{note}
The `evaluate` function takes a list of `Instance` objects as its input, where each instance contains a question, its associated hints, and the correct answers.
:::

## Lexical

The [Lexical](../../references/metrics/answer_leakage.rst#hinteval.cores.evaluation_metrics.answer_leakage.Lexical) method assesses the similarity between the hint and the answer at the word level, without considering deeper contextual meaning. This method checks for explicit word overlap, making it useful for detecting hints that directly repeat or use the same words as the answer. For more information, refer to the [📝original paper](https://dl.acm.org/doi/10.1145/3626772.3657855).

Lexical methods are available in two main variants:

- **With Stop-Words**: Considers common stop-words when evaluating overlap, making the method more permissive.
- **Without Stop-Words**: Ignores stop-words, making the method more precise in identifying relevant word overlap.

### Example

```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.answer_leakage import Lexical

lexical = Lexical(method='include_stop_words')
instance_1 = Instance(
    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven.'),
           Hint('This city is called as Vienna.')])
instance_2 = Instance(
    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('His lastname is Obama.'),
           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
instances = [instance_1, instance_2]
results = lexical.evaluate(instances)
print(results)
# [[0, 1], [1, 0]]
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in
           hint.metrics.items()]
print(metrics)
# ['answer-leakage-lexical-include_stop_words-sm: 0', 'answer-leakage-lexical-include_stop_words-sm: 1',
#  'answer-leakage-lexical-include_stop_words-sm: 1', 'answer-leakage-lexical-include_stop_words-sm: 0']
```

## Contextual

The [Contextual](../../references/metrics/answer_leakage.rst#hinteval.cores.evaluation_metrics.answer_leakage.ContextualEmbeddings) method uses embeddings to evaluate whether the hint is semantically similar to the answer. By leveraging advanced embeddings, this method captures nuanced similarities, even when different words are used to convey the same idea. It detects subtle answer leakage that might not be evident through word overlap.

:::{note}
This method supports [SentenceBert](https://sbert.net/) models to compute contextualized word embeddings.
:::
### Example

```python
from hinteval.cores import Instance, Question, Hint, Answer
from hinteval.evaluation.answer_leakage import ContextualEmbeddings

contextual = ContextualEmbeddings(sbert_model='paraphrase-multilingual-mpnet-base-v2')
instance_1 = Instance(

    question=Question('What is the capital of Austria?'),
    answers=[Answer('Vienna')],
    hints=[Hint('This city, once home to Mozart and Beethoven.'),
           Hint('This city is called as Vienna.')])
instance_2 = Instance(
    question=Question('Who was the president of USA in 2009?'),
    answers=[Answer('Barack Obama')],
    hints=[Hint('His lastname is Obama.'),
           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
instances = [instance_1, instance_2]
results = contextual.evaluate(instances)
print(results)
# [[0.495, 1.0], [0.967, 0.332]]
metrics = [f'{metric_key}: {metric_value.value}' for
           instance in instances
           for hint in instance.hints for metric_key, metric_value in
           hint.metrics.items()]
print(metrics)
# ['answer-leakage-lexical-include_stop_words-sm: 0.495', 'answer-leakage-lexical-include_stop_words-sm: 1.0',
#  'answer-leakage-lexical-include_stop_words-sm: 0.967', 'answer-leakage-lexical-include_stop_words-sm: 0.332']
```

## Comparison

For each method, we provide details on:

| Method             | Preferred Device | Cost-Effectiveness | Accuracy  | Execution Speed |
|--------------------|------------------|--------------------|-----------|-----------------|
| **Lexical**        | CPU              | Very High          | Low       | Very Fast       |
| **Contextual**     | GPU              | Moderate           | High      | Moderate        |

- **Preferred Device**: Indicates whether the method works best on CPU or GPU.
- **Cost-Effectiveness**: Evaluates how computationally expensive the method is, considering the resources needed.
- **Accuracy**: Reflects how accurate the method is in assessing the metric.
- **Execution Speed**: How quickly the method executes (e.g., Fast, Moderate, Slow).