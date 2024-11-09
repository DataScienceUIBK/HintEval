(core-concepts-metric-familiarity)=

# Familiarity

The **Familiarity** metric measures how common or well-known the information in the hints, questions, or answers is. It assesses whether the content is likely to be understood by the general audience, making it easier for users to grasp the provided hints without needing specialized knowledge.

The HintEval framework provides two different methods for computing the Familiarity metric.

- [**Word-Frequency**](#word-frequency)
- [**Wikipedia**](#wikipedia)

:::{note}
The `evaluate` function takes a list of `Hint`, `Question`, or `Answer` objects as its input, where each object contains the text that needs to be evaluated for familiarity.
:::

## Word-Frequency

The [Word-Frequency](../../references/metrics/familiarity.rst#hinteval.cores.evaluation_metrics.familiarity.WordFrequency) method evaluates the familiarity of the text by analyzing how frequently the words appear in large corpora. The [C4 corpus](https://dl.acm.org/doi/abs/10.5555/3455716.3455856) is used as a reference dataset for word frequencies, providing insight into how commonly words are used in everyday language.


Word-Frequency is available in two main variants:

- **With Stop-Words**: Considers all words in the text, including common stop-words like "the," "is," and "and."
- **Without Stop-Words**: Excludes stop-words to focus on the more meaningful terms that are likely to indicate familiarity.

### Example

```python
from hinteval.cores import Question, Hint
from hinteval.evaluation.familiarity import WordFrequency

word_frequency = WordFrequency(method='include_stop_words')
sentence_1 = Question('What is the capital of Austria?')
sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
sentences = [sentence_1, sentence_2]
results = word_frequency.evaluate(sentences)
print(results)
# [1.0, 1.0]
metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
           sent.metrics.items()]
print(metrics)
# ['familiarity-freq-include_stop_words-sm: 1.0', 'familiarity-freq-include_stop_words-sm: 1.0']
```

## Wikipedia

The [Wikipedia](../../references/metrics/familiarity.rst#hinteval.cores.evaluation_metrics.familiarity.Wikipedia) method evaluates familiarity by analyzing the popularity of the entities mentioned in the text. It does this by looking up the corresponding Wikipedia pages for each entity and using the number of views of the page as a measure of familiarity. This method helps determine how well-known the people, places, or concepts in the text are to the general public. For more information, refer to the [📝original paper](https://dl.acm.org/doi/10.1145/3626772.3657855).

### Example

```python
from hinteval.cores import Question, Hint
from hinteval.evaluation.familiarity import Wikipedia

wikipedia = Wikipedia(spacy_pipeline='en_core_web_trf')
sentence_1 = Question('What is the capital of Austria?')
sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
sentences = [sentence_1, sentence_2]
results = wikipedia.evaluate(sentences)
print(results)
# [1.0, 1.0]
metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
           sent.metrics.items()]
print(metrics)
# ['familiarity-wikipedia-trf: 1.0', 'familiarity-wikipedia-trf: 1.0']
entities = [f'{entity.entity}: {entity.metadata["wiki_views_per_month"]}' for sent in sentences for entity in
            sent.entities]
print(entities)
# ['austria: 248144', 'mozart: 233219', 'beethoven: 224128', 'austria: 248144']
```

## Comparison

For each method, we provide details on:

| Method               | Preferred Device | Cost-Effectiveness | Accuracy  | Execution Speed |
|----------------------|------------------|--------------------|-----------|-----------------|
| **Word-Frequency**    | CPU              | Very High          | Low       | Very Fast       |
| **Wikipedia**         | CPU              | High               | High      | Slow            |

- **Preferred Device**: Indicates whether the method works best on CPU or GPU.
- **Cost-Effectiveness**: Evaluates how computationally expensive the method is, considering the resources needed.
- **Accuracy**: Reflects how accurate the method is in assessing familiarity.
- **Execution Speed**: How quickly the method executes (e.g., Fast, Moderate, Slow).