(get-started-dataset-generation)=
# Generate a Synthetic Hint Dataset

This tutorial provides step-by-step guidance on how to generate a synthetic hint dataset using large language models via the TogetherAI platform. To proceed, make sure you have an active API key for TogetherAI.

```python
api_key = "your-api-key"
base_url = "https://api.together.xyz/v1"
```

## Question/Answer Pairs

First, we'll gather a collection of question/answer pairs to use as a foundation for generating `Question/Answer/Hint` triples. We'll load 10 questions from the [WebQuestions](https://aclanthology.org/D13-1160.pdf) dataset using the [🤗datasets](https://pypi.org/project/datasets/) library.

```python
from datasets import load_dataset

webq = load_dataset("Stanford/web_questions", split='test')
question_answers = webq.select_columns(['question', 'answers'])[10:20]
qa_pairs = zip(question_answers['question'], question_answers['answers'])
```

At this point, we have a set of question/answer pairs ready for creating synthetic `Question/Answer/Hint` instances.

## Dataset Creation

Next, we'll use HintEval's `Dataset` class to create a new dataset called `synthetic_hint_dataset`, which will include 10 question/answer pairs within a subset named `entire`.

```python
from hinteval import Dataset
from hinteval.cores import Subset, Instance

dataset = Dataset('synthetic_hint_dataset')
subset = Subset('entire')

for q_id, (question, answers) in enumerate(qa_pairs, 1):
    instance = Instance.from_strings(question, answers, [])
    subset.add_instance(instance, f'id_{q_id}')

dataset.add_subset(subset)
dataset.prepare_dataset(fill_question_types=True)
```

## Hint Generation

Now, we can generate 5 hints for each question using HintEval's `AnswerAware` model. For this example, we will use the [Meta LLaMA-3.1-70b-Instruct-Turbo](https://www.llama.com/) model from TogetherAI.

```python
from hinteval.model import AnswerAware

generator = AnswerAware('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 
                        api_key, base_url, num_of_hints=5, enable_tqdm=True)
generator.generate(dataset['entire'].get_instances())
```

:::{note}
Depending on the LLM provider you are using, you may need to configure the `model` and other parameters in the `AnswerAware` function. See the [Model](../concepts/model.md) and [References](../references/model.rst) for more information.
:::

## Exporting the Dataset

Once the hints are generated, we can export the synthetic hint dataset to a pickle file.

```python
dataset.store('./synthetic_hint_dataset.pickle')
```

## Viewing the Hints

Finally, let's view the hints generated for the third question in the dataset.

```python
dataset = Dataset.load('./synthetic_hint_dataset.pickle')

third_question = dataset['entire'].get_instance('id_3')
print(f'Question: {third_question.question.question}')
print(f'Answer: {third_question.answers[0].answer}')
print()
for idx, hint in enumerate(third_question.hints, 1):
    print(f'Hint {idx}: {hint.hint}')
```

Example output:

```
Question: who is governor of ohio 2011?
Answer: John Kasich

Hint 1: The answer is a Republican politician who served as the 69th governor of the state.
Hint 2: This person was a member of the U.S. House of Representatives for 18 years before becoming governor.
Hint 3: The governor was known for his conservative views and efforts to reduce government spending.
Hint 4: During their term, they implemented several reforms related to education, healthcare, and the economy.
Hint 5: This governor served two consecutive terms, from 2011 to 2019, and ran for the U.S. presidency in 2016.
```

Because of the generative nature of large language models, the output hints may vary.