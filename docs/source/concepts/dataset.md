(core-concepts-dataset)=

# Dataset

In many research studies, datasets are a crucial component. Researchers often need to create specific datasets for their studies, and typically, they use their own formats, which can make reproducibility challenging for others. This issue is also present in hint tasks. To solve this, **HintEval** provides a robust structure for hint datasets, simplifying dataset sharing and reuse among researchers.

## Dataset Structure
Each dataset in the HintEval framework is an instance of the `Dataset` class. Below is the schema and structure of this class:

<p align="center">
  <img id="thumbnail" src="../_static/imgs/dataset-diagram.png" alt="Dataset Diagram" width="600" height="400" style="cursor: pointer;">
</p>

The `Dataset` structure consists of multiple instances of the `Subset` class. Common subsets include `train`, `validation`, and `test`. Each `Subset` contains several `Instance` objects, which form the core of hint datasets. Each `Instance` includes a `Question` object, `Answer` objects, and `Hint` objects. These objects also have placeholders for storing relevant entities and evaluation scores.

## Data

Let's consider an example with two questions, each accompanied by two hints:

| Example  | Question                                  | Answer         | Hints                                                                                          |
|----------|-------------------------------------------|----------------|------------------------------------------------------------------------------------------------|
| **1**    | What is the capital of Austria?           | Vienna         | 1. This city, once home to Mozart and Beethoven.  <br> 2. This city is the best city for life in 2024. |
| **2**    | Who was the president of the USA in 2009? | Barack Obama   | 1. He was the first African-American president in U.S. history. <br> 2. He was named the 2009 Nobel Peace Prize laureate. |

## Creating the Dataset

Let's walk through how to create this dataset using the `Dataset` class.

### Step 1: Define the Dataset and Subset

First, we define a new `Dataset` called *my_dataset* and add a subset called *entire*.

```python
from hinteval import Dataset
from hinteval.cores import Subset

dataset = Dataset(name='my_dataset', url=None, version='1.0.0', description='This is my first hint dataset.')
subset = Subset('entire')
dataset.add_subset(subset)
```

### Step 2: Define Questions, Answers, and Hints

Convert the examples into the corresponding `Question`, `Answer`, and `Hint` objects.

```python
from hinteval.cores import Question, Answer, Hint

question_1 = Question(question='What is the capital of Austria?')
answer_1 = Answer(answer='Vienna')
hint_11 = Hint(hint='This city, once home to Mozart and Beethoven.')
hint_12 = Hint(hint='This city is the best city for life in 2024.')

question_2 = Question(question='Who was the president of USA in 2009?')
answer_2 = Answer(answer='Barack Obama')
hint_21 = Hint(hint='He was the first African-American president in U.S. history.')
hint_22 = Hint(hint='He was named the 2009 Nobel Peace Prize laureate.')
```

### Step 3: Create Instances

Next, we create two `Instance` objects that will form the core of our dataset and subset.

```python
from hinteval.cores import Instance

instance_1 = Instance(question=question_1, answers=[answer_1], hints=[hint_11, hint_12])
instance_2 = Instance(question=question_2, answers=[answer_2], hints=[hint_21, hint_22])
```

### Step 4: Add Instances to Subset

Finally, add the two instances to the *entire* subset.

```python
subset.add_instance(instance_1, q_id='id_1')
subset.add_instance(instance_2, q_id='id_2')
```

## Viewing the Dataset

Now, let's take a look at the structured dataset:

```python
print(dataset)
```
:::{dropdown} Output
```json
{
    "name": "my_dataset",
    "version": "1.0.0",
    "description": "This is my first hint dataset.",
    "url": null,
    "metadata": {},
    "subsets": {
        "entire": {
            "name": "entire",
            "metadata": {},
            "instances": {
                "id_1": {
                    "question": {
                        "question": "What is the capital of Austria?",
                        "question_type": {},
                        "entities": [],
                        "metrics": {},
                        "metadata": {}
                    },
                    "answers": [
                        {
                            "answer": "Vienna",
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        }
                    ],
                    "hints": [
                        {
                            "hint": "This city, once home to Mozart and Beethoven.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "This city is the best city for life in 2024.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        }
                    ],
                    "metadata": {}
                },
                "id_2": {
                    "question": {
                        "question": "Who was the president of USA in 2009?",
                        "question_type": {},
                        "entities": [],
                        "metrics": {},
                        "metadata": {}
                    },
                    "answers": [
                        {
                            "answer": "Barack Obama",
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        }
                    ],
                    "hints": [
                        {
                            "hint": "He was the first African-American president in U.S. history.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "He was named the 2009 Nobel Peace Prize laureate.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        }
                    ],
                    "metadata": {}
                }
            }
        }
    }
}
```
:::

## Preprocessed Datasets

As of October 2024, there are only a limited number of datasets for hint tasks that have been added to the HintEval framework. These datasets have been fully preprocessed and evaluated using all available methods, making them ready for immediate use. You can view the available datasets using the [available_datasets](../references/dataset.rst#hinteval.cores.dataset.dataset.Dataset.available_datasets) function and download and load them with the [download_and_load_dataset](../references/dataset.rst#hinteval.cores.dataset.dataset.Dataset.download_and_load_dataset) function.

```python
from hinteval import Dataset

Dataset.available_datasets(show_info=True, update=True)
Dataset.download_and_load_dataset('triviahg')
```

## Additional Attributes

### Metadata
Each object in the HintEval framework, such as `Entity`, `Metric`, `Answer`, `Hint`, `Question`, `Instance`, `Subset`, and `Dataset`, comes with a `metadata` attribute. This allows you to store additional, custom information about any object. This feature is particularly useful for attaching metadata like difficulty level, category, or any other context-specific details to your dataset components.

For example, let's add a *difficulty* label to a question:

```python
from hinteval.cores import Question

question = Question('In which year did Iran nationalize its oil industry under Prime Minister Mohammad Mossadegh?')
question.metadata['difficulty'] = 'Hard'

print(question)
```

:::{dropdown} Output
```json
{
    "question": "In which year did Iran nationalize its oil industry under Prime Minister Mohammad Mossadegh?",
    "question_type": {},
    "entities": [],
    "metrics": {},
    "metadata": {
        "difficulty": "Hard"
    }
}
```
:::

### Import/Export

In addition to `metadata`, each object also provides two key functions: `from_dict` and `to_dict`. These functions allow you to convert objects into dictionaries and vice versa, making it easier to save or load data as JSON or other formats.

Here’s how you can use the `from_dict` function to create a `Question` object from a dictionary:

```python
from hinteval.cores import Question

question_dict = {
    "question": "In which year did Iran nationalize its oil industry under Prime Minister Mohammad Mossadegh?",
    "question_type": {},
    "entities": [],
    "metrics": {},
    "metadata": {
        "difficulty": "Hard"
    }
}

question = Question.from_dict(question_dict)

print(question.metadata)
```

:::{dropdown} Output
```json
{
  'difficulty': 'Hard'
}
```
:::

These flexible attributes and functions help you organize and manage your dataset more effectively, ensuring you can add custom information and easily serialize or deserialize your data.

<div id="lightbox" class="lightbox" style="display:none; position:fixed; z-index:1000; left:0; top:0; width:100%; height:100%; background-color:rgba(0, 0, 0, 0.8); text-align: center;">
  <span class="lightbox-close" style="position:absolute; top:20px; right:30px; color:white; font-size:30px; font-weight:bold; cursor:pointer;">&times;</span>
  <img src="../_static/imgs/dataset-diagram.png" alt="Dataset Diagram" style="display: inline-block; margin-top: 5%; max-width:90%; max-height:90%;">
</div>

<script>
  const lightbox = document.getElementById('lightbox');
  const thumbnail = document.getElementById('thumbnail');
  const closeBtn = document.querySelector('.lightbox-close');

  thumbnail.onclick = function() {
    lightbox.style.display = 'block';
  }

  closeBtn.onclick = function() {
    lightbox.style.display = 'none';
  }

  lightbox.onclick = function(event) {
    if (event.target == lightbox) {
      lightbox.style.display = 'none';
    }
  }
</script>