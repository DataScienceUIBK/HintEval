(core-concepts-model)=

# Model

The Hint Generation task is a novel research area introduced
by [Jatowt et al. 2024](https://dl.acm.org/doi/abs/10.1145/3578337.3605119), with the first released dataset published
in 2024 by [Mozafari et al. 2024](https://dl.acm.org/doi/abs/10.1145/3626772.3657855). This emerging task brings forth
numerous open questions and research challenges. As with any new area of study, datasets are essential to explore
different aspects of the problem.

Until October 2024, HintEval includes only a limited number of datasets for hint tasks, highlighting the need for more
datasets in this field. To address this, the HintEval framework introduces the `Model` class, which allows users to
generate synthetic hint datasets.

There are two primary approaches for generating hints: [Answer Aware](../references/model.rst#hinteval.cores.model.model.AnswerAware) and [Answer Agnostic](../references/model.rst#hinteval.cores.model.model.AnswerAgnostic).

:::{dropdown} Hint Generation Approaches

- **Answer Aware**: This approach assumes there is at least one answer for each question in the dataset.
- **Answer Agnostic**: This approach assumes there are no answers provided for the questions in the dataset.
  :::

Fortunately, the HintEval framework provides dedicated classes for both approaches, enabling users to generate hints
based on their preferred method.

:::{note}
We assume you have an active API key for the TogetherAI platform and are using this platform for hint generation. In
this example, we use *meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo* as the model, which is available on the TogetherAI
platform. If you wish to use another platform, ensure the model name is valid for that platform.

For local execution, you can set `api_key` to `None`. HintEval supports running large language models (LLMs) locally
via [HuggingFace](https://huggingface.co/models).
:::

## Answer Aware Approach

Let’s consider two example questions:

| Example | Question                                  | Answer       |
|---------|-------------------------------------------|--------------|
| **1**   | What is the capital of Austria?           | Vienna       |
| **2**   | Who was the president of the USA in 2009? | Barack Obama |

To convert these examples into a hint dataset, follow the steps outlined in [Dataset](core-concepts-dataset).

```python
from hinteval import Dataset
from hinteval.cores import Subset, Instance, Question, Answer

dataset = Dataset(name='my_answer_aware_dataset', url=None, version='1.0.0',
                  description='This is my first answer-aware hint dataset.')
subset = Subset('entire')
dataset.add_subset(subset)

question_1 = Question(question='What is the capital of Austria?')
answer_1 = Answer(answer='Vienna')

question_2 = Question(question='Who was the president of USA in 2009?')
answer_2 = Answer(answer='Barack Obama')

instance_1 = Instance(question=question_1, answers=[answer_1], hints=[])
instance_2 = Instance(question=question_2, answers=[answer_2], hints=[])

subset.add_instance(instance_1, q_id='id_1')
subset.add_instance(instance_2, q_id='id_2')
```

Next, we generate hints for these questions using the `AnswerAware` class.

```python
from hinteval.model import AnswerAware

api_key = 'your_api_key'
base_url = 'https://api.together.xyz/v1'

answer_aware = AnswerAware('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key=api_key, base_url=base_url,
                           num_of_hints=4, temperature=0.3, max_tokens=1024, batch_size=1, enable_tqdm=True)
```

To generate hints, call the `generate` function:

```python
answer_aware.generate([instance_1, instance_2])
```

Now, let’s view the generated hints:

```python
print(dataset)
```

:::{dropdown} Output

```json
{
  "name": "my_answer_aware_dataset",
  "version": "1.0.0",
  "description": "This is my first answer-aware hint dataset.",
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
              "hint": "The city is located on the Danube River and is known for its grand palaces and opera houses.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "This city is famous for its coffee culture and is often referred to as the \"City of Music\".",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "The city is home to numerous museums, including one dedicated to a famous composer who lived there.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "The city is situated in the eastern part of the country and is a major cultural and economic hub.",
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
              "hint": "The answer is the first African American to hold the office of President in the United States.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "This president was a member of the Democratic Party and served two terms in office.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "He was a senator from Illinois before being elected as President in 2008.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "He was awarded the Nobel Peace Prize in 2009, just months after taking office.",
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

## Answer Agnostic Approach

Let’s now explore the **Answer Agnostic** approach with two questions:

| Example | Question                                  |
|---------|-------------------------------------------|
| **1**   | What is the capital of Austria?           |
| **2**   | Who was the president of the USA in 2009? |

Follow the same steps to create a dataset, as described in [Dataset](core-concepts-dataset).

```python
from hinteval import Dataset
from hinteval.cores import Subset, Instance, Question

dataset = Dataset(name='my_answer_agnostic_dataset', url=None, version='1.0.0',
                  description='This is my first answer-agnostic hint dataset.')
subset = Subset('entire')
dataset.add_subset(subset)

question_1 = Question(question='What is the capital of Austria?')
question_2 = Question(question='Who was the president of USA in 2009?')

instance_1 = Instance(question=question_1, answers=[], hints=[])
instance_2 = Instance(question=question_2, answers=[], hints=[])

subset.add_instance(instance_1, q_id='id_1')
subset.add_instance(instance_2, q_id='id_2')
```

To generate hints using the `AnswerAgnostic` class, follow these steps:

```python
from hinteval.model import AnswerAgnostic

api_key = 'your_api_key'
base_url = 'https://api.together.xyz/v1'

answer_agnostic = AnswerAgnostic('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key=api_key, base_url=base_url,
                                 num_of_hints=4, temperature=0.3, max_tokens=1024, batch_size=1, enable_tqdm=True)
```

Then, generate the hints:

```python
answer_agnostic.generate([instance_1, instance_2])
```

Let’s view the hints generated for these questions:

```python
print(dataset)
```

:::{dropdown} Output

```json
{
  "name": "my_answer_agnostic_dataset",
  "version": "1.0.0",
  "description": "This is my first answer-agnostic hint dataset.",
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
          "answers": [],
          "hints": [
            {
              "hint": "The capital city is located near the Danube River, which is the second-longest river in Europe.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "This city is known for its rich cultural heritage and is home to numerous museums, opera houses, and historical landmarks.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "The city is situated in the eastern part of the country and is close to the borders of several neighboring countries.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "The city's name starts with the letter \"V\" and is often associated with famous composers such as Mozart and Strauss.",
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
          "answers": [],
          "hints": [
            {
              "hint": "This president was the first African American to hold the office in U.S. history.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "He was a member of the Democratic Party and served two terms in the presidency.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "Before becoming president, he served as a United States Senator from a state in the Midwest.",
              "source": null,
              "entities": [],
              "metrics": {},
              "metadata": {}
            },
            {
              "hint": "He was awarded the Nobel Peace Prize in 2009, just a few months after taking office.",
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

## Postprocessing LLM Outputs

Each large language model (LLM) may produce outputs in its own specific format. When HintEval requests hints from an LLM, the output may follow a template unique to that model. To handle this, HintEval provides a parameter called `parse_llm_response` for both the `AnswerAware` and `AnswerAgnostic` classes. This parameter allows you to postprocess the raw output of the LLM and convert it into a list of hints.

By default, HintEval includes a parsing function that works well with **Meta LLaMA** models. However, you can customize this function to suit other models. Below is the schema for this parsing function:

```python
def my_parse_llm_response(llm_output: str) -> list[str]:
    hints_output: list[str] = []
    
    # Your custom logic to process the LLM's output and extract the hints
    # For example, you might need to split the output into individual hints or clean up the text.
    
    return hints_output
```

This function allows you to tailor the postprocessing step for any LLM you are working with, ensuring that the output format is consistent with your hint generation needs.

Here’s an example of using a custom function to postprocess the LLM output:

```python
def my_parse_llm_response(llm_output: str) -> list[str]:
    hints_output: list[str] = []

    for sentence in llm_output.split('\n'):
        hints_output.append(sentence)

    return hints_output


answer_aware = AnswerAware('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key=api_key, base_url=base_url,
                           parse_llm_response=my_parse_llm_response, num_of_hints=4, temperature=0.3, max_tokens=1024,
                           batch_size=1, enable_tqdm=True)

answer_aware.generate([instance_1, instance_2])

print(dataset)
```

:::{dropdown} Output

```json
{
    "name": "my_answer_aware_dataset",
    "version": "1.0.0",
    "description": "This is my first answer-aware hint dataset.",
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
                            "hint": "Here are 4 hints for the question:",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "1. The capital is located near the Danube River and is known for its grand palaces and opera houses.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "2. This city is home to the famous St. Stephen's Cathedral and is a popular destination for classical music lovers.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "3. The capital is situated in the eastern part of the country and has a rich history dating back to the Roman Empire.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "4. This city is famous for its coffee culture and is often referred to as the \"City of Dreams\" due to its association with famous composers such as Mozart and Strauss.",
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
                            "hint": "Here are 4 hints for the question:",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "1. The president in question was the first African American to hold the office.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "2. He was a member of the Democratic Party and served two terms from 2009 to 2017.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "3. Before becoming president, he served as a United States Senator from a state in the Midwest.",
                            "source": null,
                            "entities": [],
                            "metrics": {},
                            "metadata": {}
                        },
                        {
                            "hint": "4. He was awarded the Nobel Peace Prize in 2009, just a few months after taking office.",
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

## Common Attributes

Generating synthetic datasets using large language models (LLMs) can be computationally intensive and time-consuming. Additionally, errors such as GPU memory overflow or connection failures in API calls can interrupt the process, making it impractical to start over from scratch. To mitigate these issues, the HintEval framework provides two key features: **checkpoints** and a **memory_release** function.

### Checkpoints

To ensure that progress is not lost during hint generation, you can enable the **checkpoint** feature. This saves the current state at regular intervals, so in case of errors, you can resume from the last checkpoint instead of restarting the entire process. You can control how frequently checkpoints are saved using the `checkpoint_step` parameter, which defines the number of steps between each save.

For example, in the following code, we generate 3 hints for each question using the `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` model with a batch size of 5. We enable the checkpoint feature and set it to save progress every 10 steps, which corresponds to saving after generating 150 hints (3 hints per question * 5 questions per batch * 10 steps):

:::{warning}
To enable and use checkpointing, customize the directory where checkpoints are stored. For more information on customization, refer to [Customizations](../howtos/customizations/environment_variables.md#checkpoint-storage).
:::

```python
from hinteval.model import AnswerAgnostic

api_key = 'your_api_key'
base_url = 'https://api.together.xyz/v1'

answer_agnostic = AnswerAgnostic('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key=api_key, base_url=base_url,
                                 checkpoint=True, num_of_hints=3, batch_size=5, checkpoint_step=10)
```

### Releasing Memory

Another common issue when working with LLMs, especially on local systems, is GPU memory overflow. To avoid crashes due to memory limits, the HintEval framework provides a **release_memory** function. After generating hints, you can call this function to free up the GPU resources. This is especially useful when working on multiple large-scale tasks or running multiple models in sequence.

Here’s how you can release memory after generating hints:

```python
answer_agnostic.release_memory()
```

### Progress Tracking

For monitoring the progress of hint generation, HintEval supports the use of [tqdm](https://github.com/tqdm/tqdm), a Python library for creating progress bars. By enabling the `enable_tqdm` parameter, you can visualize the progress of the hint generation process in real-time.

Here’s an example that combines checkpointing, memory release, and progress tracking:

```python
from hinteval import Dataset
from hinteval.cores import Subset, Instance, Question
from hinteval.model import AnswerAgnostic

dataset = Dataset(name='my_answer_agnostic_dataset', url=None, version='1.0.0',
                  description='This is my first answer-agnostic hint dataset.')
subset = Subset('entire')
dataset.add_subset(subset)

question_1 = Question(question='What is the capital of Austria?')
question_2 = Question(question='Who was the president of USA in 2009?')

instance_1 = Instance(question=question_1, answers=[], hints=[])
instance_2 = Instance(question=question_2, answers=[], hints=[])

subset.add_instance(instance_1, q_id='id_1')
subset.add_instance(instance_2, q_id='id_2')

answer_agnostic = AnswerAgnostic('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key=api_key, base_url=base_url,
                                 checkpoint=True, num_of_hints=3, checkpoint_step=1, enable_tqdm=True)

answer_agnostic.generate([instance_1, instance_2])

answer_agnostic.release_memory()
```

Example progress output:
```
Generating hints using meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo: 100%|██████████| 2/2 [00:11<00:00,  5.88s/it]
```

With these features, you can effectively manage the computational demands of LLMs, ensuring progress is saved, memory usage is optimized, and generation progress is tracked in real-time.
