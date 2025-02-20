<p align="center">
  <img src="./docs/source/_static/imgs/logo-new-background.png" width="200" />
</p>

<p align="center">
  <a href="http://hinteval.readthedocs.io/"><img src="https://img.shields.io/static/v1?label=Models&message=HuggingFace&color=yellow&logo=huggingface"></a>
  <a href="http://hinteval.readthedocs.io/"><img src="https://img.shields.io/static/v1?label=Documentation&message=HintEval&color=orange&logo=Read the Docs"></a>
  <a href="https://doi.org/10.48550/arXiv.2502.00857"><img src="https://img.shields.io/static/v1?label=Paper&message=ArXiv&color=green&logo=arXiv"></a>
  <a href="https://opensource.org/license/apache-2-0"><img src="https://img.shields.io/static/v1?label=License&message=Apache-2.0&color=red"></a>
  <a href="https://pepy.tech/projects/hinteval"><img src="https://static.pepy.tech/badge/hinteval" alt="PyPI Downloads"></a>
  <a href="https://github.com/DataScienceUIBK/HintEval/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/DataScienceUIBK/HintEval.svg?label=Version&color=orange"></a>
</p>

**HintEval💡** is a powerful framework designed for both generating and evaluating hints for input questions. These hints serve as subtle clues, guiding users toward the correct answer without directly revealing it. As the first tool of its kind, HintEval allows users to create and assess hints from various perspectives. 

<p align="center">
<img src="./docs/source/_static/imgs/Framework.png">
</p>

## ✨ Features
 - **Unified Framework**: HintEval combines datasets, models, and evaluation metrics into a single Python-based library. This integration allows researchers to seamlessly conduct hint generation and evaluation tasks.
 - **Comprehensive Metrics**: Implements *five* core metrics (*fifteen* evaluation methods)—*Relevance*, *Readability*, *Convergence*, *Familiarity*, and *Answer Leakage*—with lightweight to resource-intensive methods to cater to diverse research needs.
 - **Dataset Support**: Provides access to multiple preprocessed and evaluated datasets, including [*TriviaHG*](https://github.com/DataScienceUIBK/TriviaHG), [*WikiHint*](https://github.com/DataScienceUIBK/WikiHint), [*HintQA*](https://github.com/DataScienceUIBK/HintQA), and [*KG-Hint*](https://github.com/AlexWalcher/automaticHintGeneration), supporting both *answer-aware* and *answer-agnostic* hint generation approaches.
 - **Customizability**: Allows users to define their own datasets, models, and evaluation methods with minimal effort using a structured design based on Python classes.
 - **Extensive Documentation**: Accompanied by detailed [📖online documentation](https://hinteval.readthedocs.io/) and tutorials for easy adoption.

## 🔎 Roadmap
 - **Enhanced Datasets**: Expand the repository with additional datasets to support diverse hint-related tasks.
 - **Advanced Evaluation Metrics**: Introduce new evaluation techniques such as Unieval and cross-lingual compatibility.
 - **Broader Compatibility**: Ensure support for emerging language models and APIs.
 - **Community Involvement**: Encourage contributions of new datasets, metrics, and use cases from the research community.
## 🖥️ Installation

It's recommended to install HintEval in a [virtual environment](https://docs.python.org/3/library/venv.html) using [Python 3.11.9](https://www.python.org/downloads/release/python-3119/). If you're not familiar with Python virtual environments, check out this [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/). Alternatively, you can create a new environment using [Conda](https://anaconda.org/anaconda/conda).

### Set up the virtual environment

First, create and activate a virtual environment with Python 3.11.9:

```bash
conda create -n hinteval_env python=3.11.9 --no-default-packages
conda activate hinteval_env
```

### Install PyTorch 2.4.0

You'll need PyTorch 2.4.0 for HintEval. Refer to the [PyTorch installation page](https://pytorch.org/get-started/previous-versions/) for platform-specific installation commands. If you have access to GPUs, it's recommended to install the CUDA version of PyTorch, as many of the evaluation metrics are optimized for GPU use.

### Install HintEval

Once PyTorch 2.4.0 is installed, you can install HintEval via pip:

```bash
pip install hinteval
```

For the latest features, you can install the most recent version from the main branch:

```bash
pip install git+https://github.com/DataScienceUIBK/HintEval
```

## 🏃 Quick Start

### Generate a Synthetic Hint Dataset

This tutorial provides step-by-step guidance on how to generate a synthetic hint dataset using large language models via the [TogetherAI platform](https://www.together.ai/). To proceed, ensure you have an active API key for TogetherAI.

```python
api_key = "your-api-key"
base_url = "https://api.together.xyz/v1"
```

#### Question/Answer Pairs

First, gather a collection of question/answer pairs as the foundation for generating Question/Answer/Hint triples. For example, load 10 questions from the WebQuestions dataset using the 🤗datasets library:

```python
from datasets import load_dataset

webq = load_dataset("Stanford/web_questions", split='test')
question_answers = webq.select_columns(['question', 'answers'])[10:20]
qa_pairs = zip(question_answers['question'], question_answers['answers'])
```

At this point, you have a set of question/answer pairs ready for creating synthetic Question/Answer/Hint instances.

#### Dataset Creation

Use HintEval's `Dataset` class to create a new dataset called `synthetic_hint_dataset`, which includes the 10 question/answer pairs within a subset named `entire`.

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

#### Hint Generation

Generate 5 hints for each question using HintEval’s `AnswerAware` model. For this example, we will use the Meta LLaMA-3.1-70b-Instruct-Turbo model from TogetherAI.

```python
from hinteval.model import AnswerAware

generator = AnswerAware(
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 
    api_key, base_url, num_of_hints=5, enable_tqdm=True
)
generator.generate(dataset['entire'].get_instances())
```

> **Note**: Depending on the LLM provider, you may need to configure the model and other parameters in the `AnswerAware` function. See the [📖documentation](http://hinteval.readthedocs.io/) for more information.

#### Exporting the Dataset

Once the hints are generated, export the synthetic hint dataset to a pickle file:

```python
dataset.store('./synthetic_hint_dataset.pickle')
```

#### Viewing the Hints

Finally, view the hints generated for the third question in the dataset:

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

---

### Evaluating Your Hint Dataset

Once your hint dataset is ready, it’s time to evaluate the hints. This section guides you through the evaluation process.

```python
api_key = "your-api-key"
base_url = "https://api.together.xyz/v1"
```

#### Load the Data

For this tutorial, use the synthetic dataset generated earlier. Alternatively, you can load a preprocessed dataset using the `Dataset.download_and_load_dataset()` function.

```python
from hinteval import Dataset

dataset = Dataset.load('./synthetic_hint_dataset.pickle')
```

#### Metrics

HintEval provides several metrics to evaluate different aspects of the hints:

- **Relevance**: Measures how relevant the hints are to the question.
- **Readability**: Assesses the readability of the hints.
- **Convergence**: Evaluates how effectively hints narrow down potential answers.
- **Familiarity**: Rates how common or well-known the hints' information is.
- **Answer Leakage**: Detects how much the hints reveal the correct answers.

Here’s how to import the metrics:

```python
from hinteval.evaluation.relevance import Rouge
from hinteval.evaluation.readability import MachineLearningBased
from hinteval.evaluation.convergence import LlmBased
from hinteval.evaluation.familiarity import Wikipedia
from hinteval.evaluation.answer_leakage import ContextualEmbeddings
```

#### Evaluate the Dataset

Extract the question, hints, and answers from the dataset and evaluate using different metrics:

```python
instances = dataset['entire'].get_instances()
questions = [instance.question for instance in instances]
answers = []
[answers.extend(instance.answers) for instance in instances]
hints = []
[hints.extend(instance.hints) for instance in instances]

# Example evaluations
Rouge('rougeL', enable_tqdm=True).evaluate(instances)
MachineLearningBased('random_forest', enable_tqdm=True).evaluate(questions + hints)
LlmBased('llama-3-70b', together_ai_api_key=api_key, enable_tqdm=True).evaluate(instances)
Wikipedia(enable_tqdm=True).evaluate(questions + hints + answers)
ContextualEmbeddings(enable_tqdm=True).evaluate(instances)
```

#### Exporting the Results

Export the evaluated dataset to a JSON file for further analysis:

```python
dataset.store_json('./evaluated_synthetic_hint_dataset.json')
```

> **Note**: Evaluated scores and metrics are automatically stored in the dataset. Saving the dataset includes the scores.

Refer to our [📖documentation](http://hinteval.readthedocs.io/) to learn more.

## ⚙️ Components
HintEval is modular and customizable, with core components designed to handle every stage of the hint generation and evaluation pipeline:

### 1. Dataset Management
 - **Preprocessed Datasets**: Includes widely used datasets like [TriviaHG](https://github.com/DataScienceUIBK/TriviaHG), [WikiHint](https://github.com/DataScienceUIBK/WikiHint), [HintQA](https://github.com/DataScienceUIBK/HintQA), and [KG-Hint](https://github.com/AlexWalcher/automaticHintGeneration).
 - **Dynamic Dataset Loading**: Use `Dataset.available_datasets()` to list, download, and load datasets effortlessly.
 - **Custom Dataset Creation**: Define datasets using the `Dataset` and `Instance` classes for tailored hint generation.

<p align="center">
<img src="./docs/source/_static/imgs/dataset-diagram.png">
</p>

### 2. Hint Generation Models
 - **Answer-Aware Models**: Generate hints tailored to specific answers using LLMs.
 - **Answer-Agnostic Models**: Generate hints without requiring specific answers for open-ended tasks.
### 3. Evaluation Metrics
 - **Relevance**: Measures how relevant the hints are to the question.
 - **Readability**: Assesses the readability of the hints.
 - **Convergence**: Evaluates how effectively hints narrow down potential answers.
 - **Familiarity**: Rates how common or well-known the hints' information is.
 - **Answer Leakage**: Detects how much the hints reveal the correct answers.

<p align="center">
<img src="./docs/source/_static/imgs/evaluators.png" width="50%">
</p>

### 4. Model Integration
 - Integrates seamlessly with API-based platforms (e.g., TogetherAI).
 - Supports custom models and local inference setups.

## 🤝Contributors

Community contributions are essential to our project, and we value every effort to improve it. From bug fixes to feature enhancements and documentation updates, your involvement makes a big difference, and we’re thrilled to have you join us! For more details, please refer to [development.](DEVELOPMENT.md)

### How to Add Your Own Dataset

If you have a dataset on hints that you'd like to share with the community, we'd love to help make it available within HintEval! Adding new, high-quality datasets enriches the framework and supports other users' research and study efforts.

To contribute your dataset, please reach out to us. We’ll review its quality and suitability for the framework, and if it meets the criteria, we’ll include it in our preprocessed datasets, making it readily accessible to all users.

To view the available preprocessed datasets, use the following code:

```python
from hinteval import Dataset

available_datasets = Dataset.available_datasets(show_info=True, update=True)
```

Thank you for considering this valuable contribution! Expanding HintEval's resources with your work benefits the entire community.

### How to Contribute

Follow these steps to get involved:

1. **Fork this repository** to your GitHub account.

2. **Create a new branch** for your feature or fix:

   ```bash
   git checkout -b feature/YourFeatureName
   ```

3. **Make your changes** and **commit them**:

   ```bash
   git commit -m "Add YourFeatureName"
   ```

4. **Push the changes** to your branch:

   ```bash
   git push origin feature/YourFeatureName
   ```

5. **Submit a Pull Request** to propose your changes.

Thank you for helping make this project better!


## 🪪License
This project is licensed under the Apache-2.0 License - see the [LICENSE](https://opensource.org/license/apache-2-0) file for details.

## ✨Citation
If you find this work useful, please cite [📜our paper](https://doi.org/10.48550/arXiv.2502.00857):
### Plain

Mozafari, J., Piryani, B., Abdallah, A., & Jatowt, A. (2025). HintEval: A Comprehensive Framework for Hint Generation and Evaluation for Questions. arXiv preprint arXiv:2502.00857.

### Bibtex
```bibtex
@ARTICLE{mozafari2025hintevalcomprehensiveframeworkhint,
       author = {{Mozafari}, Jamshid and {Piryani}, Bhawna and {Abdallah}, Abdelrahman and {Jatowt}, Adam},
        title = "{HintEval: A Comprehensive Framework for Hint Generation and Evaluation for Questions}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computation and Language, Computer Science - Information Retrieval},
         year = 2025,
        month = feb,
          doi = {10.48550/arXiv.2502.00857}
}
```

## 🙏Acknowledgments
Thanks to our contributors and the University of Innsbruck for supporting this project.
