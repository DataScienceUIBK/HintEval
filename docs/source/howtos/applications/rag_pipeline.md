(how-to-guides-applications-rag)=
# RAG Pipeline

This tutorial demonstrates how to implement a **RAG (Retrieval-Augmented Generation)** pipeline using hints as context. RAG combines the strengths of both a retrieval mechanism and a generative model to produce more accurate and contextually relevant responses. In this pipeline, the system retrieves hints related to a question and uses them to help guide a large language model (LLM) in generating more precise answers.

The workflow includes:
1. **Generating a dataset**: Create a dataset with questions and answers.
2. **Generating hints**: Use a model to generate hints that will later be retrieved in the RAG process.
3. **RAG Pipeline**: Use hints in the retrieval process to help the LLM generate accurate answers.
4. **Displaying results**: Display the predicted answers alongside the actual (ground truth) answers.

:::{note}
We assume you have an active API key for the TogetherAI platform and are using this platform for hint generation using
LLM. In this example, we use *meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo* as the model, which is available on the
TogetherAI platform. If you wish to use another platform, ensure the model name is valid for that platform.

For local execution, you can set `api_key` to `None`. HintEval supports running large language models (LLMs) locally
via [HuggingFace](https://huggingface.co/models).
:::

:::{Warning}
The output may vary from the example shown below due to the inherent non-deterministic nature of large language models.
:::

## Imports

Before starting the pipeline, ensure you have the necessary libraries and imports for dataset preparation, hint generation, and LLM interaction.

```python
import os
import random
from tqdm import tqdm
from prettytable import PrettyTable
from openai import OpenAI
from hinteval import Dataset
from hinteval.cores import Instance, Subset
from hinteval.model import AnswerAgnostic
```

These imports are essential for the various tasks within the pipeline:
- **`os`**: Used for setting environment variables, such as checkpoint directories.
- **`random`**: For randomness operations like selecting few-shot examples.
- **`tqdm`**: Provides progress bars for loops, improving visibility of task completion.
- **`PrettyTable`**: Used to display the final comparison of predicted and ground truth answers in a readable format.
- **`openai.OpenAI`**: For interacting with the OpenAI API to retrieve model completions.
- **`hinteval.Dataset`, `Instance`, `Subset`**: These classes manage dataset loading, creation, and instance handling.
- **`hinteval.model.AnswerAgnostic`**: The class responsible for generating answer-agnostic hints based on questions.

## 1. Generating a Dataset

In the first step, we create a dataset containing questions and corresponding answers. At this stage, no hints are included in the dataset, which will be added later.

```python
def generating_dataset():
    dataset = Dataset(name='my_dataset')
    dataset.add_subset(Subset('entire'))
    kg_dataset = Dataset.download_and_load_dataset('KG-Hint')
    kg_instances_partial = kg_dataset['entire'].get_instances()[10:30]
    for instance in kg_instances_partial:
        new_instance = Instance(question=instance.question, answers=instance.answers, hints=[])
        dataset['entire'].add_instance(new_instance)
    return dataset
```

- **Dataset creation**: The function creates a new dataset named `my_dataset` and adds a subset called `entire`.
- **KG-Hint dataset**: It loads an existing dataset (KG-Hint) that contains questions, answers, and hints. The tutorial selects a partial subset of instances (from index 10 to 30) to work with.
- **New instances**: Each selected instance consists of a question and its answers. Hints are initially set as empty lists and will be populated in the next step.

## 2. Generating Hints

Next, hints are generated for the dataset. These hints will later serve as part of the information retrieved in the RAG process to guide the LLM in generating answers.

```python
def generate_hints(dataset: Dataset):
    answer_agnostic = AnswerAgnostic('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key=api_key, batch_size=5,
                                     checkpoint=True, enable_tqdm=True)
    dataset_instances = dataset['entire'].get_instances()
    answer_agnostic.generate(dataset_instances)
```

- **AnswerAgnostic model**: We use a pre-trained model (`meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`) to generate hints in answer agnostic scenario.
- **Generate hints**: The function `generate` processes the dataset instances and generates hints, which are then associated with each question. These hints will be stored and used during the RAG pipeline.

## 3. Generating Prompts with Hints

This step involves creating a prompt for the LLM by combining the question with the generated hints. The hints act as context that helps the model generate a more accurate answer.

```python
def generate_prompt(question, context):
    return f"""Based on the context, answer the following question:
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
```

- **Generate prompt**: This function formats the input to be passed to the LLM. It clearly separates the generated hints (context) from the question, making it easier for the model to understand the context and generate an appropriate answer.

## 4. Few-Shot Learning with Examples

Few-shot learning is used to provide the model with a small number of example interactions (question-hints-answer triples). This helps improve the model's performance by showing it how to handle similar cases.

```python
def generate_shots(dataset: Dataset, num_of_shots=5):
    examples = []
    example_shots = random.sample(dataset['entire'].get_instances(), num_of_shots)

    for example in example_shots:
        example_question = example.question.question
        example_context = '\n'.join([hint.hint for hint in example.hints])
        example_answer = example.answers[0].answer

        prompt = generate_prompt(example_question, example_context)
        examples.append({"role": "user", "content": prompt})
        examples.append({"role": "assistant", "content": example_answer})

    return examples
```

- **Random selection**: This function selects a random subset of instances from the dataset to serve as few-shot learning examples.
- **Example formatting**: For each example, the function retrieves the question, the associated hints (context), and the correct answer. It then generates a structured prompt to guide the model and appends the expected answer.

## 5. RAG Pipeline

The RAG pipeline is the core of this tutorial. In this step, the model retrieves the hints for each question, formats them into a prompt, and then uses the generative model to produce an answer. The predicted answers are then compared with the actual answers.

```python
def rag(dataset: Dataset):
    system_prompt = "You are an expert assistant with access to a vast knowledge base. When a user asks a question, retrieve the most relevant information from the knowledge base and generate a clear, concise, and informative answer."
    model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    pipeline = OpenAI(base_url=base_url, api_key=api_key)

    predicted_answers = []
    ground_truths = []

    for instance in tqdm(dataset['entire'].get_instances(), desc='Rag pipeline'):
        messages = [{"role": "system", "content": system_prompt}]
        example_shots = generate_shots(dataset, 5)
        messages.extend(example_shots)

        instance_question = instance.question.question
        instance_context = '\n'.join([hint.hint for hint in instance.hints])
        instance_answer = instance.answers[0].answer

        prompt = generate_prompt(instance_question, instance_context)
        messages.append({"role": "user", "content": prompt})
        answer = pipeline.chat.completions.create(model=model_name, messages=messages)
        predicted_answer = answer.choices[0].message.content.strip()

        predicted_answers.append(predicted_answer)
        ground_truths.append(instance_answer)
    return predicted_answers, ground_truths
```

- **System prompt**: This prompt defines the system's behavior, instructing it to retrieve relevant information and generate an accurate answer.
- **Few-shot learning**: The model uses example question-hints-answer triples as few-shot learning inputs to guide its responses.
- **Main loop**: For each instance, the model retrieves the question and associated hints (context), combines them into a prompt, and generates an answer using the LLM.
- **Result collection**: The predicted answers and ground truth answers are stored for comparison.

## 6. Displaying the Results

Once the RAG pipeline completes, the predicted answers and actual answers are displayed side by side in a table for easy comparison.

```python
def print_result_table(predicted_answers, correct_answers):
    table = PrettyTable(['Predicted', 'Ground Truth'])
    for predicted, ground  in zip(predicted_answers, correct_answers):
        table.add_row([predicted, ground])
    print(table)
```

- **PrettyTable**: This function uses the PrettyTable library to create a structured table showing the predicted and actual answers side by side.

## 7. Running the Pipeline

Finally, the complete pipeline is executed. The steps include dataset generation, hint generation, running the RAG pipeline to answer questions, and displaying the results.

```python
def pipeline():
    dataset = generating_dataset()
    generate_hints(dataset)
    predicted_answers, ground_truths = rag(dataset)
    print_result_table(predicted_answers, ground_truths)
```

- **End-to-end pipeline**: This function orchestrates the entire process, from dataset creation to result display. It runs through the pipeline steps in sequence, generating the dataset, producing hints, and running the RAG pipeline.


## 8. Main

The `main` serves as the entry point for running the entire pipeline. In this block, we define the necessary configurations, such as the **base URL** for the OpenAI API, the **API key**, and other important settings like checkpoints and random seeds.

```python
if __name__ == '__main__':
    base_url = 'https://api.together.xyz/v1'
    api_key = 'your_api_key' 
    os.environ['HINTEVAL_CHECKPOINT_DIR'] = './rag_checkpoint'
    random.seed(1234)

    pipeline()
```

- **Base URL**:  `base_url = 'https://api.together.xyz/v1'`  
   This specifies the URL for the API endpoint used in the pipeline. In this case, the example is configured to use the Together API for language models. If you're using another API provider or service, replace this with the appropriate URL.

- **API Key**:  `api_key = 'your_api_key'`  
   The `api_key` is essential for authenticating requests to the API. You will need to replace `'your_api_key'` with a valid API key from the service you're using (e.g., OpenAI or other providers).

- **Checkpoint Directory**:  `os.environ['HINTEVAL_CHECKPOINT_DIR'] = './rag_checkpoint'`  
   The environment variable `HINTEVAL_CHECKPOINT_DIR` is set to define the directory where the model checkpoints are stored. Checkpoints allow the pipeline to save progress, ensuring that long-running processes can be resumed later if necessary. Here, the checkpoints are saved in the local directory `./rag_checkpoint`.

- **Random Seed**:  `random.seed(1234)`  
   Setting a seed for randomness ensures reproducibility. By fixing the seed value (in this case, 1234), the random selections made in the pipeline (e.g., for few-shot learning or data sampling) will always produce the same results when the code is re-run, making it easier to debug and compare experiments.

- **Pipeline Execution**:  `pipeline()`  
   This function call runs the entire pipeline, starting from dataset generation, hint generation, running the RAG process, and displaying the results. The pipeline is fully orchestrated and ready for execution once the script is run.


## Example Output

```
Checkpoint will be created and reload for Model-AnswerAgnostic from: /rag_checkpoint/answer_agnostic_meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo.pickle
Generating hints using meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo: 100%|██████████| 4/4 [00:37<00:00,  9.31s/it]
Rag pipeline: 100%|██████████| 20/20 [00:22<00:00,  1.13s/it]

+----------------------------+------------------------+
|         Predicted          |      Ground Truth      |
+----------------------------+------------------------+
|           1985             |          1984          |
|    Niccoló Machiavelli     |  Niccoló Machiavelli   |
|           Paris            |         Paris          |
|           1943             |          1943          |
|       Drew Weissman        |       Elan Musk        |
|         Thailand           |        Thailand        |
|           1877             |          1877          |
|        Jeff Bezos          |       Bill Gates       |
|           China            |        Morocco         |
|           1933             |          1937          |
|        Marie Curie         | Marie Skłodowska-Curie |
|           Bonn             |         Berlin         |
|           1971             |          1971          |
|      Matt Groening         |        Gröning         |
|        Louisiana           |        Alabama         |
|           1962             |          1962          |
|      Angela Merkel         |       Jef Bezos        |
|          Mexico            |         Mexico         |
|           1971             |          1970          |
|       Bill Clinton         |      Bill Clinton      |
+----------------------------+------------------------+
```

The table compares the predicted answers with the ground truth answers, showing the performance of the RAG pipeline. The predicted answers are generated by retrieving the relevant hints for each question and using them to guide the model's responses.
