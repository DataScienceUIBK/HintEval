(how-to-guides-applications-finetuning)=

# Fine-tuning

This tutorial demonstrates how to fine-tune a **large language model** using a hint dataset and leverage it for generating hints. Fine-tuning is a crucial machine learning process where a pre-trained model is adapted to a smaller, task-specific dataset. This process allows the model to retain general knowledge from pre-training and apply it to more specific tasks, such as generating hints for specific questions.

The workflow includes:
1. **Preparing the dataset**: Formatting the dataset to be compatible with the fine-tuning process.
2. **Uploading the dataset**: Uploading the formatted dataset to the fine-tuning platform.
3. **Fine-tuning the model**: Running the fine-tuning process on the dataset.
4. **Checking the model status**: Monitoring the fine-tuning job until completion.
5. **Using the fine-tuned model**: Deploying the fine-tuned model to generate hints.
6. **Generating hints**: Using the fine-tuned model to generate hints for specific test questions.

:::{note}
We assume you have an active API key for the TogetherAI platform and are using it for fine-tuning and generating hints. In this example, we fine-tune the *meta-llama/Meta-Llama-3-70B-Instruct-Turbo* model, which is available on the TogetherAI platform.
:::

:::{Warning}
The output may vary from the example shown below due to the inherent non-deterministic nature of large language models.
:::

## Installing Together Library
Before running this pipeline, you need to install the TogetherAI Python library to interact with the TogetherAI platform.

You can install it via pip:

```bash
pip install together==1.3.1
```
For more information on the Together Python library, please refer to the [Together Python GitHub repository](https://github.com/togethercomputer/together-python).

:::{note} Ensure you have the Together library installed to execute fine-tuning and interaction with the TogetherAI platform. 
:::

## Imports

Before starting, ensure you have the necessary libraries and imports for your fine-tuning process. These modules will handle dataset preparation, model fine-tuning, and hint generation.

```python
import os
import random
import json
import time
from datetime import timedelta
from hinteval import Dataset
from hinteval.model import AnswerAware
from hinteval.cores import Instance, Subset
from together import Together
from together.utils import check_file
from together.types.finetune import FinetuneJobStatus
```

These imports include:
- **`os`**, **`random`**, **`json`**, and **`time`**: Standard Python modules for handling file system, randomness, JSON, and timing operations.
- **`timedelta`**: From `datetime` to measure elapsed time during fine-tuning.
- **`hinteval`**: Provides classes and functions for managing datasets, models, and instances.
- **`Together`**: From the TogetherAI library, for fine-tuning and managing job status on the TogetherAI platform.
- **`FinetuneJobStatus`**: Used to check and monitor the status of the fine-tuning job.

## 1. Preparing the Dataset

This step involves preparing a hint dataset and converting it into a format that can be used for fine-tuning a language model. This dataset consists of questions, answers, and hints.

```python
def to_llama_template(dataset: Dataset):
    system_prompt = 'You are a helpful assistant that generates hints for user questions. You are given the question, and your goal is to generate hints for the question.'
    user_prompt = 'Generate {} hints for the following question without using "{}" word in the hints. Question: {}'
    llama_format = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    {model_answer}<|eot_id|>
    """
    prompts = []
    for instance in dataset['train'].get_instances():
        instance_question = instance.question.question
        instance_answer = instance.answers[0].answer
        instance_hints = instance.hints
        prompt_dict = dict()
        prompt_dict['system'] = system_prompt
        prompt_dict['instruction'] = user_prompt.format(len(instance_hints), instance_answer, instance_question)
        prompt_dict['output'] = ''
        for idx, hint in enumerate(instance_hints, 1):
            prompt_dict['output'] += f'{idx}. {hint.hint}\n'
        jsonl_line = {
            "text": llama_format.format(
                system_prompt=prompt_dict["system"],
                user_question=prompt_dict["instruction"],
                model_answer=prompt_dict["output"]
            )
        }
        prompts.append(jsonl_line)
    random.shuffle(prompts)
    with open('./wikihint.jsonl', "w", encoding="utf-8") as new_file:
        for prompt in prompts:
            new_file.write(json.dumps(prompt))
            new_file.write("\n")
    return './wikihint.jsonl'
```

- **`system_prompt`**: This is the instruction provided to the model, stating that it should act as a helpful assistant tasked with generating hints for the user's question.
  
- **`user_prompt`**: This template is customized to instruct the model to generate a specific number of hints for the given question. The function also specifies a constraint to avoid using the answer directly in the hints, enhancing the subtlety of the generated hints.

- **`llama_format`**: This format is used to create a text-based input for the model, structured with markers like `<|begin_of_text|>` and `<|eot_id|>`. This is the typical format for instructing the LLaMA model during fine-tuning.

- **Loop through the dataset**: 
  - **`for instance in dataset['train'].get_instances():`**: The function iterates over each instance in the dataset (each instance contains a question, answer, and hints).
  - **`prompt_dict`**: The dictionary `prompt_dict` is created for each instance, storing the `system_prompt`, `instruction`, and the output hints.
  - **Hints processing**: The function iterates over each hint associated with a question and adds them to the prompt in the format: `1. Hint 1`, `2. Hint 2`, and so on.
  
- **Shuffling and saving**: After processing all instances, the prompts are shuffled to ensure randomness during fine-tuning. Finally, the dataset is written to a JSONL (JSON Lines) file `wikihint.jsonl`.

This step ensures the dataset is in the correct format for fine-tuning with the LLaMA model.

## 2. Uploading the Dataset

Once the dataset is prepared, it is uploaded to the fine-tuning platform to start the fine-tuning process.

```python
def upload_train_dataset(train_file):
    for info in client.files.list().data:
        if info.filename.startswith('wikihint'):
            client.files.delete(info.id)

    report = check_file(train_file)
    assert report["is_check_passed"]

    response = client.files.upload(file=train_file)
    train_file_id = response.model_dump()["id"]

    return train_file_id
```

- **Check for existing files**: Before uploading, the function lists all files in the cloud storage (`client.files.list()`) and deletes any previously uploaded file that starts with the name "wikihint".

- **File validation**: The file is validated using the `check_file` function to ensure it meets the platform’s requirements before uploading.

- **Uploading the dataset**: The file is then uploaded using the `client.files.upload` function. The response from the server includes a unique file ID, which is extracted and returned.

This step ensures that the dataset is successfully uploaded to the fine-tuning platform and ready for processing.

## 3. Fine-tuning the Model

After the dataset is uploaded, the model is fine-tuned on the hint dataset.

```python
def finetune(train_file_id, n_epochs):
    job = client.fine_tuning.create(
        suffix=f"wikihint-finetuned",
        model=f"meta-llama/Meta-Llama-3-8B-Instruct",
        training_file=train_file_id,
        n_epochs=n_epochs,
        batch_size=16,
        learning_rate=1e-5
    )
    return job.id
```

- **Fine-tuning job**: 
  - **`suffix`**: Adds a suffix to the job name (e.g., `wikihint-finetuned`) to indicate that this model is specifically fine-tuned for the hint generation task on WikiHint dataset.
  - **`model`**: Specifies the base model used for fine-tuning (`meta-llama/Meta-Llama-3-8B-Instruct`).
  - **`training_file`**: The ID of the training file (previously returned from the upload step) is used to link the dataset with the fine-tuning job.
  - **`n_epochs`**: Sets the number of training epochs, which controls how many times the model will iterate over the dataset.
  - **`batch_size` and `learning_rate`**: These parameters control the optimization process during fine-tuning.

The fine-tuning job is created and started, and the function returns the job ID to track progress.

## 4. Checking the Fine-tuning Status

Since fine-tuning can take time, this function checks the job status periodically and waits until the process is complete.

```python
def check_status(job_id):
    response = client.fine_tuning.retrieve(job_id)
    print("The model is currently being fine-tuned... Please wait.")
    while response.status != FinetuneJobStatus.STATUS_COMPLETED:
        response = client.fine_tuning.retrieve(job_id)
        time.sleep(5)
    print("Model fine-tuning is complete.")
    return response.output_name
```

- **Retrieving the job status**: The function continuously retrieves the status of the fine-tuning job using `client.fine_tuning.retrieve(job_id)` and waits for the status to become `STATUS_COMPLETED`.

- **Polling with a delay**: The function sleeps for 5 seconds between each status check to avoid overwhelming the API with requests.

Once the fine-tuning is complete, the model's output name is returned, indicating that the model is ready for use.

## 5. Generating a Test Dataset

After fine-tuning the model, we generate a test dataset to evaluate its ability to generate hints.

```python
def generating_test():
    dataset = Dataset(name='my_dataset')
    dataset.add_subset(Subset('test'))
    wikihint = Dataset.download_and_load_dataset('WikiHint')
    wikihint_instances_partial = wikihint['test'].get_instances()[:5]
    for instance in wikihint_instances_partial:
        new_instance = Instance.from_strings(question=instance.question.question, answers=[instance.answers[0].answer],
                                             hints=[])
        dataset['test'].add_instance(new_instance)
    return dataset
```

- **Loading the dataset**: The function downloads and loads the `WikiHint` dataset, which contains questions, answers, and hints.

- **Partial selection**: For testing purposes, the function selects a subset of the test instances (first five instances) from the dataset.

- **Reformatting for evaluation**: It creates new instances where only the question and answers are retained, while hints are left empty. These empty hints will later be filled by the fine-tuned model during hint generation.

This step prepares a test dataset for evaluating the fine-tuned model.

## 6. Generating Hints

With the fine-tuned model, we generate hints for the test dataset.

```python
def generate_hint(deployed_model_name):
    answer_aware = AnswerAware(deployed_model_name, api_key=api_key, base_url=base_url, num_of_hints=2, batch_size=5,
                               enable_tqdm=True)
    test = generating_test()
    test_instances = test['test'].get_instances()
    answer_aware.generate(test_instances)
```

- **Model setup**: The function sets up the `AnswerAware` model (a version of the fine-tuned model) using the model name provided by the user, the API key, and the base URL.

- **Test dataset**: It loads the test dataset and retrieves its instances.

- **Generating hints**: The `answer_aware.generate` function generates the hints for each test instance, updating the dataset with the generated hints.

This step evaluates the fine-tuned model by generating hints for new questions.

## 7. Running the Pipeline

This function orchestrates the entire pipeline, starting from dataset preparation to fine-tuning, and ending with hint generation.

```python
def pipeline():
    wikihint = Dataset.download_and_load_dataset('WikiHint')
    train_file = to_llama_template(wikihint)
    train_file_id = upload_train_dataset(train_file)
    job_id = finetune(train_file_id, 3)
    start_time = time.time()
    finetuned_model_name = check_status(job_id)
    end_time = time.time()
    execution_time = timedelta(seconds=end_time - start_time)
    print(f'Execution time: {execution_time}')
    print(f'Model name: {finetuned_model_name}')
    deployed_model_name = input('Deploy your fine-tuned model and then enter the deployed model name: ')
    generate_hint(deployed_model_name)
    print('Stop deploying your fine-tuned model.')
```

- **Dataset preparation**: The dataset is loaded, formatted, and uploaded to the platform.

- **Fine-tuning**: The model is fine-tuned, and the function waits for completion, recording the total execution time.

- **Deployment**: The user is prompted to deploy the fine-tuned model, and its name is passed to the `generate_hint` function for evaluation.

This function combines all the previously defined steps into a single end-to-end process.

:::{note} For instructions on how to deploy your fine-tuned model on TogetherAI, please refer to the [official TogetherAI documentation](https://docs.together.ai/docs/fine-tuning-overview#deploying-your-fine-tuned-model).
:::

## 8. Main

The `main` function serves as the entry point for running the entire pipeline.


```python
if __name__ == '__main__':
    base_url = 'https://api.together.xyz/v1'
    api_key = 'your_api_key' 
    random.seed(1234)
    client = Together(api_key=api_key)

    pipeline()
```

- **API setup**: The base URL and API key are configured for accessing the TogetherAI platform.
- **Random seed**: The seed ensures reproducibility of the random operations in the pipeline.

The pipeline is then executed, orchestrating all the steps described above.

## Example Output

```
Uploading file wikihint.jsonl: 100%|██████████| 955k/955k [00:00<00:00, 1.14MB/s]
The model is currently being fine-tuned... Please wait.
Model fine-tuning is complete.
Execution time: 0:08:40.903188
Model name: your_finetuned_model_name
You should deploy your fine-tuned model and then enter the deployed model name: your_finetuned_model_name-deployed
Generating hints using your_finetuned_model_name-deployed: 100%|██████████| 1/1 [00:02<00:00,  2.20s/it]
Stop deploying your fine-tuned model.
```

This output shows the progression of the pipeline, from dataset upload to model fine-tuning, and hint generation. The model has been successfully fine-tuned and deployed to generate hints for new questions.