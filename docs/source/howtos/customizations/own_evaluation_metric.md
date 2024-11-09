(how-to-guides-customizations-own-eval-metric)=

# Your Own Evaluation Metric

The **HintEval** framework enables users to implement their own evaluation methods for any metric. If the built-in methods don’t fit your requirements, you can easily define and customize your own.

## Creating a Custom Readability Method

In this guide, we will walk you through the process of creating a custom *Readability* evaluation method.

### 1. Import the Required Classes and Packages

Start by importing the necessary classes and packages. The `Question` and `Hint` objects represent the input data, while `Metric` is used to store and represent evaluation results.

```python
import itertools
from typing import Union, List
from tqdm import tqdm
from hinteval.cores import Metric, Question, Hint
from hinteval.cores.evaluation_core import Readability
```

### 2. Define Your Class by Extending *Readability*

Next, define a new class (e.g., `MyOwnReadabilityMethod`) that extends the `Readability` class. This will allow your custom method to inherit key functionalities and interact seamlessly with the broader HintEval system.

#### Checkpoints and Progress

To include support for **checkpoints** (which save progress after processing a set number of items) and **progress bars** (to track processing status), be sure to integrate these features into your `__init__` method. Also, set a unique name for the checkpoint file in the `_file_name` attribute.

```python
class MyOwnReadabilityMethod(Readability):
    def __init__(self, checkpoint: bool = False, checkpoint_step: int = 1, enable_tqdm: bool = False):
        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = 'my_own_readability.pickle'  # Unique checkpoint file name to avoid conflicts
```

### 3. Implement the Custom Readability Method

Now, define the custom readability method that computes a score for each `Question` or `Hint` object. For example, the method below calculates readability based on word count, though you can customize this logic.

```python
def _YOUR_METHOD(self, sentence: Union[Question, Hint]) -> float:
    # Custom logic to compute readability score
    # For example, you could calculate readability based on sentence length, complexity, etc.
    text = sentence.question if isinstance(sentence, Question) else sentence.hint
    return float(len(text.split()))  # Example: word count as readability
```

### 4. Implement the *evaluate* Method

The `evaluate` method must be implemented to compute readability scores and follow the signature of other readability evaluation methods.

#### Input Validation

Use the `_validate_input` method to ensure that the input list contains valid `Question` or `Hint` objects.

```python
def evaluate(self, sentences: List[Union[Question, Hint]], **kwargs) -> List[float]:
    self._validate_input(sentences)
```

#### Checkpoint Handling (Optional)

For checkpoint support, you can use pre-implemented functions like `_load_content_checkpoint()` to load any previously saved progress.

```python
checkpoint_content = self._load_content_checkpoint()
results = [] if checkpoint_content is None else checkpoint_content['results']
item_counter = itertools.count(1)  # Track progress
```

#### Progress Bar (Optional)

If you enable a progress bar, use `tqdm` to display the evaluation progress as sentences are processed.

```python
sentences_stream = tqdm(sentences, total=len(sentences),
                            desc=f'Evaluating readability metric using My Own Method') if self.enable_tqdm else sentences
```

### 5. Process Each Sentence and Compute Readability

Loop through each sentence, applying your custom method to calculate readability scores.

```python
for sentence in sentences_stream:
    # Skip already processed sentences (if using checkpoints)
    _idx = next(item_counter)
    final_step = len(results)
    if _idx <= final_step:
        continue

    # Compute readability score using your custom method
    result = self._YOUR_METHOD(sentence)
    results.append(result)

    # Save progress at regular intervals (checkpoint)
    if (_idx % self.checkpoint_step == 0 or _idx == len(sentences)) and self.checkpoint_path is not None:
        self._store_checkpoint({'results': results})
```

### 6. Assign the Metric to Each Sentence

Once the scores are computed, store them as `Metric` objects within each `Question` or `Hint` in the `sentences` list.

```python
results = [round(res, 3) for res in results]  # Round to 3 decimal places for consistency
for idx, sentence in enumerate(sentences):
    sentence.metrics[f'readability-my-own-method'] = Metric('readability', results[idx])
```

### 7. Return the Computed Scores

Finally, return the computed list of readability scores.

```python
return results
```

## Complete Class Implementation

Here’s the final implementation of the custom readability evaluation method:

```python
import itertools
from typing import Union, List
from tqdm import tqdm
from hinteval.cores import Metric, Question, Hint
from hinteval.cores.evaluation_core import Readability

class MyOwnReadabilityMethod(Readability):
    def __init__(self, checkpoint: bool = False, checkpoint_step: int = 1, enable_tqdm: bool = False):
        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = 'my_own_readability.pickle'  # Unique checkpoint file name

    def _YOUR_METHOD(self, sentence: Union[Question, Hint]) -> float:
        # Example: return word count as readability score
        text = sentence.question if isinstance(sentence, Question) else sentence.hint
        return float(len(text.split()))

    def evaluate(self, sentences: List[Union[Question, Hint]], **kwargs) -> List[float]:
        self._validate_input(sentences)

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        sentences_stream = tqdm(sentences, total=len(sentences),
                                    desc=f'Evaluating readability metric using My Own Method') if self.enable_tqdm else sentences

        for sentence in sentences_stream:
            _idx = next(item_counter)
            final_step = len(results)
            if _idx <= final_step:
                continue

            result = self._YOUR_METHOD(sentence)
            results.append(result)

            if (_idx % self.checkpoint_step == 0 or _idx == len(sentences)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [round(res, 3) for res in results]
        for idx, sentence in enumerate(sentences):
            sentence.metrics[f'readability-my-own-method'] = Metric('readability', results[idx])

        return results
```

## Example Usage

Here's how to apply your new readability method:

```python
import os
os.environ["HINTEVAL_CHECKPOINT_DIR"] = "./check_dir"

from hinteval import Dataset

dataset = Dataset.download_and_load_dataset('WikiHint')
readability = MyOwnReadabilityMethod(checkpoint=True, checkpoint_step=10, enable_tqdm=True)

hints = []
[hints.extend(instance.hints) for instance in dataset['train'].get_instances()]

results = readability.evaluate(hints)
print(results)
```

### Example Output

```
Checkpoint will be created and reloaded for Readability-MyOwnReadabilityMethod from: /check_dir/my_own_readability.pickle
Evaluating readability metric using My Own Method: 100%|██████████| 4500/4500 [00:00<00:00, 63794.28it/s]
[9.0, 11.0, 15.0, 4.0, 11.0, 20.0, 13.0, 11.0, 13.0, 21.0, 11.0, 23.0, 26.0, 17.0, 23.0, 13.0, 23.0, 14.0, 11.0, ... ]
```

This comprehensive guide enables you to create a custom evaluation metric that fits your needs while utilizing all the features of the **HintEval** framework.
