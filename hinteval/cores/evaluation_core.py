import gc
import os
import torch
import pickle
from abc import ABC, abstractmethod
from typing import Union, List
from hinteval.cores.dataset_core import Question, Answer, Hint, Instance


class _Evaluation(ABC):
    def __init__(self, checkpoint, checkpoint_step, enable_tqdm):
        self.enable_tqdm = enable_tqdm
        self.checkpoint_step = checkpoint_step
        self.checkpoint_path = None
        if checkpoint:
            if os.environ['HINTEVAL_CHECKPOINT_DIR'] == '':
                raise ValueError(
                    'Checkpoint saving is enabled, but the checkpoint directory is not set.\n'
                    'To set the environment variable, use the following commands:\n'
                    '\n'
                    'import os\n'
                    'os.environ["HINTEVAL_CHECKPOINT_DIR"] = "/path/to/your/directory"\n'
                )
            self.checkpoint_path = os.path.abspath(os.environ['HINTEVAL_CHECKPOINT_DIR'])

    def _reload_checkpoint(self):
        _CHECKPOINT_DIR = os.environ['HINTEVAL_CHECKPOINT_DIR']
        if os.path.isfile(_CHECKPOINT_DIR):
            raise Exception(
                f"The path '{_CHECKPOINT_DIR}' points to a file, but a directory is expected. "
                "Please ensure that the checkpoint path is a directory where the checkpoint files can be stored, "
                "rather than a specific file."
            )
        new_path = os.path.join(_CHECKPOINT_DIR, self._file_name)
        if not os.path.exists(new_path):
            return dict()
        try:
            with open(new_path, 'rb') as file:
                data = pickle.load(file)
                return data
        except Exception as e:
            metric_name = self.__class__.__bases__[0].__name__
            raise Exception(
                f"An error occurred while loading the checkpoint file for {metric_name} metric: {e}")

    def _store_checkpoint(self, output):
        _CHECKPOINT_DIR = os.environ['HINTEVAL_CHECKPOINT_DIR']
        os.makedirs(_CHECKPOINT_DIR, exist_ok=True)
        try:
            new_path = os.path.join(_CHECKPOINT_DIR, self._file_name)
            with open(new_path, 'wb') as file:
                pickle.dump(output, file)
        except Exception as e:
            metric_name = self.__class__.__bases__[0].__name__
            raise Exception(
                f"An error occurred while saving the checkpoint file for {metric_name} metric: {e}")

    def _load_content_checkpoint(self):
        parent_name = self.__class__.__bases__[0].__name__
        self_name = self.__class__.__name__
        if self.checkpoint_path is not None:
            checkpoint_file_path = os.path.join(self.checkpoint_path, self._file_name)
            if os.path.exists(checkpoint_file_path):
                checkpoint_content = self._reload_checkpoint()
                print(f"Checkpoint successfully reloaded for {parent_name}-{self_name} from: {checkpoint_file_path}")
                return checkpoint_content
            else:
                print(
                    f"Checkpoint will be created and reload for {parent_name}-{self_name} from: {checkpoint_file_path}")
                return None
        return None

    def release_memory(self):
        """
        Releases the memory used by the class instance.

        This method deletes the instance of the class and triggers garbage collection to free up memory.

        Examples
        --------
        >>> from hinteval.evaluation.familiarity import Wikipedia
        >>>
        >>> wikipedia = Wikipedia(spacy_pipeline='en_core_web_sm')
        >>> wikipedia.release_memory()
        """
        for attr in list(self.__dict__.keys()):  # Use list() to avoid runtime changes
            delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()



class Relevance(_Evaluation):
    @abstractmethod
    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        pass

    def _validate_input(self, inputs):
        valid_class = Instance
        if not all(isinstance(text, valid_class) for text in inputs):
            raise ValueError(f"All items for evaluating relevance must be instances of the Instance class.")
        for text in inputs:
            if not isinstance(text.question, Question):
                raise ValueError(
                    f"The question of all instances for evaluating relevance must be an instance of Question class.")
            for hint in text.hints:
                if not isinstance(hint, Hint):
                    raise ValueError(
                        f"The hints of all instances for evaluating relevance must be an instance of Hint class.")


class Readability(_Evaluation):
    @abstractmethod
    def evaluate(self, sentences: List[Union[Question, Hint]], **kwargs) -> List[float]:
        pass

    def _validate_input(self, inputs):
        valid_classes = (Question, Hint)
        if not all(isinstance(text, valid_classes) for text in inputs):
            raise ValueError(
                "All items for evaluating readability must be instances of the following classes: Question or Hint.")


class Convergence(_Evaluation):
    @abstractmethod
    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        pass

    def _validate_input(self, inputs):
        valid_class = Instance
        if not all(isinstance(text, valid_class) for text in inputs):
            raise ValueError("All items for evaluating convergence must be instances of the Instance class.")
        for text in inputs:
            if not isinstance(text.question, Question):
                raise ValueError(
                    f"The question of all instances for evaluating convergence must be an instance of Question class.")
            for hint in text.hints:
                if not isinstance(hint, Hint):
                    raise ValueError(
                        f"The hints of all instances for evaluating convergence must be an instance of Hint class.")
            if self.__class__.__name__ == 'LlmBased':
                if len(text.answers) == 0:
                    raise ValueError(f"All instances for evaluating convergence must include an answer at least.")
                for answer in text.answers:
                    if not isinstance(answer, Answer):
                        raise ValueError(
                            f"The answers of all instances for evaluating convergence must be an instance of Answer class.")


class Familiarity(_Evaluation):
    @abstractmethod
    def evaluate(self, sentences: List[Union[Question, Hint, Answer]], **kwargs) -> List[float]:
        pass

    def _validate_input(self, inputs):
        valid_classes = (Question, Hint, Answer)
        if not all(isinstance(text, valid_classes) for text in inputs):
            raise ValueError(
                "All items for evaluating familiarity must be instances of the following classes: Question, Hint, or Answer.")


class AnswerLeakage(_Evaluation):
    @abstractmethod
    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        pass

    def _validate_input(self, inputs):
        valid_class = Instance
        if not all(isinstance(text, valid_class) for text in inputs):
            raise ValueError("All items for evaluating answer leakage must be instances of the Instance class.")
        for text in inputs:
            for hint in text.hints:
                if not isinstance(hint, Hint):
                    raise ValueError(
                        f"The hints of all instances for evaluating answer leakage must be an instance of Hint class.")
                if len(text.answers) == 0:
                    raise ValueError(f"All instances for evaluating answer leakage must include an answer at least.")
                for answer in text.answers:
                    if not isinstance(answer, Answer):
                        raise ValueError(
                            f"The answers of all instances for evaluating answer leakage must be an instance of Answer class.")
