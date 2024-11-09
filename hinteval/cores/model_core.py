import gc
import os
import torch
import pickle
from abc import ABC, abstractmethod
from typing import List
from hinteval.cores.dataset_core import Instance, Question, Answer


class _Model(ABC):
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
        >>> from hinteval.model import AnswerAware
        >>>
        >>> answer_aware = AnswerAware(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        ...                           api_key='your_api_key',
        ...                           base_url='base_url',
        ...                           num_of_hints=2)
        >>> answer_aware.release_memory()
        """

        del self
        gc.collect()
        torch.cuda.empty_cache()


class Model(_Model):
    @abstractmethod
    def generate(self, instances: List[Instance], **kwargs) -> List[List[str]]:
        pass

    def _validate_input(self, inputs):
        valid_class = Instance
        if not all(isinstance(text, valid_class) for text in inputs):
            raise ValueError("All items for generating hints must be instances of the Instance class.")
        for text in inputs:
            if not isinstance(text.question, Question):
                raise ValueError(
                    f"The question of all instances for generating hints must be an instance of Question class.")
            if self.__class__.__name__ == 'AnswerAware':
                if len(text.answers) == 0:
                    raise ValueError(f"All instances for generating hints must include an answer at least.")
                for answer in text.answers:
                    if not isinstance(answer, Answer):
                        raise ValueError(
                            f"The answers of all instances for generating hints must be an instance of Answer class.")
