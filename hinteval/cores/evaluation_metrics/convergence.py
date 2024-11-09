import torch
import os
import itertools
import asyncio
import transformers
import json
import numpy as np
from hinteval.cores.evaluation_core import Convergence
from hinteval.cores.dataset_core import Metric, Instance
from hinteval.utils.functions.download_manager import ConvergenceSpecificityDownloader, ConvergenceNNDownloader, ConvergenceLLMDownloader
from hinteval.utils.convergence.api_based.can_ans_generator import CanAnsGenerator as Can_Ans_Generator_API
from hinteval.utils.convergence.api_based.hint_scorer import HintScorer as Hint_Scorer_API
from hinteval.utils.convergence.local.can_ans_generator import CanAnsGenerator as Can_Ans_Generator_Local
from hinteval.utils.convergence.local.hint_scorer import HintScorer as Hint_Scorer_Local
from hinteval.utils.convergence.metrics import Metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from typing import List, Literal
from torch.utils.data import DataLoader
from datasets import Dataset
from tqdm import tqdm
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class Specificity(Convergence):
    """
    Class for evaluating specificity of :class:`Hint` using neural network models such as BERT and RoBERTa `[27]`_.

    .. _[27]: https://dl.acm.org/doi/abs/10.1145/3477495.3531734

    Attributes
    ----------
    checkpoint : bool
        Whether checkpointing is enabled.
    checkpoint_step : int
        Step interval for checkpointing.
    enable_tqdm : bool
        Whether the tqdm progress bar is enabled.

    References
    ----------
    .. [27] Jiexin Wang, Adam Jatowt, and Masatoshi Yoshikawa. 2022. ArchivalQA: A Large-scale Benchmark Dataset for Open-Domain Question Answering over Historical News Collections. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 3025–3035. https://doi.org/10.1145/3477495.3531734

    See Also
    --------
    :class:`NeuralNetworkBased` : Class for evaluating convergence between question and hints using neural network models such as BERT and RoBERTa.
    :class:`LlmBased` : Class for evaluating convergence between question and hints using large language models such as LLaMA-3-8b and LLaMA-3-70b models.

    """

    def __init__(self, model_name: Literal['bert-base', 'roberta-large'] = 'bert-base', batch_size: int = 256,
                 checkpoint: bool = False, checkpoint_step: int = 1, force_download=False, enable_tqdm=False):
        """
        Initializes the Specificity class with the specified neural network model `[28]`_.

        .. _[28]: https://dl.acm.org/doi/abs/10.1145/3477495.3531734

        Parameters
        ----------
        model_name : {'bert-base', 'roberta-large'}, default 'bert-base'
            The neural network model to use.
        batch_size : int, default 256
            The batch size for processing.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        force_download : bool, default False
            Whether to force download of models.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        ValueError
            If the provided model name is not valid.
        Exception
            If downloading of models fails.

        Examples
        --------
        >>> from  hinteval.evaluation.convergence import Specificity
        >>>
        >>> specificity = Specificity(model_name='bert-base', batch_size=64, checkpoint=True, checkpoint_step=10, enable_tqdm=True)

        References
        ----------
        .. [28] Jiexin Wang, Adam Jatowt, and Masatoshi Yoshikawa. 2022. ArchivalQA: A Large-scale Benchmark Dataset for Open-Domain Question Answering over Historical News Collections. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 3025–3035. https://doi.org/10.1145/3477495.3531734

        See Also
        --------
        :class:`NeuralNetworkBased` : Class for evaluating convergence between question and hints using neural network models such as BERT and RoBERTa.
        :class:`LlmBased` : Class for evaluating convergence between question and hints using large language models such as LLaMA-3-8b and LLaMA-3-70b models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'specificity_{model_name}.pickle'
        self._model_name = model_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        ConvergenceSpecificityDownloader.download(model_name, force_download)
        model_dir = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'convergence-specificity', self._model_name)
        self._config = AutoConfig.from_pretrained(model_dir, num_labels=2)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=self._config)
        _tokenizer_type = 'bert-base-uncased' if model_name == 'bert-base' else 'roberta-large'
        self._tokenizer = AutoTokenizer.from_pretrained(_tokenizer_type, do_lower_case=True)
        self._model.to(self._device)
        self._result_dict = {0: 'general', 1: 'specific'}

    @staticmethod
    def _softmax(logits):
        exp_logits = np.exp(logits)
        logits_sum = np.sum(exp_logits, axis=1)
        return exp_logits / logits_sum[:, np.newaxis]

    def _tokenize_function(self, examples):
        return self._tokenizer(examples['question'], examples['hint'], max_length=128, padding='max_length',
                               truncation=True)

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the specificity of the :class:`Hint` of the given instances using the specified neural network model `[29]`_.

        .. _[29]: https://dl.acm.org/doi/abs/10.1145/3477495.3531734

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of specificity scores for each instance.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint`, with names based on the model, such as "convergence-specificity-bert-base".

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.convergence import Specificity
        >>>
        >>> specificity = Specificity(model_name='bert-base')
        >>> instance_1 = Instance(
        ...     question=Question('What is the capital of Austria?'),
        ...     answers=[Answer('Vienna')],
        ...     hints=[Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')])
        >>> instance_2 = Instance(
        ...     question=Question('Who was the president of USA in 2009?'),
        ...     answers=[Answer('Barack Obama')],
        ...     hints=[Hint('He was the first African-American president in U.S. history.')])
        >>> instances = [instance_1, instance_2]
        >>> results = specificity.evaluate(instances)
        >>> print(results)
        # [[1], [1]]
        >>> classes = [sent.hints[0].metrics['convergence-specificity-bert-base'].metadata['description'] for sent in instances]
        >>> print(classes)
        # ['specific', 'specific']
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...            instance in instances
        ...            for hint in instance.hints for metric_key, metric_value in
        ...            hint.metrics.items()]
        >>> print(metrics)
        # ['convergence-specificity-bert-base: 1', 'convergence-specificity-bert-base: 1']


        References
        ----------
        .. [29] Jiexin Wang, Adam Jatowt, and Masatoshi Yoshikawa. 2022. ArchivalQA: A Large-scale Benchmark Dataset for Open-Domain Question Answering over Historical News Collections. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22). Association for Computing Machinery, New York, NY, USA, 3025–3035. https://doi.org/10.1145/3477495.3531734

        See Also
        --------
        :class:`NeuralNetworkBased` : Class for evaluating convergence between question and hints using neural network models such as BERT and RoBERTa.
        :class:`LlmBased` : Class for evaluating convergence between question and hints using large language models such as LLaMA-3-8b and LLaMA-3-70b models.

        """

        self._validate_input(instances)
        questions = []
        hints = []
        for instance in instances:
            questions.extend([instance.question] * len(instance.hints))
            hints.extend(instance.hints)

        data = {
            'question': [question.question for question in questions],
            'hint': [hint.hint for hint in hints]
        }

        _dataset = Dataset.from_dict(data)
        tokenized_dataset = _dataset.map(self._tokenize_function, batched=True)
        if self._model_name == 'bert-base':
            columns = ['input_ids', 'token_type_ids', 'attention_mask']
        else:
            columns = ['input_ids', 'attention_mask']

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        tokenized_dataset.set_format(type='torch', columns=columns)
        data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=self.batch_size)
        data_loader_stream = tqdm(data_loader, total=len(data_loader),
                                  desc=f'Evaluating specificity metric using {self._model_name}') if self.enable_tqdm else data_loader
        for batch in data_loader_stream:
            _idx = next(item_counter)
            final_step = len(results) // self.batch_size
            if _idx <= final_step:
                continue

            self._model.eval()
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self._model(**batch)
                logits = outputs.logits
                preds = logits.detach().cpu().numpy()
                probabilities = self._softmax(preds)
                score_array = np.argmax(probabilities, axis=1).tolist()
                results.extend(score_array)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    data_loader_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        for idx, h in enumerate(hints):
            h.metrics[f'convergence-specificity-{self._model_name}'] = Metric('convergence', results[idx])
            h.metrics[f'convergence-specificity-{self._model_name}'].metadata[f'description'] = self._result_dict[
                results[idx]]
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([results.pop(0) for _ in range(num_of_hint)])
        return final_results


class NeuralNetworkBased(Convergence):
    """
    Class for evaluating convergence between question and hints using neural network models such as BERT and RoBERTa.

    Attributes
    ----------
    checkpoint : bool
        Whether checkpointing is enabled.
    checkpoint_step : int
        Step interval for checkpointing.
    enable_tqdm : bool
        Whether the tqdm progress bar is enabled.

    See Also
    --------
    :class:`Specificity` : Class for evaluating specificity of Hint using neural network models such as BERT and RoBERTa models.
    :class:`LlmBased` : Class for evaluating convergence between question and hints using large language models such as LLaMA-3-8b and LLaMA-3-70b models.

    """

    def __init__(self, model_name: Literal['bert-base', 'roberta-large'] = 'bert-base', batch_size: int = 256,
                 checkpoint: bool = False, checkpoint_step: int = 1, force_download=False, enable_tqdm=False):
        """
        Initializes the NeuralNetworkBased class with the specified neural network model.

        Parameters
        ----------
        model_name : {'bert-base', 'roberta-large'}, default 'bert-base'
            The neural network model to use.
        batch_size : int, default 256
            The batch size for processing.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        force_download : bool, default False
            Whether to force download of models.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        ValueError
            If the provided model name is not valid.
        Exception
            If downloading of models fails.

        Examples
        --------
        >>> from hinteval.evaluation.convergence import NeuralNetworkBased
        >>>
        >>> neural_network = NeuralNetworkBased(model_name='bert-base', batch_size=64, checkpoint=True, checkpoint_step=10, enable_tqdm=True)

        See Also
        --------
        :class:`Specificity` : Class for evaluating specificity of Hint using neural network models such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating convergence between question and hints using large language models such as LLaMA-3-8b and LLaMA-3-70b models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'convergence_{model_name}.pickle'
        self._model_name = model_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        ConvergenceNNDownloader.download(model_name, force_download)
        model_dir = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'convergence-nn', self._model_name)
        self._config = AutoConfig.from_pretrained(model_dir, num_labels=11)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=self._config)
        _tokenizer_type = 'bert-base-uncased' if model_name == 'bert-base' else 'roberta-large'
        self._tokenizer = AutoTokenizer.from_pretrained(_tokenizer_type, do_lower_case=True)
        self._model.to(self._device)

    @staticmethod
    def _softmax(logits):
        exp_logits = np.exp(logits)
        logits_sum = np.sum(exp_logits, axis=1)
        return exp_logits / logits_sum[:, np.newaxis]

    def _tokenize_function(self, examples):
        return self._tokenizer(examples['question'], examples['hint'], max_length=128, padding='max_length',
                               truncation=True)

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the convergence between question and hints of the given instances using the specified neural network model.

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of convergence scores for each instance.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint`, with names based on the model, such as "convergence-nn-bert-base".

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.convergence import NeuralNetworkBased
        >>>
        >>> neural_network = NeuralNetworkBased(model_name='bert-base')
        >>> instance_1 = Instance(
        ...     question=Question('What is the capital of Austria?'),
        ...     answers=[Answer('Vienna')],
        ...     hints=[Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')])
        >>> instance_2 = Instance(
        ...     question=Question('Who was the president of USA in 2009?'),
        ...     answers=[Answer('Barack Obama')],
        ...     hints=[Hint('He was named the 2009 Nobel Peace Prize laureate')])
        >>> instances = [instance_1, instance_2]
        >>> results = neural_network.evaluate(instances)
        >>> print(results)
        # [[1.0], [1.0]]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...            instance in instances
        ...            for hint in instance.hints for metric_key, metric_value in
        ...            hint.metrics.items()]
        >>> print(metrics)
        # ['convergence-nn-bert-base: 1.0', 'convergence-nn-bert-base: 1.0']

        See Also
        --------
        :class:`Specificity` : Class for evaluating specificity of Hint using neural network models such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating convergence between question and hints using large language models such as LLaMA-3-8b and LLaMA-3-70b models.

        """

        self._validate_input(instances)
        questions = []
        hints = []
        for instance in instances:
            questions.extend([instance.question] * len(instance.hints))
            hints.extend(instance.hints)
        data = {
            'question': [question.question for question in questions],
            'hint': [hint.hint for hint in hints]
        }

        _dataset = Dataset.from_dict(data)
        tokenized_dataset = _dataset.map(self._tokenize_function, batched=True)
        if self._model_name == 'bert-base':
            columns = ['input_ids', 'token_type_ids', 'attention_mask']
        else:
            columns = ['input_ids', 'attention_mask']

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        tokenized_dataset.set_format(type='torch', columns=columns)
        data_loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=self.batch_size)
        data_loader_stream = tqdm(data_loader, total=len(data_loader),
                                  desc=f'Evaluating convergence metric using {self._model_name}') if self.enable_tqdm else data_loader

        for batch in data_loader_stream:
            _idx = next(item_counter)
            final_step = len(results) // self.batch_size
            if _idx <= final_step:
                continue

            self._model.eval()
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self._model(**batch)
                logits = outputs.logits
                preds = logits.detach().cpu().numpy()
                probabilities = self._softmax(preds)
                score_array = np.argmax(probabilities, axis=1).tolist()
                score_array = [round(scr / 10, 1) for scr in score_array]
                results.extend(score_array)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    data_loader_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        for idx, h in enumerate(hints):
            h.metrics[f'convergence-nn-{self._model_name}'] = Metric('convergence', results[idx])
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([results.pop(0) for _ in range(num_of_hint)])
        return final_results


class LlmBased(Convergence):
    """
    Class for evaluating convergence between question and hints using large language models such as LLaMA-3-8b and LLaMA-3-70b `[30]`_.

    .. _[30]: https://dl.acm.org/doi/10.1145/3626772.3657855

    Attributes
    ----------
    checkpoint : bool
        Whether checkpointing is enabled.
    checkpoint_step : int
        Step interval for checkpointing.
    enable_tqdm : bool
        Whether the tqdm progress bar is enabled.

    References
    ----------
    .. [30] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

    See Also
    --------
    :class:`Specificity` : Class for evaluating specificity of Hint using neural network models such as BERT and RoBERTa models.
    :class:`NeuralNetworkBased` : Class for evaluating convergence between question and hints using neural network models such as BERT and RoBERTa.

    """

    def __init__(self, model_name: Literal['llama-3-8b', 'llama-3-70b'] = 'llama-3-8b', together_ai_api_key: str = None,
                 checkpoint: bool = False, checkpoint_step: int = 1, enable_tqdm=False):
        """
        Initializes the LlmBased class with the specified large language model `[31]`_.

        .. _[31]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        model_name : {'llama-3-8b', 'llama-3-70b'}, default 'llama-3-8b'
            The large language model to use.
        together_ai_api_key : str, optional
            Specifies the API key for the Together.ai platform, required for accessing and interacting with the model.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        ValueError
            If the provided model name is not valid.
        Exception
            If downloading of models fails.

        Notes
        -----
        If `together_ai_api_key` is None, the evaluator will attempt to download the model and run it locally.


        Examples
        --------
        >>> from hinteval.evaluation.convergence import LlmBased
        >>>
        >>> llm = LlmBased(model_name='llama-3-8b', together_ai_api_key='your_api_key', checkpoint=True, checkpoint_step=10, enable_tqdm=True)

        References
        ----------
        .. [31] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`Specificity` : Class for evaluating specificity of Hint using neural network models such as BERT and RoBERTa models.
        :class:`NeuralNetworkBased` : Class for evaluating convergence between question and hints using neural network models such as BERT and RoBERTa.

        """

        if model_name not in ['llama-3-8b', 'llama-3-70b']:
            raise ValueError(
                f'Invalid model name: "{model_name}".\n'
                'Please choose one of the following valid models:\n'
                '- llama-3-8b: A version of the LLaMA (Large Language Model Meta AI) model with 8 billion parameters. Suitable for various NLP tasks with a balance between performance and computational requirements.\n'
                '- llama-3-70b: A larger LLaMA model with 70 billion parameters, providing enhanced performance and capabilities for complex NLP tasks at the cost of higher computational resources.'
            )
        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'convergence_{model_name}.pickle'
        self._batch_size = 1
        self._api_key = together_ai_api_key
        self._base_url = 'https://api.together.xyz/v1'
        self._model_name = model_name
        self._metrics = Metrics()
        if self._api_key is None:
            self._model_type = 'meta-llama/Meta-Llama-3-8B-Instruct' if model_name == 'llama-3-8b' else 'meta-llama/Meta-Llama-3-70B-Instruct'
            self._pipeline = transformers.pipeline(
                "text-generation",
                model=self._model_type,
                device_map="auto"
            )
            self._pipeline.tokenizer.pad_token_id = self._pipeline.model.config.eos_token_id
            self._candidate_generator = Can_Ans_Generator_Local(11, self._pipeline)
            self._hint_evaluator = Hint_Scorer_Local(self._pipeline)
        else:
            ConvergenceLLMDownloader.download(force_download=True)
            with open(os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'convergence-llm', 'together_models.json'), mode='r') as f:
                together_models = json.load(f)
            self._model_type = together_models['llama-3-8b'] if model_name == 'llama-3-8b' else together_models['llama-3-70b']
            self._candidate_generator = Can_Ans_Generator_API(11, self._base_url, self._api_key, self._model_type)
            self._hint_evaluator = Hint_Scorer_API(self._base_url, self._api_key, self._model_type)

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the convergence between question and hints of the given instances using the specified large language model `[32]`_.

        .. _[32]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of convergence scores for each instance.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint`, with names based on the model, such as "convergence-llm-llama-3-8b".

        This function also stores the candidate answers in the `metadata` of the :class:`Question`. Moreover, it stores the scores for each hint in the `metadata` attribute of the :class:`Hint`.


        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.convergence import LlmBased
        >>>
        >>> llm = LlmBased(model_name='llama-3-8b', together_ai_api_key='your_api_key')
        >>> instance_1 = Instance(
        ...     question=Question('What is the capital of Austria?'),
        ...     answers=[Answer('Vienna')],
        ...     hints=[Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')])
        >>> instance_2 = Instance(
        ...     question=Question('Who was the president of USA in 2009?'),
        ...     answers=[Answer('Barack Obama')],
        ...     hints=[Hint('He was the first African-American president in U.S. history.')])
        >>> instances = [instance_1, instance_2]
        >>> results = llm.evaluate(instances)
        >>> print(results)
        # [[0.91], [1.0]]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...        instance in instances
        ...        for hint in instance.hints for metric_key, metric_value in
        ...        hint.metrics.items()]
        >>> print(metrics)
        # ['convergence-llm-llama-3-8b: 0.91', 'convergence-llm-llama-3-8b: 1.0']
        >>> scores = [hint.metrics['convergence-llm-llama-3-8b'].metadata['scores'] for inst in instances for hint in inst.hints]
        >>> print(scores[0])
        # {'Salzburg': 1, 'Graz': 0, 'Innsbruck': 0, 'Linz': 0, 'Klagenfurt': 0, 'Bregenz': 0, 'Wels': 0, 'St. Pölten': 0, 'Eisenstadt': 0, 'Sankt Johann impong': 0, 'Vienna': 1}
        >>> print(scores[1])
        # {'George W. Bush': 0, 'Bill Clinton': 0, 'Jimmy Carter': 0, 'Donald Trump': 0, 'Joe Biden': 0, 'Ronald Reagan': 0, 'Richard Nixon': 0, 'Gerald Ford': 0, 'Franklin D. Roosevelt': 0, 'Theodore Roosevelt': 0, 'Barack Obama': 1}

        References
        ----------
        .. [32] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`Specificity` : Class for evaluating specificity of Hint using neural network models such as BERT and RoBERTa models.
        :class:`NeuralNetworkBased` : Class for evaluating convergence between question and hints using neural network models such as BERT and RoBERTa.

        """

        self._validate_input(instances)
        pairs = []
        for idx, instance in enumerate(instances):
            q = instance.question.question
            hs = [h.hint for h in instance.hints]
            a = instance.answers[0].answer
            cands = instance.question.metadata[
                f'candidate_answers-{self._model_name}'] if f'candidate_answers-{self._model_name}' in instance.question.metadata else None
            pairs.append((q, hs, a, cands))

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        scores_lst = [] if checkpoint_content is None else checkpoint_content['scores_lst']
        candidate_answers = [] if checkpoint_content is None else checkpoint_content['candidate_answers']
        item_counter = itertools.count(1)

        question_answer_pairs_stream = tqdm(pairs, total=len(pairs),
                                            desc=f'Evaluating convergence metric using {self._model_name}') if self.enable_tqdm else pairs
        for question, hints, answer, candidates in question_answer_pairs_stream:
            _idx = next(item_counter)
            final_step = len(candidate_answers) // self._batch_size
            if _idx <= final_step:
                continue
            if candidates is None:
                candidates = self._candidate_generator.generate_candidate_answers(question, answer)
            if self._api_key is None:
                scores = self._hint_evaluator.rate(hints, candidates)
            else:
                scores = asyncio.run(self._hint_evaluator.rate(hints, candidates))
            convergences = self._metrics.compute_metrics(scores)
            scores_lst.extend(scores)
            results.extend(convergences)
            candidate_answers.append(candidates)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    question_answer_pairs_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint(
                    {'results': results, 'scores_lst': scores_lst, 'candidate_answers': candidate_answers})

        hints = []
        [hints.extend(instance.hints) for instance in instances]
        for idx, h in enumerate(hints):
            h.metrics[f'convergence-llm-{self._model_name}'] = Metric('convergence', results[idx])
            h.metrics[f'convergence-llm-{self._model_name}'].metadata['scores'] = scores_lst[idx]
        for idx, instance in enumerate(instances):
            instance.question.metadata[f'candidate_answers-{self._model_name}'] = candidate_answers[idx]
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([results.pop(0) for _ in range(num_of_hint)])
        return final_results
