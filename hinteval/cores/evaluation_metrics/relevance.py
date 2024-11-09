import spacy
import os
import re
import torch
import itertools
import numpy as np
import torch.nn as nn
from typing import List, Literal
from tqdm import tqdm
from hinteval.cores.evaluation_core import Relevance
from hinteval.cores.dataset_core import Metric, Instance
from hinteval.utils.functions.download_manager import SpacyDownloader, RelevanceNonContextualDownloader, \
    RelevanceContextualDownloader
from hinteval.utils.relevance.relevance_fixed import Batch, UnknownWordVecCache
from hinteval.utils.relevance.LiteModel import PairwiseConv, MPCNNLite
from hinteval.utils.relevance.answer_relevancy.api_based import AnswerRelevancy as AR_API
from hinteval.utils.relevance.answer_relevancy.local import AnswerRelevancy as AR_LOCAL
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from rouge_score import rouge_scorer
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class Rouge(Relevance):
    """
    Class for evaluating relevance between question and hints using ROUGE metrics `[3]`_ .

    .. _[3]: https://aclanthology.org/W04-1013/


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
    .. [3] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

    See Also
    --------
    :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe).
    :class:`ContextualEmbeddings` : Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models.
    :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

    """

    def __init__(self, model: Literal['rouge1', 'rouge2', 'rougeL'] = 'rouge1', checkpoint: bool = False,
                 checkpoint_step: int = 1, enable_tqdm: bool = False):
        """
        Initializes the Rouge class with the specified ROUGE model `[4]`_ .

        .. _[4]: https://aclanthology.org/W04-1013/

        Parameters
        ----------
        model : Literal['rouge1', 'rouge2', 'rougeL'], default 'rouge1'
            The ROUGE model to use.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Examples
        --------
        >>> from hinteval.evaluation.relevance import Rouge
        >>>
        >>> rouge = Rouge(model='rouge1', checkpoint=True, checkpoint_step=100, enable_tqdm=True)

        Raises
        ------
        ValueError
            If the provided model name is not valid.

        References
        ----------
        .. [4] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

        See Also
        --------
        :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe).
        :class:`ContextualEmbeddings` : Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        if model not in ['rouge1', 'rouge2', 'rougeL']:
            raise ValueError(
                f'Invalid model name: "{model}".\n'
                'Please choose one of the following valid models:\n'
                '- rouge1: ROUGE-1, which measures the overlap of unigrams (single words) between the generated and reference text\n'
                '- rouge2: ROUGE-2, which measures the overlap of bigrams (two-word sequences) between the generated and reference text\n'
                '- rougeL: ROUGE-L, which measures the longest common subsequence (LCS) between the generated and reference text, considering both precision and recall'
            )
        self._file_name = f'relevance_{model}.pickle'
        self._batch_size = 1
        self._model = model
        self._scorer = rouge_scorer.RougeScorer([self._model], use_stemmer=True)

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the relevance of the question and hints of the given instances using the ROUGE metric `[5]`_.

        .. _[5]: https://aclanthology.org/W04-1013/

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of relevance scores for each instance.

        Notes
        ----------
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint` of the `instances`, with names based on the model, such as "relevance-rouge1".

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.relevance import Rouge
        >>>
        >>> rouge = Rouge(model='rouge1')
        >>> instance_1 = Instance(
        ...     question=Question('What is the capital of Austria?'),
        ...     answers=[Answer('Vienna')],
        ...     hints=[Hint('This city, once home to Mozart and Beethoven.'),
        ...            Hint('This city is the best city for life in 2024.')])
        >>> instance_2 = Instance(
        ...     question=Question('Who was the president of USA in 2009?'),
        ...     answers=[Answer('Barack Obama')],
        ...     hints=[Hint('He was the first African-American president in U.S. history.'),
        ...            Hint('He was named the 2009 Nobel Peace Prize laureate.')])
        >>> instances = [instance_1, instance_2]
        >>> results = rouge.evaluate(instances)
        >>> print(results)
        # [[0.0, 0.25], [0.421, 0.353]]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...            instance in instances
        ...            for hint in instance.hints for metric_key, metric_value in
        ...            hint.metrics.items()]
        >>> print(metrics)
        # ['relevance-rouge1: 0.0', 'relevance-rouge1: 0.25', 'relevance-rouge1: 0.421', 'relevance-rouge1: 0.353']


        References
        ----------
        .. [5] Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.


        See Also
        --------
        :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe).
        :class:`ContextualEmbeddings` : Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

        """

        self._validate_input(instances)

        questions = []
        hints = []
        for instance in instances:
            questions.extend([instance.question] * len(instance.hints))
            hints.extend(instance.hints)
        question_hint = list(zip(questions, hints))

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        question_hint_stream = tqdm(question_hint, total=len(question_hint),
                                    desc=f'Evaluating relevance metric using {self._model}') if self.enable_tqdm else question_hint
        for question, hint in question_hint_stream:
            _idx = next(item_counter)
            final_step = len(results) // self._batch_size
            if _idx <= final_step:
                continue

            scores = self._scorer.score(question.question, hint.hint)
            results.append(scores[self._model].fmeasure)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    question_hint_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [round(res, 3) for res in results]
        for idx, qh in enumerate(question_hint):
            qh[1].metrics[f'relevance-{self._model}'] = Metric('relevance', results[idx])
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([results.pop(0) for _ in range(num_of_hint)])
        return final_results


class NonContextualEmbeddings(Relevance):
    """
    Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe) `[6]`_.

    .. _[6]: https://arxiv.org/abs/1909.01059

    Attributes
    ----------
    batch_size : int
        The batch size for processing.
    checkpoint : bool
        Whether checkpointing is enabled.
    checkpoint_step : int
        Step interval for checkpointing.
    enable_tqdm : bool
        Whether the tqdm progress bar is enabled.


    References
    ----------
    .. [6] Mozafari, Jamshid, Mohammad Ali Nematbakhsh, and Afsaneh Fatemi. "Attention-based pairwise multi-perspective convolutional neural network for answer selection in question answering." arXiv preprint arXiv:1909.01059 (2019).


    See Also
    --------
    :class:`Rouge` : Class for evaluating relevance between question and hints using ROUGE metrics.
    :class:`ContextualEmbeddings` : Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models.
    :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

    """

    def __init__(self, glove_version: Literal['glove.6B', 'glove.42B'] = 'glove.6B', spacy_pipeline: Literal[
        'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'] = 'en_core_web_sm',
                 batch_size: int = 256, checkpoint: bool = False, checkpoint_step: int = 1, force_download=False,
                 enable_tqdm=False):
        """
        Initializes the NonContextualEmbeddings class with the specified GloVe version and spaCy pipeline `[7]`_.

        .. _[7]: https://arxiv.org/abs/1909.01059

        Parameters
        ----------
        glove_version : {'glove.6B', 'glove.42B'}, default 'glove.6B'
            The version of GloVe embeddings to use.
        spacy_pipeline : {'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'}, default 'en_core_web_sm'
            The spaCy pipeline to use for tokenization and stop words.
        batch_size : int, default 256
            The batch size for processing.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        force_download : bool, default False
            Whether to force download of GloVe embeddings and spaCy models.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        ValueError
            If the provided model name is not valid.
        Exception
            If downloading of embeddings or models fails.


        Examples
        --------
        >>> from hinteval.evaluation.relevance import NonContextualEmbeddings
        >>>
        >>> non_contextual = NonContextualEmbeddings(glove_version='glove.6B', spacy_pipeline='en_core_web_sm',
        ...                                           batch_size=64, checkpoint=True, checkpoint_step=2, enable_tqdm=True)


        References
        ----------
        .. [7] Mozafari, Jamshid, Mohammad Ali Nematbakhsh, and Afsaneh Fatemi. "Attention-based pairwise multi-perspective convolutional neural network for answer selection in question answering." arXiv preprint arXiv:1909.01059 (2019).


        See Also
        --------
        :class:`Rouge` : Class for evaluating relevance between question and hints using ROUGE metrics.
        :class:`ContextualEmbeddings` : Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self._file_name = f'relevance_non_contextual_{glove_version}.pickle'
        self._glove_version = glove_version
        self._spacy_pipeline = spacy_pipeline
        self._sigmoid = nn.Sigmoid()

        SpacyDownloader.download(spacy_pipeline)
        self._stop_words = spacy.load(spacy_pipeline).Defaults.stop_words
        self._batch = Batch(self._stop_words, self._device, glove_version, force_download, UnknownWordVecCache.unk)
        RelevanceNonContextualDownloader.download(force_download)
        embedding = nn.Embedding(59253, 300)
        filter_widths = list(range(1, 4)) + [np.inf]
        self._model = PairwiseConv(MPCNNLite(300, 300, 20, filter_widths, 300, 0.5, 4,
                                             'basic', True, embedding))
        state_dict = torch.load(
            os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'relevance-non-contextual', 'APMPCNN.model'))
        self._model.load_state_dict(state_dict)
        self._model = self._model.to(self._device)
        self._model.eval()

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the relevance of the question and hints of the given instances using non-contextual embeddings such as Glove `[8]`_.

        .. _[8]: https://arxiv.org/abs/1909.01059

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of relevance scores for each instance.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint` of the `instances`, with names based on the model, such as "relevance-non-contextual-6B-sm".

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.relevance import NonContextualEmbeddings
        >>>
        >>> non_contextual = NonContextualEmbeddings(glove_version='glove.6B')
        >>> instance_1 = Instance(
        ...     question=Question('What is the capital of Austria?'),
        ...     answers=[Answer('Vienna')],
        ...     hints=[Hint('This city, once home to Mozart and Beethoven.'),
        ...            Hint('This city is the best city for life in 2024.')])
        >>> instance_2 = Instance(
        ...     question=Question('Who was the president of USA in 2009?'),
        ...     answers=[Answer('Barack Obama')],
        ...     hints=[Hint('He was the first African-American president in U.S. history.'),
        ...            Hint('He was named the 2009 Nobel Peace Prize laureate.')])
        >>> instances = [instance_1, instance_2]
        >>> results = non_contextual.evaluate(instances)
        >>> print(results)
        # [[0.867, 0.889], [0.91, 0.891]]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...            instance in instances
        ...            for hint in instance.hints for metric_key, metric_value in
        ...            hint.metrics.items()]
        >>> print(metrics)
        # ['relevance-non-contextual-6B-sm: 0.867', 'relevance-non-contextual-6B-sm: 0.889', 'relevance-non-contextual-6B-sm: 0.91', 'relevance-non-contextual-6B-sm: 0.891']

        References
        ----------
        .. [8] Mozafari, Jamshid, Mohammad Ali Nematbakhsh, and Afsaneh Fatemi. "Attention-based pairwise multi-perspective convolutional neural network for answer selection in question answering." arXiv preprint arXiv:1909.01059 (2019).

        See Also
        --------
        :class:`Rouge` : Class for evaluating relevance using ROUGE metrics.
        :class:`ContextualEmbeddings` : Class for evaluating relevance using contextual embeddings such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

        """

        self._validate_input(instances)
        questions = []
        hints = []
        for instance in instances:
            questions.extend([instance.question] * len(instance.hints))
            hints.extend(instance.hints)

        sentences_1 = [question.question for question in questions]
        sentences_2 = [hint.hint for hint in hints]

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        data_loader = self._batch.get_iterator(sentences_1, sentences_2, batch_size=self.batch_size)
        data_loader_stream = tqdm(data_loader, total=len(data_loader),
                                  desc=f'Evaluating relevance metric using {self._glove_version}') if self.enable_tqdm else data_loader
        for batch in data_loader_stream:
            _idx = next(item_counter)
            final_step = len(results) // self.batch_size
            if _idx <= final_step:
                continue

            scores = self._model.convModel(batch.sentence_1, batch.sentence_2, batch.ext_feats)
            scores = self._model.linearLayer(scores)
            scores = self._sigmoid(scores)
            score_array = scores.cpu().data.numpy().reshape(-1).tolist()
            results.extend(score_array)
            del scores

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    data_loader_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [round(res, 3) for res in results]
        for idx, h in enumerate(hints):
            glove_version = re.search(r'\d+B', self._glove_version).group()
            spacy_pipeline = re.search(r'web_(\w+)$', self._spacy_pipeline).group(1)
            metadata = {'glove_version': f'{self._glove_version}.300d', 'spacy_pipeline': self._spacy_pipeline}
            h.metrics[f'relevance-non-contextual-{glove_version}-{spacy_pipeline}'] = Metric('relevance', results[idx],
                                                                                             metadata=metadata)
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([results.pop(0) for _ in range(num_of_hint)])
        return final_results


class ContextualEmbeddings(Relevance):
    """
    Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models `[9]`_.

    .. _[9]: https://aclanthology.org/2020.lrec-1.676/

    Attributes
    ----------
    batch_size : int
        The batch size for processing.
    checkpoint : bool
        Whether checkpointing is enabled.
    checkpoint_step : int
        Step interval for checkpointing.
    enable_tqdm : bool
        Whether the tqdm progress bar is enabled.

    References
    ----------
    .. [9] Md Tahmid Rahman Laskar, Jimmy Xiangji Huang, and Enamul Hoque. 2020. Contextualized Embeddings based Transformer Encoder for Sentence Similarity Modeling in Answer Selection Task. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 5505–5514, Marseille, France. European Language Resources Association.

    See Also
    --------
    :class:`Rouge` : Class for evaluating relevance between question and hints using ROUGE metrics.
    :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe)
    :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

    """

    def __init__(self, model_name: Literal['bert-base', 'roberta-large'] = 'bert-base', batch_size: int = 256,
                 checkpoint: bool = False, checkpoint_step: int = 1, force_download=False, enable_tqdm=False):
        """
        Initializes the ContextualEmbeddings class with the specified model name `[10]`_.

        .. _[10]: https://aclanthology.org/2020.lrec-1.676/

        Parameters
        ----------
        model_name : {'bert-base', 'roberta-large'}, default 'bert-base'
            The name of the contextual model to use.
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
        >>> from hinteval.evaluation.relevance import ContextualEmbeddings
        >>>
        >>> contextual = ContextualEmbeddings(model_name='bert-base', batch_size=64, checkpoint=True, checkpoint_step=2, enable_tqdm=True)

        References
        ----------
        .. [10] Md Tahmid Rahman Laskar, Jimmy Xiangji Huang, and Enamul Hoque. 2020. Contextualized Embeddings based Transformer Encoder for Sentence Similarity Modeling in Answer Selection Task. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 5505–5514, Marseille, France. European Language Resources Association.


        See Also
        --------
        :class:`Rouge` : Class for evaluating relevance between question and hints using ROUGE metrics.
        :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe)
        :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._model_name = model_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self._file_name = f'relevance_contextual_{model_name}.pickle'
        RelevanceContextualDownloader.download(model_name, force_download)
        model_dir = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'relevance-contextual', self._model_name)
        self._config = AutoConfig.from_pretrained(model_dir, num_labels=2)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=self._config)
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=True)
        self._model.to(self._device)

    @staticmethod
    def _softmax(logits):
        exp_logits = np.exp(logits)
        logits_sum = np.sum(exp_logits, axis=1)
        return exp_logits / logits_sum[:, np.newaxis]

    def _tokenize_function(self, examples):
        return self._tokenizer(examples['question'], examples['answer'], padding='max_length', truncation=True)

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the relevance of the question and hints of the given instances using large language models `[11]`_.

        .. _[11]: https://aclanthology.org/2020.lrec-1.676/

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of relevance scores for each instance.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint` of the `instances`, with names based on the model, such as "relevance-contextual-bert-base".

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.relevance import ContextualEmbeddings
        >>>
        >>> contextual = ContextualEmbeddings(model_name='bert-base')
        >>> instance_1 = Instance(
        ...     question=Question('What is the capital of Austria?'),
        ...     answers=[Answer('Vienna')],
        ...     hints=[Hint('This city, once home to Mozart and Beethoven.'),
        ...            Hint('This city is the best city for life in 2024.')])
        >>> instance_2 = Instance(
        ...     question=Question('Who was the president of USA in 2009?'),
        ...     answers=[Answer('Barack Obama')],
        ...     hints=[Hint('He was the first African-American president in U.S. history.'),
        ...            Hint('He was named the 2009 Nobel Peace Prize laureate.')])
        >>> instances = [instance_1, instance_2]
        >>> results = contextual.evaluate(instances)
        >>> print(results)
        # [[1.0, 1.0], [1.0, 1.0]]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...            instance in instances
        ...            for hint in instance.hints for metric_key, metric_value in
        ...            hint.metrics.items()]
        >>> print(metrics)
        # ['relevance-contextual-bert-base: 1.0', 'relevance-contextual-bert-base: 1.0', 'relevance-contextual-bert-base: 1.0', 'relevance-contextual-bert-base: 1.0']

        References
        ----------
        .. [11] Md Tahmid Rahman Laskar, Jimmy Xiangji Huang, and Enamul Hoque. 2020. Contextualized Embeddings based Transformer Encoder for Sentence Similarity Modeling in Answer Selection Task. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pages 5505–5514, Marseille, France. European Language Resources Association.

        See Also
        --------
        :class:`Rouge` : Class for evaluating relevance between question and hints using ROUGE metrics.
        :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe)
        :class:`LlmBased` : Class for evaluating relevance between question and hints using large language models.

        """

        self._validate_input(instances)
        questions = []
        hints = []
        for instance in instances:
            questions.extend([instance.question] * len(instance.hints))
            hints.extend(instance.hints)

        data = {
            'question': [question.question for question in questions],
            'answer': [hint.hint for hint in hints]
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
                                  desc=f'Evaluating relevance metric using {self._model_name}') if self.enable_tqdm else data_loader

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
                score_array = probabilities[:, 0].reshape(-1).tolist()
                results.extend(score_array)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    data_loader_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [round(res, 3) for res in results]
        for idx, h in enumerate(hints):
            h.metrics[f'relevance-contextual-{self._model_name}'] = Metric('relevance', results[idx])
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([results.pop(0) for _ in range(num_of_hint)])
        return final_results


class LlmBased(Relevance):
    """
    Class for evaluating relevance between question and hints using large language models `[12]`_.

    .. _[12]: https://aclanthology.org/2024.eacl-demo.16/

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
    .. [12] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs: Automated Evaluation of Retrieval Augmented Generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 150–158, St. Julians, Malta. Association for Computational Linguistics.


    See Also
    --------
    :class:`Rouge` : Class for evaluating relevance between question and hints using ROUGE metrics.
    :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe)
    :class:`ContextualEmbeddings` : Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models.
    """

    def __init__(self, model_name: str,
                 api_key: str = None,
                 base_url: str = 'https://api.together.xyz/v1',
                 checkpoint: bool = False,
                 checkpoint_step: int = 1,
                 enable_tqdm=False):
        """
        Initializes the LlmBased class with the specified large language model `[13]`_.

        .. _[13]: https://aclanthology.org/2024.eacl-demo.16/

        Parameters
        ----------
        model_name : str
            The large language model to use for evaluation relevance.
        api_key : str, optional
            Specifies the API key required for accessing and interacting with the model.
        base_url : str, default 'https://api.together.xyz/v1'
            Specifies the base URL for the API endpoints. This URL is used to construct full API request URLs.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        ValueError
            If the provided model name is not valid or other parameters are out of acceptable range.
        Exception
            If initialization fails due to API or model access issues.

        Notes
        ------
        If `api_key` is None, the evaluator will attempt to download the model and run it locally.


        Examples
        --------
        >>> from hinteval.evaluation.relevance import LlmBased
        >>>
        >>> relevance = LlmBased(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key='your_api_key',
        ...                base_url='base_url', batch_size=1, checkpoint=True, checkpoint_step=5, enable_tqdm=True)

        References
        ----------
        .. [13] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs: Automated Evaluation of Retrieval Augmented Generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 150–158, St. Julians, Malta. Association for Computational Linguistics.

        See Also
        --------
        :class:`Rouge` : Class for evaluating relevance between question and hints using ROUGE metrics.
        :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe)
        :class:`ContextualEmbeddings` : Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models.
        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'relevance_llm_{model_name.replace("/", "_")}.pickle'
        self._api_key = api_key
        self._base_url = base_url
        self._batch_size = 1
        self._model_name = model_name

        if self._api_key is None:
            self._evaluator = AR_LOCAL(model_name=model_name, threshold=0.1)
        else:
            self._evaluator = AR_API(model_name=model_name, api_key=self._api_key,base_url=self._base_url, threshold=0.1)

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the relevance of the question and hints of the given instances using large language models `[14]`_.

        .. _[14]: https://aclanthology.org/2024.eacl-demo.16/

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of relevance scores for each instance.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint` of the `instances`, with names based on the model, such as "relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo".

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.relevance import LlmBased
        >>>
        >>> llm = LlmBased(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key='your_api_key', enable_tqdm=True)
        >>> instance_1 = Instance(
        ...    question=Question('What is the capital of Austria?'),
        ...    answers=[Answer('Vienna')],
        ...    hints=[Hint('This city, once home to Mozart and Beethoven.'),
        ...           Hint('This city is the best city for life in 2024.')])
        >>> instance_2 = Instance(
        ...    question=Question('Who was the president of USA in 2009?'),
        ...    answers=[Answer('Barack Obama')],
        ...    hints=[Hint('He was the first African-American president in U.S. history.'),
        ...           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
        >>> instances = [instance_1, instance_2]
        >>> results = llm.evaluate(instances)
        >>> print(results)
        # [[1.00, 0.81], [1.00, 0.95]]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...           instance in instances
        ...           for hint in instance.hints for metric_key, metric_value in
        ...           hint.metrics.items()]
        >>> print(metrics)
        # ['relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 1.00', 'relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 0.81',
        #  'relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 1.00', 'relevance-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 0.95']

        References
        ----------
        .. [14] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. RAGAs: Automated Evaluation of Retrieval Augmented Generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 150–158, St. Julians, Malta. Association for Computational Linguistics.

        See Also
        --------
        :class:`Rouge` : Class for evaluating relevance between question and hints using ROUGE metrics.
        :class:`NonContextualEmbeddings` : Class for evaluating relevance between question and hints using non-contextual embeddings such as word embeddings (GloVe)
        :class:`ContextualEmbeddings` : Class for evaluating relevance between question and hints using contextual embeddings such as BERT and RoBERTa models.

        """

        self._validate_input(instances)
        pairs = []
        for idx, instance in enumerate(instances):
            q = instance.question.question
            hs = [h.hint for h in instance.hints]
            pairs.append((q, hs))

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        question_answer_pairs_stream = tqdm(pairs, total=len(pairs),
                                            desc=f'Evaluating relevance metric using {self._model_name.replace("/", "_")}') if self.enable_tqdm else pairs
        for question, hints in question_answer_pairs_stream:
            _idx = next(item_counter)
            final_step = len(results) // self._batch_size
            if _idx <= final_step:
                continue
            scores = self._evaluator.compute_relevancy(question, hints)
            results.append(scores)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    question_answer_pairs_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [result for result in results]
        new_results = []
        [new_results.extend(result) for result in results]
        hints = []
        [hints.extend(instance.hints) for instance in instances]
        for idx, h in enumerate(hints):
            h.metrics[f'relevance-llm-{self._model_name.replace("/", "_")}'] = Metric('relevance', new_results[idx])
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([new_results.pop(0) for _ in range(num_of_hint)])
        return final_results
