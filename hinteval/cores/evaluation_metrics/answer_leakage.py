import spacy
import re
import torch
import string
import itertools
from hinteval.utils.functions.download_manager import SpacyDownloader
from sentence_transformers import SentenceTransformer
from tok import word_tokenize
from typing import List, Literal
from tqdm import tqdm
from hinteval.cores.evaluation_core import AnswerLeakage
from hinteval.cores.dataset_core import Metric, Instance
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class Lexical(AnswerLeakage):
    """
    Class for evaluating answer leakage of hints using lexical comparison `[36]`_ .

    .. _[36]: https://dl.acm.org/doi/10.1145/3626772.3657855


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
    .. [36] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

    See Also
    --------
    :class:`ContextualEmbeddings` : Class for evaluating answer leakage of hints using contextual word embeddings.

    """

    def __init__(self, method: Literal['include_stop_words', 'exclude_stop_words'] = 'include_stop_words',
                 spacy_pipeline: Literal[
                     'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'] = 'en_core_web_sm',
                 batch_size: int = 256, checkpoint: bool = False, checkpoint_step: int = 1, enable_tqdm: bool = False):
        """
        Initializes the Lexical class with the specified method and spaCy pipeline. `[37]`_.

        .. _[37]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        method : {'include_stop_words', 'exclude_stop_words'}, default 'include_stop_words'
            The method to use for answer leakage analysis.
        spacy_pipeline : {'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'}, default 'en_core_web_sm'
            The spaCy pipeline to use for stop words.
        batch_size : int, default 256
            The batch size for processing.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        ValueError
            If the provided method name is not valid.

        Examples
        --------
        >>> from hinteval.evaluation.answer_leakage import Lexical
        >>>
        >>> lexical = Lexical(method='include_stop_words', spacy_pipeline='en_core_web_sm', batch_size=256, checkpoint=True, checkpoint_step=10, enable_tqdm=True)

        References
        ----------
        .. [37] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`ContextualEmbeddings` : Class for evaluating answer leakage of hints using contextual word embeddings.

        """

        if method not in ['include_stop_words', 'exclude_stop_words']:
            raise ValueError(
                f'Invalid method name: "{method}".\n'
                'Please choose one of the following valid methods:\n'
                '- include_stop_words: Includes common stop words (e.g., "the", "and", "in") in the analysis. This method is useful when you want to preserve these words for certain text processing tasks.\n'
                '- exclude_stop_words: Removes common stop words from the analysis. This method is useful for focusing on the more meaningful content of the text by eliminating less significant words.'
            )
        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'answer_leakage_lexical_{method}.pickle'
        self.batch_size = batch_size
        self._method = method
        self._spacy_pipeline = spacy_pipeline
        SpacyDownloader.download(spacy_pipeline)
        self._spacy_model = spacy.load(self._spacy_pipeline)
        self._stop_words = self._spacy_model.Defaults.stop_words

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the answer leakage of the hints of the given instances using the lexical comparison `[38]`_.

        .. _[38]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of answer leakage scores for each instance.

        Notes
        ----------
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint` of the `instances`, with names based on the model, such as "answer-leakage-lexical-include_stop_words-sm".

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.answer_leakage import Lexical
        >>>
        >>> lexical = Lexical(method='include_stop_words')
        >>> instance_1 = Instance(
        ...    question=Question('What is the capital of Austria?'),
        ...    answers=[Answer('Vienna')],
        ...    hints=[Hint('This city, once home to Mozart and Beethoven.'),
        ...           Hint('This city is called as Vienna.')])
        >>> instance_2 = Instance(
        ...    question=Question('Who was the president of USA in 2009?'),
        ...    answers=[Answer('Barack Obama')],
        ...    hints=[Hint('His lastname is Obama.'),
        ...           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
        >>> instances = [instance_1, instance_2]
        >>> results = lexical.evaluate(instances)
        >>> print(results)
        # [[0, 1], [1, 0]]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...           instance in instances
        ...           for hint in instance.hints for metric_key, metric_value in
        ...           hint.metrics.items()]
        >>> print(metrics)
        # ['answer-leakage-lexical-include_stop_words-sm: 0', 'answer-leakage-lexical-include_stop_words-sm: 1',
        #  'answer-leakage-lexical-include_stop_words-sm: 1', 'answer-leakage-lexical-include_stop_words-sm: 0']


        References
        ----------
        .. [38] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855


        See Also
        --------
        :class:`ContextualEmbeddings` : Class for evaluating answer leakage of hints using contextual word embeddings.

        """

        self._validate_input(instances)

        answers = []
        hints = []
        for instance in instances:
            answers.extend([instance.answers[0]] * len(instance.hints))
            hints.extend(instance.hints)
        answer_hint = list(zip(answers, hints))
        answer_hint_batches = [answer_hint[i:i + self.batch_size] for i in range(0, len(answer_hint), self.batch_size)]

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        answer_hint_stream = tqdm(answer_hint_batches, total=len(answer_hint_batches),
                                  desc=f'Evaluating answer leakage metric based on the lexical comparison{" without " if self._method == "exclude_stop_words" else " "}considering stop words') if self.enable_tqdm else answer_hint_batches

        for answers_hints in answer_hint_stream:
            _answers, _hints = zip(*answers_hints)
            _idx = next(item_counter)
            final_step = len(results) // self.batch_size
            if _idx <= final_step:
                continue

            hints_docs = self._spacy_model.pipe([hint.hint.lower() for hint in _hints], batch_size=self.batch_size)
            answers_docs = self._spacy_model.pipe([answer.answer.lower() for answer in _answers],
                                                  batch_size=self.batch_size)

            for answer, hint in zip(answers_docs, hints_docs):
                hint_lemma = [token.lemma_ for token in hint]
                answer_lemma = [token.lemma_ for token in answer]
                if self._method == 'exclude_stop_words':
                    filtered_answer = [w for w in answer_lemma if
                                       not w in self._stop_words and w not in string.punctuation and w != '\'s']
                else:
                    filtered_answer = [w for w in answer_lemma if w not in string.punctuation and w != '\'s']
                is_answer_leakage = False
                for a_lemma in filtered_answer:
                    if a_lemma in hint_lemma and not is_answer_leakage:
                        is_answer_leakage = True
                results.append(int(is_answer_leakage))

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    answer_hint_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [round(res, 3) for res in results]
        for idx, h in enumerate(hints):
            spacy_pipeline = re.search(r'web_(\w+)$', self._spacy_pipeline).group(1)
            metadata = {'spacy_pipeline': self._spacy_pipeline}
            h.metrics[f'answer-leakage-lexical-{self._method}-{spacy_pipeline}'] = Metric('answer-leakage',
                                                                                          results[idx],
                                                                                          metadata=metadata)
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([results.pop(0) for _ in range(num_of_hint)])
        return final_results


class ContextualEmbeddings(AnswerLeakage):
    """
    Class for evaluating answer leakage of hints using contextual word embeddings.

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
    :class:`Lexical` : Class for evaluating answer leakage of hints using lexical comparison.

    """

    def __init__(self, sbert_model: str = 'all-mpnet-base-v2',
                 method: Literal['include_stop_words', 'exclude_stop_words'] = 'include_stop_words',
                 spacy_pipeline: Literal[
                     'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'] = 'en_core_web_sm',
                 checkpoint: bool = False, checkpoint_step: int = 1, enable_tqdm: bool = False):
        """
        Initializes the ContextualEmbeddings class with the sentence bert model, specified method and spaCy pipeline.

        Parameters
        ----------
        sbert_model: str, default 'all-mpnet-base-v2'
            The sentence bert model to use to generate word embeddings.
        method : {'include_stop_words', 'exclude_stop_words'}, default 'include_stop_words'
            The method to use for answer leakage analysis.
        spacy_pipeline : {'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'}, default 'en_core_web_sm'
            The spaCy pipeline to use for stop words.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        ValueError
            If the provided method name is not valid.
        Exception
            If downloading of spaCy or sentence bert models fails.

        Examples
        --------
        >>> from hinteval.evaluation.answer_leakage import ContextualEmbeddings
        >>>
        >>> contextual = ContextualEmbeddings(sbert_model='paraphrase-multilingual-mpnet-base-v2', method='include_stop_words',
        ...                                  spacy_pipeline='en_core_web_sm', checkpoint=True,
        ...                                  checkpoint_step=250, enable_tqdm=True)

        See Also
        --------
        :class:`Lexical` : Class for evaluating answer leakage of hints using lexical comparison.

        """

        if method not in ['include_stop_words', 'exclude_stop_words']:
            raise ValueError(
                f'Invalid method name: "{method}".\n'
                'Please choose one of the following valid methods:\n'
                '- include_stop_words: Includes common stop words (e.g., "the", "and", "in") in the analysis. This method is useful when you want to preserve these words for certain text processing tasks.\n'
                '- exclude_stop_words: Removes common stop words from the analysis. This method is useful for focusing on the more meaningful content of the text by eliminating less significant words.'
            )
        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'answer_leakage_contextual_{method}.pickle'
        self._batch_size = 1
        self._model_name = sbert_model
        self._method = method
        self._spacy_pipeline = spacy_pipeline
        SpacyDownloader.download(spacy_pipeline)
        self._spacy_model = spacy.load(self._spacy_pipeline)
        self._stop_words = self._spacy_model.Defaults.stop_words
        self._model = SentenceTransformer(self._model_name)

    def _similarity(self, hint_words, answer):
        hints_embeddings = self._model.encode(hint_words)
        answer_embedding = self._model.encode(answer)

        similarities = self._model.similarity(hints_embeddings, answer_embedding)
        max_similarity = torch.max(similarities)
        return max_similarity.numpy().item()
        # avg_similarity = torch.mean(similarities)
        # return max_similarity.numpy().item(), avg_similarity.numpy().item()

    def evaluate(self, instances: List[Instance], **kwargs) -> List[List[float]]:
        """
        Evaluates the answer leakage of the hints of the given instances using the contextual word embeddings.

        Parameters
        ----------
        instances : List[Instance]
            List of instances to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[float]]
            List of answer leakage scores for each instance.

        Notes
        ----------
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Hint` of the `instances`, with names based on the model, such as "answer-leakage-contextual-include_stop_words-sm".

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Hint, Answer
        >>> from hinteval.evaluation.answer_leakage import ContextualEmbeddings
        >>>
        >>> contextual = ContextualEmbeddings(sbert_model='paraphrase-multilingual-mpnet-base-v2')
        >>> instance_1 = Instance(
        ...    question=Question('What is the capital of Austria?'),
        ...    answers=[Answer('Vienna')],
        ...    hints=[Hint('This city, once home to Mozart and Beethoven.'),
        ...           Hint('This city is called as Vienna.')])
        >>> instance_2 = Instance(
        ...    question=Question('Who was the president of USA in 2009?'),
        ...    answers=[Answer('Barack Obama')],
        ...    hints=[Hint('His lastname is Obama.'),
        ...           Hint('He was named the 2009 Nobel Peace Prize laureate.')])
        >>> instances = [instance_1, instance_2]
        >>> results = contextual.evaluate(instances)
        >>> print(results)
        # [[0.495, 1.0], [0.967, 0.332]]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for
        ...           instance in instances
        ...           for hint in instance.hints for metric_key, metric_value in
        ...           hint.metrics.items()]
        >>> print(metrics)
        # ['answer-leakage-lexical-include_stop_words-sm: 0.495', 'answer-leakage-lexical-include_stop_words-sm: 1.0',
        #  'answer-leakage-lexical-include_stop_words-sm: 0.967', 'answer-leakage-lexical-include_stop_words-sm: 0.332']


        See Also
        --------
        :class:`Lexical` : Class for evaluating answer leakage of hints using lexical comparison.

        """

        self._validate_input(instances)
        answers = []
        hints = []
        for instance in instances:
            answers.extend([instance.answers[0]] * len(instance.hints))
            hints.extend(instance.hints)
        answer_hint = list(zip(answers, hints))

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        answer_hint_stream = tqdm(answer_hint, total=len(answer_hint),
                                  desc=f'Evaluating answer leakage metric using {self._model_name}{" without " if self._method == "exclude_stop_words" else " "}considering stop words') if self.enable_tqdm else answer_hint

        for answer, hint in answer_hint_stream:
            _idx = next(item_counter)
            final_step = len(results) // self._batch_size
            if _idx <= final_step:
                continue
            hint_lemma = word_tokenize(hint.hint)
            answer_lemma = word_tokenize(answer.answer)
            if self._method == 'exclude_stop_words':
                filtered_answer = [w for w in answer_lemma if
                                   not w in self._stop_words and w not in string.punctuation and w != '\'s']
            else:
                filtered_answer = [w for w in answer_lemma if w not in string.punctuation and w != '\'s']
            answer = " ".join(filtered_answer)
            answer_leakage = self._similarity(hint_lemma, answer)
            results.append(answer_leakage)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    answer_hint_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [round(res, 3) for res in results]
        for idx, h in enumerate(hints):
            spacy_pipeline = re.search(r'web_(\w+)$', self._spacy_pipeline).group(1)
            metadata = {'spacy_pipeline': self._spacy_pipeline, 'model': self._model_name}
            h.metrics[f'answer-leakage-contextual-{self._method}-{spacy_pipeline}'] = Metric('answer-leakage',
                                                                                             results[idx],
                                                                                             metadata=metadata)
        final_results = []
        for instance in instances:
            num_of_hint = len(instance.hints)
            final_results.append([results.pop(0) for _ in range(num_of_hint)])
        return final_results
