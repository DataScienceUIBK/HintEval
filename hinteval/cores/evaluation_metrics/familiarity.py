import os
import re
import itertools
import spacy
import json
from hinteval.cores.evaluation_core import Familiarity
from hinteval.cores.dataset_core import Question, Hint, Answer, Metric
from hinteval.utils.familiarity.popularity import Popularity
from hinteval.utils.familiarity.metrics import Metrics
from hinteval.utils.functions.download_manager import SpacyDownloader, FamiliarityFrequencyDownloader
from typing import List, Union, Literal
from tqdm import tqdm
from tok import word_tokenize


class WordFrequency(Familiarity):
    """
    Class for evaluating familiarity of :class:`Question`, :class:`Hint`, or :class:`Answer` based on word frequency analysis on Common Crawl.

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
    :class:`Wikipedia` : Class for evaluating familiarity of Question, Hint, or Answer using number of views of corresponding wikipedia page.

    """

    def __init__(self, method: Literal['include_stop_words', 'exclude_stop_words'] = 'include_stop_words',
                 spacy_pipeline: Literal[
                     'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'] = 'en_core_web_sm',
                 checkpoint: bool = False, checkpoint_step: int = 1, force_download=False, enable_tqdm=False):
        """
        Initializes the WordFrequency class with the specified method and spaCy pipeline.

        Parameters
        ----------
        method : {'include_stop_words', 'exclude_stop_words'}, default 'include_stop_words'
            The method to use for word frequency analysis.
        spacy_pipeline : {'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'}, default 'en_core_web_sm'
            The spaCy pipeline to use for tokenization.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        force_download : bool, default False
            Whether to force download of necessary resources.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        ValueError
            If the provided method name is not valid.
        Exception
            If downloading of resources fails.

        Examples
        --------
        >>> from hinteval.evaluation.familiarity import WordFrequency
        >>>
        >>> word_frequency = WordFrequency(method='include_stop_words', spacy_pipeline='en_core_web_sm', checkpoint=True, checkpoint_step=10, enable_tqdm=True)

        See Also
        --------
        :class:`Wikipedia` : Class for evaluating familiarity of Question, Hint, or Answer using number of views of corresponding wikipedia page.
        """

        if method not in ['include_stop_words', 'exclude_stop_words']:
            raise ValueError(
                f'Invalid method name: "{method}".\n'
                'Please choose one of the following valid methods:\n'
                '- include_stop_words: Includes common stop words (e.g., "the", "and", "in") in the analysis. This method is useful when you want to preserve these words for certain text processing tasks.\n'
                '- exclude_stop_words: Removes common stop words from the analysis. This method is useful for focusing on the more meaningful content of the text by eliminating less significant words.'
            )
        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'familiarity_{method}.pickle'
        self._batch_size = 1
        self._method = method
        FamiliarityFrequencyDownloader.download(force_download)
        with open(os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'familiarity-freq', 'word_frequency_normalized.json'),
                  mode='r',
                  encoding='utf-8') as f:
            self._word_frequency = json.load(f)
        self._spacy_pipeline = spacy_pipeline
        SpacyDownloader.download(spacy_pipeline)
        self._stop_words = spacy.load(self._spacy_pipeline).Defaults.stop_words

    def evaluate(self, sentences: List[Union[Question, Hint, Answer]], **kwargs) -> List[float]:
        """
        Evaluates the familiarity of the given :class:`Question`, :class:`Hint`, or :class:`Answer` using word frequency analysis on Common Crawl.

        Parameters
        ----------
        sentences : List[Union[Question, Hint, Answer]]
            List of sentences to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[float]
            List of familiarity scores for each sentence.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the `Question`, `Hint`, or `Answer`, with names based on the method, such as "familiarity-freq-include_stop_words-sm".

        Examples
        --------
        >>> from hinteval.cores import Question, Hint
        >>> from hinteval.evaluation.familiarity import WordFrequency
        >>>
        >>> word_frequency = WordFrequency(method='include_stop_words')
        >>> sentence_1 = Question('What is the capital of Austria?')
        >>> sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
        >>> sentences = [sentence_1, sentence_2]
        >>> results = word_frequency.evaluate(sentences)
        >>> print(results)
        # [1.0, 1.0]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
        ...    sent.metrics.items()]
        >>> print(metrics)
        # ['familiarity-freq-include_stop_words-sm: 1.0', 'familiarity-freq-include_stop_words-sm: 1.0']

        See Also
        --------
        :class:`Wikipedia` : Class for evaluating familiarity of Question, Hint, or Answer using number of views of corresponding wikipedia page.
        """

        self._validate_input(sentences)

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        sentence_stream = tqdm(sentences, total=len(sentences),
                               desc=f'Evaluating familiarity metric based on the word frequency{" without " if self._method == "exclude_stop_words" else " "}considering stop words') if self.enable_tqdm else sentences
        for _sentence in sentence_stream:
            _idx = next(item_counter)
            final_step = len(results) // self._batch_size
            if _idx <= final_step:
                continue

            if isinstance(_sentence, Question):
                doc = _sentence.question
            elif isinstance(_sentence, Hint):
                doc = _sentence.hint
            else:
                doc = _sentence.answer
            sent = doc.lower().strip()
            tokens = word_tokenize(sent)
            sent_weights = []
            for token in tokens:
                if self._method == 'include_stop_words' and token in self._stop_words:
                    continue
                weight = 0
                if token in self._word_frequency:
                    weight = self._word_frequency[token]
                sent_weights.append(weight)
            if len(sent_weights) > 0:
                sent_weight = sum(sent_weights) / len(sent_weights)
            else:
                sent_weight = 0.0
            results.append(sent_weight)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    sentence_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [round(res, 3) for res in results]
        for idx, s in enumerate(sentences):
            spacy_pipeline = re.search(r'web_(\w+)$', self._spacy_pipeline).group(1)
            metadata = {'spacy_pipeline': self._spacy_pipeline}
            s.metrics[f'familiarity-freq-{self._method}-{spacy_pipeline}'] = Metric('familiarity', results[idx],
                                                                                    metadata=metadata)
        return results


class Wikipedia(Familiarity):
    """
    Class for evaluating familiarity of :class:`Question`, :class:`Hint`, or :class:`Answer` using the number of views of corresponding Wikipedia pages `[33]`_.

    .. _[33]: https://dl.acm.org/doi/10.1145/3626772.3657855

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
    .. [33] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

    See Also
    --------
    :class:`WordFrequency` : Class for evaluating familiarity of Question, Hint, or Answer based on word frequency analysis on Common Crawl.

    """

    def __init__(self, spacy_pipeline: Literal[
        'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'] = 'en_core_web_sm',
                 checkpoint: bool = False, checkpoint_step: int = 1, enable_tqdm=False):
        """
        Initializes the Wikipedia class with the specified spaCy pipeline `[34]`_.

        .. _[34]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        spacy_pipeline : {'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'}, default 'en_core_web_sm'
            The spaCy pipeline to use for tokenization.
        checkpoint : bool, default False
            Whether to enable checkpointing. If True, the environment variable `HINTEVAL_CHECKPOINT_DIR` must be set.
        checkpoint_step : int, default 1
            Step interval for checkpointing. Note that each step corresponds to one batch.
        enable_tqdm : bool, default False
            Whether to enable tqdm progress bar.

        Raises
        ------
        Exception
            If downloading of spaCy models fails.

        Examples
        --------
        >>> from hinteval.evaluation.familiarity import Wikipedia
        >>>
        >>> wikipedia = Wikipedia(spacy_pipeline='en_core_web_sm', checkpoint=True, checkpoint_step=10, enable_tqdm=True)

        References
        ----------
        .. [34] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`WordFrequency` : Class for evaluating familiarity of Question, Hint, or Answer based on word frequency analysis on Common Crawl.
        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'familiarity_wikipedia.pickle'
        self._batch_size = 1
        self._spacy_pipeline = spacy_pipeline
        SpacyDownloader.download(spacy_pipeline)
        self._spacy_model = spacy.load(self._spacy_pipeline)
        self._popularity = Popularity(spacy_pipeline)
        self._metrics = Metrics()

    def evaluate(self, sentences: List[Union[Question, Hint, Answer]], **kwargs) -> List[float]:
        """
        Evaluates the familiarity of the given :class:`Question`, :class:`Hint`, or :class:`Answer` using the number of views of corresponding Wikipedia pages `[35]`_.

        .. _[35]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        sentences : List[Union[Question, Hint, Answer]]
            List of sentences to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[float]
            List of familiarity scores for each sentence.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the `Question`, `Hint`, or `Answer`, with names based on the method, such as "familiarity-wikipedia-sm".

        This function also stores number of views for each entity as :class:`Entity` objects within the `entities` attribute.

        Examples
        --------
        >>> from hinteval.cores import Question, Hint
        >>> from hinteval.evaluation.familiarity import Wikipedia
        >>>
        >>> wikipedia = Wikipedia(spacy_pipeline='en_core_web_trf')
        >>> sentence_1 = Question('What is the capital of Austria?')
        >>> sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
        >>> sentences = [sentence_1, sentence_2]
        >>> results = wikipedia.evaluate(sentences)
        >>> print(results)
        # [1.0, 1.0]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
        ...    sent.metrics.items()]
        >>> print(metrics)
        # ['familiarity-wikipedia-trf: 1.0', 'familiarity-wikipedia-trf: 1.0']
        >>> entities = [f'{entity.entity}: {entity.metadata["wiki_views_per_month"]}' for sent in sentences for entity in
        ...    sent.entities]
        >>> print(entities)
        # ['austria: 248144', 'mozart: 233219', 'beethoven: 224128', 'austria: 248144']


        References
        ----------
        .. [35] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`WordFrequency` : Class for evaluating familiarity of Question, Hint, or Answer based on word frequency analysis on Common Crawl.
        """

        self._validate_input(sentences)

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        sentence_stream = tqdm(sentences, total=len(sentences),
                               desc=f'Evaluating familiarity metric using Wikipedia') if self.enable_tqdm else sentences
        for _sentence in sentence_stream:
            _idx = next(item_counter)
            final_step = len(results) // self._batch_size
            if _idx <= final_step:
                continue

            if isinstance(_sentence, Question):
                doc = _sentence.question
                is_word = False
            elif isinstance(_sentence, Hint):
                doc = _sentence.hint
                is_word = False
            else:
                doc = _sentence.answer
                is_word = True
            sent = doc.lower().strip()
            sent_entities, sent_pops_normalized = self._popularity.popularity(sent, is_word)
            familiarity = self._metrics.compute_metrics(sent_pops_normalized)
            results.append((sent_entities, familiarity))

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    sentence_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        for idx, s in enumerate(sentences):
            spacy_pipeline = re.search(r'web_(\w+)$', self._spacy_pipeline).group(1)
            metadata = {'spacy_pipeline': self._spacy_pipeline}
            s.metrics[f'familiarity-wikipedia-{spacy_pipeline}'] = Metric('familiarity', round(results[idx][1], 3),
                                                                          metadata=metadata)
            for entity in results[idx][0]:
                if entity not in s.entities:
                    s.entities.append(entity)
        return [res[1] for res in results]
