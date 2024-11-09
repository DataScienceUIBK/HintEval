import os
import re
import itertools
import joblib
import spacy
import lftk
import torch
import numpy as np
import xgboost as xgb
from hinteval.cores.evaluation_core import Readability
from hinteval.cores.dataset_core import Question, Hint, Metric
from hinteval.utils.functions.download_manager import SpacyDownloader, ReadabilityMLDownloader, ReadabilityNNDownloader
from hinteval.utils.readability.local import ReadmeReadability as RR_LOCAL
from hinteval.utils.readability.api_based import ReadmeReadability as RR_API
from typing import Literal, List, Union
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from torch.utils.data import DataLoader
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class TraditionalIndexes(Readability):
    """
    Class for evaluating readability of :class:`Question` or :class:`Hint` using traditional readability indexes `[15]`_.

    .. _[15]: https://aclanthology.org/2023.bea-1.1/

    Attributes
    ----------
    spacy_pipeline : str
        The spaCy pipeline to use for tokenization.
    checkpoint : bool
        Whether checkpointing is enabled.
    checkpoint_step : int
        Step interval for checkpointing.
    enable_tqdm : bool
        Whether the tqdm progress bar is enabled.

    References
    ----------
    .. [15] Bruce W. Lee and Jason Lee. 2023. LFTK: Handcrafted Features in Computational Linguistics. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 1–19, Toronto, Canada. Association for Computational Linguistics.

    See Also
    --------
    :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
    :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.
    :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

    """

    def __init__(self, method: Literal[
        'flesch_kincaid_reading_ease', 'gunning_fog_index', 'smog_index', 'coleman_liau_index', 'automated_readability_index'] = 'flesch_kincaid_reading_ease',
                 spacy_pipeline: Literal[
                     'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'] = 'en_core_web_sm',
                 checkpoint: bool = False, checkpoint_step: int = 1, enable_tqdm=False):
        """
        Initializes the TraditionalIndexes class with the specified readability method and spaCy pipeline `[16]`_.

        .. _[16]: https://aclanthology.org/2023.bea-1.1/

        Parameters
        ----------
        method : {'flesch_kincaid_reading_ease', 'gunning_fog_index', 'smog_index', 'coleman_liau_index', 'automated_readability_index'}, default 'flesch_kincaid_reading_ease'
            The readability method to use.
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
        ValueError
            If the provided method name is not valid.
        Exception
            If downloading of spaCy models fails.

        Examples
        --------
        >>> from hinteval.evaluation.readability import TraditionalIndexes
        >>>
        >>> traditional_indexes = TraditionalIndexes(method='flesch_kincaid_reading_ease', spacy_pipeline='en_core_web_sm',
        ...                                          checkpoint=True, checkpoint_step=10, enable_tqdm=True)


        References
        ----------
        .. [16] Bruce W. Lee and Jason Lee. 2023. LFTK: Handcrafted Features in Computational Linguistics. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 1–19, Toronto, Canada. Association for Computational Linguistics.


        See Also
        --------
        :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
        :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        if method not in ['flesch_kincaid_reading_ease', 'gunning_fog_index', 'smog_index', 'coleman_liau_index',
                          'automated_readability_index']:
            raise ValueError(
                f'Invalid method name: "{method}".\n'
                'Please choose one of the following valid methods:\n'
                '- flesch_kincaid_reading_ease: Measures text readability based on sentence length and syllable count. Higher scores indicate easier readability.\n'
                '- gunning_fog_index: Estimates the years of formal education needed to understand the text on the first reading. It considers sentence length and complex word count.\n'
                '- smog_index: Estimates the years of education required to understand a piece of writing. It focuses on the number of polysyllabic words in the text.\n'
                '- coleman_liau_index: Calculates readability based on characters per word and sentences per text. It is a formula that gives a grade level for the text.\n'
                '- automated_readability_index: Provides an estimate of the readability based on character count, word count, and sentence count. It also yields a grade level for the text.'
            )
        self._file_name = f'readability_{method}.pickle'
        self._batch_size = 1
        self._method = method
        self._method_dict = {'flesch_kincaid_reading_ease': 'fkre', 'gunning_fog_index': 'fogi', 'smog_index': 'smog',
                             'coleman_liau_index': 'cole', 'automated_readability_index': 'auto'}
        self._spacy_pipeline = spacy_pipeline
        SpacyDownloader.download(spacy_pipeline)
        self._spacy_model = spacy.load(spacy_pipeline)

    def evaluate(self, sentences: List[Union[Question, Hint]], **kwargs) -> List[float]:
        """
        Evaluates the readability of the given :class:`Question` or :class:`Hint` using the specified method `[17]`_.

        .. _[17]: https://aclanthology.org/2023.bea-1.1/

        Parameters
        ----------
        sentences : List[Union[Question, Hint]]
            List of sentences to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[float]
            List of readability scores for each sentence.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Question` or :class:`Hint`, with names based on the method, such as "readability-flesch_kincaid_reading_ease-sm".

        Examples
        --------
        >>> from hinteval.cores import Question, Hint
        >>> from hinteval.evaluation.readability import TraditionalIndexes
        >>>
        >>> traditional_indexes = TraditionalIndexes(method='flesch_kincaid_reading_ease')
        >>> sentence_1 = Question('What is the capital of Austria?')
        >>> sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
        >>> sentences = [sentence_1, sentence_2]
        >>> results = traditional_indexes.evaluate(sentences)
        >>> print(results)
        # [87.945, 69.994]
        >>> metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
        ...        sent.metrics.items()]
        >>> print(metrics)
        # ['readability-flesch_kincaid_reading_ease-sm: 87.945', 'readability-flesch_kincaid_reading_ease-sm: 69.994']

        References
        ----------
        .. [17] Bruce W. Lee and Jason Lee. 2023. LFTK: Handcrafted Features in Computational Linguistics. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 1–19, Toronto, Canada. Association for Computational Linguistics.

        See Also
        --------
        :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
        :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

        """

        self._validate_input(sentences)

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        sentence_stream = tqdm(sentences, total=len(sentences),
                               desc=f'Evaluating readability metric using {self._method}') if self.enable_tqdm else sentences
        for _sentence in sentence_stream:
            _idx = next(item_counter)
            final_step = len(results) // self._batch_size
            if _idx <= final_step:
                continue

            if isinstance(_sentence, Question):
                doc = self._spacy_model(_sentence.question)
            else:
                doc = self._spacy_model(_sentence.hint)
            LFTK = lftk.Extractor(docs=doc)
            LFTK.customize(stop_words=True, punctuations=False, round_decimal=3)
            extracted_feature = LFTK.extract(features=[self._method_dict[self._method]])
            results.append(extracted_feature[self._method_dict[self._method]])

            if (_idx % self.checkpoint_step == 0 or _idx == len(sentence_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        results = [round(res, 3) for res in results]
        for idx, s in enumerate(sentences):
            spacy_pipeline = re.search(r'web_(\w+)$', self._spacy_pipeline).group(1)
            metadata = {'spacy_pipeline': self._spacy_pipeline}
            s.metrics[f'readability-{self._method}-{spacy_pipeline}'] = Metric('readability', results[idx],
                                                                               metadata=metadata)
        return results


class MachineLearningBased(Readability):
    """
    Class for evaluating readability of :class:`Question` or :class:`Hint` using machine learning methods such as XGBoost and Random-Forest models `[18]`_.

    .. _[18]: https://aclanthology.org/2023.bea-1.37/

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
    .. [18] Fengkai Liu and John Lee. 2023. Hybrid Models for Sentence Readability Assessment. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 448–454, Toronto, Canada. Association for Computational Linguistics.

    See Also
    --------
    :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
    :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.
    :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

    """

    def __init__(self, method: Literal['xgboost', 'random_forest'] = 'xgboost',
                 spacy_pipeline: Literal[
                     'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'] = 'en_core_web_sm',
                 checkpoint: bool = False, checkpoint_step: int = 1, force_download=False, enable_tqdm=False):
        """
        Initializes the MachineLearningBased class with the specified machine learning method and spaCy pipeline `[19]`_.

        .. _[19]: https://aclanthology.org/2023.bea-1.37/

        Parameters
        ----------
        method : {'xgboost', 'random_forest'}, default 'xgboost'
            The machine learning method to use.
        spacy_pipeline : {'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'}, default 'en_core_web_sm'
            The spaCy pipeline to use for tokenization.
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
            If the provided method name is not valid.
        Exception
            If downloading of spaCy models or machine learning models fails.

        Examples
        --------
        >>> from hinteval.evaluation.readability import MachineLearningBased
        >>>
        >>> machine_learning = MachineLearningBased(method='xgboost', spacy_pipeline='en_core_web_sm',
        ...                                          checkpoint=True, checkpoint_step=10, enable_tqdm=True)

        References
        ----------
        .. [19] Fengkai Liu and John Lee. 2023. Hybrid Models for Sentence Readability Assessment. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 448–454, Toronto, Canada. Association for Computational Linguistics.

        See Also
        --------
        :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
        :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'readability_{method}.pickle'
        self._batch_size = 1
        self._method = method
        self._spacy_pipeline = spacy_pipeline
        SpacyDownloader.download(spacy_pipeline)
        self._spacy_model = spacy.load(spacy_pipeline)
        ReadabilityMLDownloader.download(method, force_download)
        if method == 'xgboost':
            self._ml = xgb.Booster()
            self._ml.load_model(
                os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'readability-ml', f'{method}.model'))
        else:
            self._ml = joblib.load(os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'readability-ml', f'{method}.model'))
        self._selected_features = ['t_word', 't_stopword', 't_punct', 't_syll', 't_syll2', 't_syll3', 't_uword',
                                   't_char', 'a_word_ps', 'a_char_ps', 'a_syll_ps', 'a_stopword_ps', 't_kup', 't_bry',
                                   't_subtlex_us_zipf', 'a_kup_ps', 'a_bry_ps', 'a_subtlex_us_zipf_ps', 't_n_ent',
                                   'a_n_ent_ps', 'uber_ttr', 'uber_ttr_no_lem', 'n_adj', 'n_adp', 'n_adv', 'n_aux',
                                   'n_det', 'n_noun', 'n_num', 'n_pron', 'n_propn', 'n_punct', 'n_verb', 'n_space',
                                   'n_uadj', 'n_uadp', 'n_uadv', 'n_unoun', 'n_unum', 'n_upron', 'n_upropn', 'n_uverb',
                                   'a_adj_ps', 'a_adp_ps', 'a_adv_ps', 'a_aux_ps', 'a_det_ps', 'a_noun_ps', 'a_num_ps',
                                   'a_pron_ps', 'a_propn_ps', 'a_punct_ps', 'a_verb_ps', 'a_space_ps', 'fkre', 'fkgl',
                                   'fogi', 'smog', 'cole', 'auto']
        self._result_dict = {0: 'beginner', 1: 'intermediate', 2: 'advanced'}

    def evaluate(self, sentences: List[Union[Question, Hint]], **kwargs) -> List[float]:
        """
        Evaluates the readability of the given :class:`Question` or :class:`Hint` using the specified machine learning method `[20]`_.

        .. _[20]: https://aclanthology.org/2023.bea-1.37/

        Parameters
        ----------
        sentences : List[Union[Question, Hint]]
            List of sentences to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[float]
            List of readability scores for each sentence.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Question` or :class:`Hint`, with names based on the method, such as "readability-ml-xgboost-sm".

        Examples
        --------
        >>> from hinteval.cores import Question, Hint
        >>> from hinteval.evaluation.readability import MachineLearningBased
        >>>
        >>> machine_learning = MachineLearningBased(method='xgboost')
        >>> sentence_1 = Question('What is the capital of Austria?')
        >>> sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
        >>> sentences = [sentence_1, sentence_2]
        >>> results = machine_learning.evaluate(sentences)
        >>> print(results)
        # [0, 0]
        >>> classes = [sent.metrics['readability-ml-xgboost-sm'].metadata['description'] for sent in sentences]
        >>> print(classes)
        # ['beginner', 'beginner']
        >>> metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
        ...        sent.metrics.items()]
        >>> print(metrics)
        # ['readability-ml-xgboost-sm: 0', 'readability-ml-xgboost-sm: 0']

        References
        ----------
        .. [20] Fengkai Liu and John Lee. 2023. Hybrid Models for Sentence Readability Assessment. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 448–454, Toronto, Canada. Association for Computational Linguistics.

        See Also
        --------
        :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
        :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.
        :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

        """

        self._validate_input(sentences)
        _sentences = []

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        sentence_stream = tqdm(sentences, total=len(sentences),
                               desc=f'Evaluating readability metric using {self._method}') if self.enable_tqdm else sentences
        for _sentence in sentence_stream:
            _idx = next(item_counter)
            final_step = len(results) // self._batch_size
            if _idx <= final_step:
                continue

            if isinstance(_sentence, Question):
                doc = self._spacy_model(_sentence.question)
            else:
                doc = self._spacy_model(_sentence.hint)
            LFTK = lftk.Extractor(docs=doc)
            LFTK.customize(stop_words=True, punctuations=False, round_decimal=3)
            extracted_features = [list(LFTK.extract(self._selected_features).values())]
            if self._method == 'xgboost':
                extracted_features = xgb.DMatrix(extracted_features)
            results.append(self._ml.predict(extracted_features).astype(int).tolist()[0])

            if (_idx % self.checkpoint_step == 0 or _idx == len(sentence_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        for idx, s in enumerate(sentences):
            spacy_pipeline = re.search(r'web_(\w+)$', self._spacy_pipeline).group(1)
            s.metrics[f'readability-ml-{self._method}-{spacy_pipeline}'] = Metric('readability', results[idx])
            s.metrics[f'readability-ml-{self._method}-{spacy_pipeline}'].metadata[f'description'] = self._result_dict[
                results[idx]]
            s.metrics[f'readability-ml-{self._method}-{spacy_pipeline}'].metadata[
                f'spacy_pipeline'] = self._spacy_pipeline
        return results


class NeuralNetworkBased(Readability):
    """
    Class for evaluating readability of :class:`Question` or :class:`Hint` using neural network models such as BERT and RoBERTa `[21]`_.

    .. _[21]: https://aclanthology.org/2023.bea-1.37/

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
    .. [21] Fengkai Liu and John Lee. 2023. Hybrid Models for Sentence Readability Assessment. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 448–454, Toronto, Canada. Association for Computational Linguistics.

    See Also
    --------
    :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
    :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
    :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

    """

    def __init__(self, model_name: Literal['bert-base', 'roberta-large'] = 'bert-base', batch_size: int = 256,
                 checkpoint: bool = False, checkpoint_step: int = 1, force_download=False, enable_tqdm=False):
        """
        Initializes the NeuralNetworkBased class with the specified neural network model `[22]`_.

        .. _[22]: https://aclanthology.org/2023.bea-1.37/

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
        >>> from hinteval.evaluation.readability import NeuralNetworkBased
        >>>
        >>> neural_network = NeuralNetworkBased(model_name='bert-base', batch_size=64, checkpoint=True, checkpoint_step=10, enable_tqdm=True)

        References
        ----------
        .. [22] Fengkai Liu and John Lee. 2023. Hybrid Models for Sentence Readability Assessment. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 448–454, Toronto, Canada. Association for Computational Linguistics.

        See Also
        --------
        :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
        :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
        :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'readability_{model_name}.pickle'
        self._model_name = model_name
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        ReadabilityNNDownloader.download(model_name, force_download)
        model_dir = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'readability-nn', self._model_name)
        self._config = AutoConfig.from_pretrained(model_dir, num_labels=3)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=self._config)
        _tokenizer_type = 'bert-base-uncased' if model_name == 'bert-base' else 'roberta-large'
        self._tokenizer = AutoTokenizer.from_pretrained(_tokenizer_type, do_lower_case=True)
        self._model.to(self._device)
        self._result_dict = {0: 'beginner', 1: 'intermediate', 2: 'advanced'}

    @staticmethod
    def _softmax(logits):
        exp_logits = np.exp(logits)
        logits_sum = np.sum(exp_logits, axis=1)
        return exp_logits / logits_sum[:, np.newaxis]

    def _tokenize_function(self, sentence):
        return self._tokenizer(sentence['sent'], padding='max_length', max_length=128, truncation=True)

    def evaluate(self, sentences: List[Union[Question, Hint]], **kwargs) -> List[float]:
        """
        Evaluates the readability of the given :class:`Question` or :class:`Hint` using the specified neural network model `[23]`_.

        .. _[23]: https://aclanthology.org/2023.bea-1.37/

        Parameters
        ----------
        sentences : List[Union[Question, Hint]]
            List of sentences to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[float]
            List of readability scores for each sentence.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Question` or :class:`Hint`, with names based on the model, such as "readability-nn-bert-base".

        Examples
        --------
        >>> from hinteval.cores import Question, Hint
        >>> from hinteval.evaluation.readability import NeuralNetworkBased
        >>>
        >>> neural_network = NeuralNetworkBased(model_name='bert-base')
        >>> sentence_1 = Question('What is the capital of Austria?')
        >>> sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
        >>> sentences = [sentence_1, sentence_2]
        >>> results = neural_network.evaluate(sentences)
        >>> print(results)
        # [0, 0]
        >>> classes = [sent.metrics['readability-nn-bert-base'].metadata['description'] for sent in sentences]
        >>> print(classes)
        # ['beginner', 'beginner']
        >>> metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
        ...        sent.metrics.items()]
        >>> print(metrics)
        # ['readability-nn-bert-base: 0', 'readability-nn-bert-base: 0']

        References
        ----------
        .. [23] Fengkai Liu and John Lee. 2023. Hybrid Models for Sentence Readability Assessment. In Proceedings of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), pages 448–454, Toronto, Canada. Association for Computational Linguistics.

        See Also
        --------
        :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
        :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
        :class:`LlmBased` : Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models.

        """

        self._validate_input(sentences)
        _sentences = []
        for _sentence in sentences:
            if isinstance(_sentence, Question):
                _sentences.append({'sent': _sentence.question})
            else:
                _sentences.append({'sent': _sentence.hint})
        _dataset = Dataset.from_list(_sentences)
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
                                  desc=f'Evaluating readability metric using {self._model_name}') if self.enable_tqdm else data_loader

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

        for idx, s in enumerate(sentences):
            s.metrics[f'readability-nn-{self._model_name}'] = Metric('readability', results[idx])
            s.metrics[f'readability-nn-{self._model_name}'].metadata[f'description'] = \
                self._result_dict[results[idx]]
        return results


class LlmBased(Readability):
    """
    Class for evaluating readability of :class:`Question` or :class:`Hint` using large language models `[24]`_.

    .. _[24]: https://arxiv.org/abs/2305.14463

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
    .. [24] Naous, Tarek, et al. "ReadMe++: Benchmarking Multilingual Language Models for Multi-Domain Readability Assessment." arXiv preprint arXiv:2305.14463 (2024).


    See Also
    --------
    :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
    :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
    :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.

    """

    def __init__(self, model_name: str,
                 api_key: str = None,
                 base_url: str = 'https://api.together.xyz/v1',
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 max_tokens: int = 512,
                 batch_size: int = 10,
                 checkpoint: bool = False,
                 checkpoint_step: int = 1,
                 enable_tqdm=False):
        """
        Initializes the LlmBased class with the specified large language model `[25]`_.

        .. _[25]: https://arxiv.org/abs/2305.14463

        Parameters
        ----------
        model_name : str
            The large language model to use for evaluation readability.
        api_key : str, optional
            Specifies the API key required for accessing and interacting with the model.
        base_url : str, default 'https://api.together.xyz/v1'
            Specifies the base URL for the API endpoints. This URL is used to construct full API request URLs.
        temperature : float, default 0.7
            Sampling temperature to control the diversity of generated content.
        top_p : float, default 1.0
            Nucleus sampling cutoff to control the randomness of generation.
        max_tokens : int, default 1024
            Maximum number of tokens to generate for each hint.
        batch_size : int, default 10
            Number of instances processed in one batch during generation.
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
        -----
        If `api_key` is None, the evaluator will attempt to download the model and run it locally.


        Examples
        --------
        >>> from hinteval.evaluation.readability import LlmBased
        >>>
        >>> readability = LlmBased(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', api_key='your_api_key',
        ...                  base_url='base_url', temperature=0.7, top_p=1.0, max_tokens=1024, batch_size=10,
        ...                 checkpoint=True, checkpoint_step=5, enable_tqdm=True)

        References
        ----------
        .. [25] Naous, Tarek, et al. "ReadMe++: Benchmarking Multilingual Language Models for Multi-Domain Readability Assessment." arXiv preprint arXiv:2305.14463 (2024).

        See Also
        --------
        :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
        :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
        :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.

        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'readability_llm_{model_name.replace("/", "_")}.pickle'
        self._api_key = api_key
        self._base_url = base_url
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self.batch_size = batch_size
        self._model_name = model_name
        self._result_dict = {0: 'beginner', 1: 'intermediate', 2: 'advanced'}

        if self._api_key is None:
            self._evaluator = RR_LOCAL(model_name=model_name, temperature=self._temperature, top_p=self._top_p,
                                       max_tokens=self._max_tokens)
        else:
            self._evaluator = RR_API(model_name=model_name, api_key=self._api_key, base_url=base_url,
                                     temperature=self._temperature, top_p=self._top_p, max_tokens=self._max_tokens)

    def evaluate(self, sentences: List[Union[Question, Hint]], **kwargs) -> List[float]:
        """
        Evaluates the readability of the question and hints of the given instances using large language models `[26]`_.

        .. _[26]: https://arxiv.org/abs/2305.14463

        Parameters
        ----------
        sentences : List[Union[Question, Hint]]
            List of sentences to evaluate.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[float]
            List of readability scores for each sentence.

        Notes
        -----
        This function stores the scores as :class:`Metric` objects within the `metrics` attribute of the :class:`Question` or :class:`Hint`, with names based on the model, such as "readability-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo".

        Examples
        --------
        >>> from hinteval.cores import Question, Hint
        >>> from hinteval.evaluation.readability import LlmBased
        >>>
        >>> llm = LlmBased(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        ...          api_key='your_api_key', base_url='base_url', batch_size=2,
        ...          enable_tqdm=True)
        >>> sentence_1 = Question('What is the capital of Austria?')
        >>> sentence_2 = Hint('This city, once home to Mozart and Beethoven, is the capital of Austria.')
        >>> sentences = [sentence_1, sentence_2]
        >>> results = llm.evaluate(sentences)
        >>> print(results)
        # [0, 0]
        >>> classes = [sent.metrics['readability-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo'].metadata['description'] for sent in sentences]
        >>> print(classes)
        # ['beginner', 'beginner']
        >>> metrics = [f'{metric_key}: {metric_value.value}' for sent in sentences for metric_key, metric_value in
        ...           sent.metrics.items()]
        >>> print(metrics)
        # ['readability-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 0', 'readability-llm-meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo: 0']


        References
        ----------
        .. [26] Naous, Tarek, et al. "ReadMe++: Benchmarking Multilingual Language Models for Multi-Domain Readability Assessment." arXiv preprint arXiv:2305.14463 (2024).

        See Also
        --------
        :class:`TraditionalIndexes` : Class for evaluating readability of Question or Hint using traditional readability indexes.
        :class:`MachineLearningBased` : Class for evaluating readability of Question or Hint using machine learning such as XGBoost and Random-Forest models.
        :class:`NeuralNetworkBased` : Class for evaluating readability of Question or Hint using contextual embeddings such as BERT and RoBERTa models.

        """

        self._validate_input(sentences)
        _sentences = []
        for _sentence in sentences:
            if isinstance(_sentence, Question):
                _sentences.append(_sentence.question)
            else:
                _sentences.append(_sentence.hint)
        _sentences = [_sentences[i:i + self.batch_size] for i in range(0, len(_sentences), self.batch_size)]

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        sentences_stream = tqdm(_sentences, total=len(_sentences),
                                desc=f'Evaluating readability metric using {self._model_name.replace("/", "_")}') if self.enable_tqdm else _sentences
        for sentence in sentences_stream:
            _idx = next(item_counter)
            final_step = len(results) // self.batch_size
            if _idx <= final_step:
                continue
            scores = self._evaluator.compute_readability(sentence)
            results.extend(scores)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    sentences_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        for idx, s in enumerate(sentences):
            s.metrics[f'readability-llm-{self._model_name.replace("/", "_")}'] = Metric('readability', results[idx])
            s.metrics[f'readability-llm-{self._model_name.replace("/", "_")}'].metadata[f'description'] = \
                self._result_dict[results[idx]]
            s.metrics[f'readability-llm-{self._model_name.replace("/", "_")}'].metadata[
                'temperature'] = self._temperature
            s.metrics[f'readability-llm-{self._model_name.replace("/", "_")}'].metadata['top_p'] = self._top_p
            s.metrics[f'readability-llm-{self._model_name.replace("/", "_")}'].metadata['max_tokens'] = self._max_tokens
        return results
