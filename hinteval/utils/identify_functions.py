from typing import Union, List, Literal
from hinteval.cores.dataset_core import Question, Answer, Hint, Instance
from hinteval.utils.functions.ent_spacy import EntitySpacy
from hinteval.utils.functions.question_classification import QC


def identify_entities(texts: List[Union[Instance, Question, Answer, Hint]], batch_size,
                      spacy_pipeline: Literal['en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'],
                      enable_tqdm):
    """
    Identifies entities in the given texts using spaCy `[1]`_.

    .. _[1]: https://spacy.io/

    Parameters
    ----------
    texts : list of Union[Instance, Question, Answer, Hint]
        A list of objects (instances of :class:`Instance`, :class:`Question`, :class:`Answer`, or :class:`Hint`) to perform entity recognition on.
    batch_size : int
        The batch size for processing texts.
    spacy_pipeline : {'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'}
        The spaCy pipeline to use for entity recognition.
    enable_tqdm : bool
        Whether to enable tqdm progress bar.

    Raises
    ------
    ValueError
        If any item in texts is not an instance of the supported classes.

    Examples
    --------
    >>> from hinteval.cores import Instance
    >>> from hinteval.utils.identify_functions import identify_entities
    >>>
    >>> instance = Instance.from_strings("What is the capital of Austria?", ["Vienna"], ["This city is famous for its music and culture."])
    >>> identify_entities([instance], batch_size=64, spacy_pipeline='en_core_web_sm', enable_tqdm=True)
    >>> print(instance.question.entities)
    # [{entity='Austria', ent_type='GPE', start_index=23, end_index=30, metadata={}}]

    References
    ----------
    .. [1] https://spacy.io/
    """

    sentences = []
    refs = []
    if isinstance(texts, list):
        for text in texts:
            if isinstance(text, Instance):
                if text.question is not None:
                    sentences.append(text.question.question)
                    refs.append(text.question)
                if text.answers is not None and isinstance(text.answers, list):
                    sentences.extend([txt.answer for txt in text.answers])
                    refs.extend([txt for txt in text.answers])
                if text.hints is not None and isinstance(text.hints, list):
                    sentences.extend([txt.hint for txt in text.hints])
                    refs.extend([txt for txt in text.hints])
            elif isinstance(text, Question):
                sentences.append(text.question)
                refs.append(text)
            elif isinstance(text, Answer):
                sentences.append(text.answer)
                refs.append(text)
            elif isinstance(text, Hint):
                sentences.append(text.hint)
                refs.append(text)
            else:
                raise ValueError(
                    f'All items for detecting entities must be instances of the following classes: Instance, Question, Answer, or Hint.')
    else:
        raise ValueError(
            f'All items for detecting entities must be instances of the following classes: Instance, Question, Answer, or Hint.')
    ent_spacy = EntitySpacy(batch_size, spacy_pipeline, enable_tqdm)
    ent_spacy.predict(refs, sentences)


def identify_question_type(texts: List[Question], batch_size, force_download, enable_tqdm):
    """
    Identifies question types for the given questions using a pre-trained classifier `[2]`_.

    .. _[2]: https://www.sciencedirect.com/science/article/pii/S0306457322001959

    Parameters
    ----------
    texts : list of :class:`Question`
        A list of :class:`Question` instances to perform question type classification on.
    batch_size : int
        The batch size for processing texts.
    force_download : bool
        Whether to force download the question classification model files.
    enable_tqdm : bool
        Whether to enable tqdm progress bar.

    Raises
    ------
    ValueError
        If any item in texts is not an instance of the :class:`Question` class.

    Examples
    --------
    >>> from hinteval.cores import Question
    >>> from hinteval.utils.identify_functions import identify_question_type
    >>>
    >>> questions = [Question("What is the capital of Austria?")]
    >>> identify_question_type(questions, batch_size=64, force_download=True, enable_tqdm=True)
    >>> print(questions[0].question_type)
    # {'major': 'LOC:LOCATION', 'minor': 'city:City'}

    References
    ----------
    .. [2] Mallikarjuna, C., Sivanesan, S., 2022. Question classification using limited labelled data. Information Processing & Management 59 (6), 103094.
    """

    sentences = []
    if isinstance(texts, list):
        for text in texts:
            if isinstance(text, Question):
                sentences.append(text.question)
            else:
                raise ValueError(f'All items for detecting question type must be instances of the Question classes.')
    else:
        raise ValueError(f'All items for detecting question type must be instances of the Question classes.')
    qc = QC(batch_size, force_download, enable_tqdm)
    qc.predict(texts, sentences)
