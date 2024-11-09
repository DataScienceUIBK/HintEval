import json
import random as rnd
from typing import List, Dict, Union, Any

class NotComparableException(Exception):
    def __init__(self, operator, name_1, name_2):
        self.operator = operator
        self.name_1 = name_1
        self.name_2 = name_2
        super().__init__(f"'{operator}' not supported between metrics of '{name_1}' and '{name_2}'")

    def __str__(self):
        return f"'{self.operator}' not supported between metrics of '{self.name_1}' and '{self.name_2}'"


class _CoreEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Entity, Metric, Question, Answer, Hint, Instance, Subset)):
            return obj.to_dict()
        return super().default(obj)


class Metric:
    """
    A class used to represent a Metric, which includes a name, a value, and optional metadata.

    Attributes
    ----------
    name : str
        The name of the metric.
    value : Union[str,int, float]
        The value associated with the metric.
    metadata : dict[str, Union[str,int, float]]
        The metadata associated with the metric.
    """

    def __init__(self, name: str, value: Union[str,int, float], metadata: Dict[str, Union[str,int, float]] = None):
        """
        Initializes a new instance of the Metric class.

        Parameters
        ----------
        name : str
            The name of the metric.
        value : Union[str,int, float]
            The value of the metric.
        metadata : dict[str, Union[str,int, float]], optional
            A dictionary containing metadata for the metric (default is an empty dictionary).

        Examples
        --------
        >>> from hinteval.cores import Metric
        >>>
        >>> metric = Metric("readability", 0.4, {"model": "bert-base"})
        >>> print(metric.name, metric.value, metric.metadata)
        # readability 0.4 {'model': 'bert-base'}

        See Also
        --------
        from_dict :
            Creates a Metric instance from a dictionary.

        """

        self.name = name
        self.value = value
        self.metadata: Dict[str, Union[str,int, float]] = metadata if metadata is not None else {}

    def to_dict(self):
        """
        Converts the Metric instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Metric instance with keys 'name', 'value', and 'metadata'.

        Examples
        -------
        >>> from hinteval.cores import Metric
        >>>
        >>> metric = Metric("readability", 0.4, {"model": "bert-base"})
        >>> print(metric.to_dict())
        # {'name': 'readability', 'value': 0.4, 'metadata': {'model': 'bert-base'}}

        See Also
        --------
        from_dict :
            Creates a Metric instance from a dictionary.

        """
        return {
            "name": self.name,
            "value": self.value,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Creates a Metric instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
           A dictionary containing the keys 'name', 'value', and 'metadata'.

        Returns
        -------
        Metric
           A new instance of Metric initialized with the data from the dictionary.

        Examples
        -------
        >>> from hinteval.cores import Metric
        >>>
        >>> data = {'name': 'readability', 'value': 0.4, 'metadata': {'model': 'bert-base'}}
        >>> metric = Metric.from_dict(data)
        >>> print(metric.name, metric.value, metric.metadata)
        # readability 0.4 {'model': 'bert-base'}

        See Also
        --------
        to_dict :
            Converts the Metric instance into a dictionary.

        Raises
        ------
        KeyError
           If the 'name', 'value', or 'metadata' keys are missing in the dictionary.
        """

        return cls(name=data["name"], value=data["value"], metadata=data["metadata"] if 'metadata' in data else None)

    def __eq__(self, other):
        if isinstance(other, Metric) and self.name == other.name:
            return self.value == other.value
        raise NotComparableException('==', self.name, other.name)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return json.dumps(self.to_dict(), cls=_CoreEncoder, indent=4)


class Entity:
    """
    A class to represent an entity with its type and position in the text, along with optional metadata.

    Attributes
    ----------
    entity : str
        The textual representation of the entity.
    ent_type : str
        The type of the entity, e.g., 'PERSON', 'LOCATION'.
    start_index : int
        The start index of the entity in the text.
    end_index : int
        The end index of the entity in the text, non-inclusive.
    metadata : dict[str, Union[str,int, float]]
        Optional additional metadata about the entity.
    """
    def __init__(self, entity: str, ent_type: str, start_index: int, end_index: int, metadata: Dict[str, Union[str,int, float]] = None):
        """
        Initializes an Entity instance.

        Parameters
        ----------
        entity : str
            The textual content of the entity.
        ent_type : str
            The type of the entity, such as 'PERSON' or 'LOCATION'.
        start_index : int
            The index of the first character of the entity in the text.
        end_index : int
            The index of the character immediately after the last character of the entity in the text.
        metadata : dict[str, Union[str,int, float]], optional
            A dictionary of additional data related to the entity (default is an empty dictionary).

        Examples
        -------
        >>> from hinteval.cores import Entity
        >>>
        >>> entity = Entity("Lionel Messi", "PERSON", 0, 12, {"familiarity": 1.0})
        >>> print(entity.entity, entity.ent_type, entity.start_index, entity.end_index, entity.metadata)
        # Lionel Messi PERSON 0 12 {'familiarity': 1.0}

        See Also
        --------
        from_dict :
            Creates an Entity instance from a dictionary.

        """

        self.entity = entity
        self.ent_type = ent_type
        self.start_index = start_index
        self.end_index = end_index
        self.metadata: Dict[str, Union[str,int, float]] = metadata if metadata is not None else {}

    def to_dict(self):
        """
        Converts the Entity instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representing the Entity with keys 'entity', 'ent_type', 'start_index', 'end_index', and 'metadata'.

        Examples
        -------
        >>> from hinteval.cores import Entity
        >>>
        >>> entity = Entity("Lionel Messi", "PERSON", 0, 12, {"familiarity": 1.0})
        >>> print(entity.to_dict())
        # {'entity': 'Lionel Messi', 'ent_type': 'PERSON', 'start_index': 0, 'end_index': 12, 'metadata': {'familiarity': 1.0}}

        See Also
        --------
        from_dict :
            Creates an Entity instance from a dictionary.

        """

        return {
            "entity": self.entity,
            "ent_type": self.ent_type,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Creates an Entity instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing keys 'entity', 'ent_type', 'start_index', 'end_index', and 'metadata'.

        Returns
        -------
        Entity
            A new instance of Entity initialized from the provided dictionary.

        Examples
        -------
        >>> from hinteval.cores import Entity
        >>>
        >>> data = {'entity': 'Lionel Messi', 'ent_type': 'PERSON', 'start_index': 0,
        ...         'end_index': 12, 'metadata': {'familiarity': 1.0}}
        >>> entity = Entity.from_dict(data)
        >>> print(entity.entity, entity.metadata['familiarity'])
        # Lionel Messi 1.0

        See Also
        --------
        to_dict :
            Converts the Entity instance into a dictionary.

        Raises
        ------
        KeyError
            If any of the required keys ('entity', 'ent_type', 'start_index', 'end_index') are missing in the dictionary.
        """

        return cls(entity=data["entity"], ent_type=data["ent_type"], start_index=data["start_index"],
                   end_index=data["end_index"], metadata=data["metadata"] if 'metadata' in data else None)

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.entity == other.entity and self.ent_type == other.ent_type and self.start_index == other.start_index and self.end_index == other.end_index
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return json.dumps(self.to_dict(), cls=_CoreEncoder, indent=4)


class Question:
    """
    A class to represent a structured question with associated types, entities, metrics, and optional metadata.

    Attributes
    ----------
    question : str
        The text of the question.
    question_type : dict[str, str]
        A dictionary mapping question aspects to their types.
    entities : list[Entity]
        A list of :class:`Entity` instances associated with the question.
    metrics : dict[str, Metric]
        A dictionary of :class:`Metric` instances associated with the question, keyed by their names.
    metadata : dict[str, Union[str,int, float]]
        Optional additional metadata about the question.
    """

    def __init__(self, question: str, question_type: Dict[str, str] = None, entities: List[Entity] = None,
                 metrics: Dict[str, Metric] = None, metadata: Dict[str, Union[str,int, float]] = None):
        """
        Initializes a new instance of the Question class.

        Parameters
        ----------
        question : str
            The text of the question.
        question_type : dict[str, str], optional
            A dictionary mapping question aspects to their types (default is an empty dictionary).
        entities : list[Entity], optional
            A list of :class:`Entity` objects associated with the question (default is an empty list).
        metrics : dict[str, Metric], optional
            A dictionary where keys are metric names and values are :class:`Metric` objects (default is an empty dictionary).
        metadata : dict[str, Union[str,int, float]], optional
            A dictionary containing metadata about the question (default is an empty dictionary).

        Examples
        -------
        >>> from hinteval.cores import Entity, Metric, Question
        >>>
        >>> question = "What is the capital of Austria?"
        >>> question_type = {"major": "LOC:LOCATION"}
        >>> entities = [Entity("Austria", "LOCATION", 23, 30, {"familiarity": 1.0})]
        >>> metrics = {"readability": Metric("readability", 0.8)}
        >>> metadata = {"source": "https://en.wikipedia.org/wiki/Austria"}
        >>> q = Question(question, question_type, entities, metrics, metadata)
        >>> print(q.question)
        # What is the capital of Austria?

        See Also
        --------
        from_dict :
            Creates a Question instance from a dictionary.

        """

        self.question = question
        self.question_type: Dict[str, str] = question_type if question_type is not None else {}
        self.entities: List[Entity] = entities if entities is not None else []
        self.metrics: Dict[str, Metric] = metrics if metrics is not None else {}
        self.metadata: Dict[str, Union[str,int, float]] = metadata if metadata is not None else {}

    def to_dict(self):
        """
        Converts the Question instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Question instance including all its attributes.

        Examples
        -------
        >>> from hinteval.cores import Entity, Metric, Question
        >>>
        >>> question = Question(
        ...     "What is the capital of Austria?",
        ...     {"major": "LOC:LOCATION"},
        ...     [Entity("Austria", "LOCATION", 23, 30, {"familiarity": 1.0})],
        ...     {"readability": Metric("readability", 0.8)},
        ...     {"source": "https://en.wikipedia.org/wiki/Austria"}
        ... )
        >>> print(question.to_dict())
        # {
        #     'question': 'What is the capital of Austria?',
        #     'question_type': {'type': 'LOC:LOCATION'},
        #     'entities': [
        #         {'entity': 'Austria', 'ent_type': 'LOCATION', 'start_index': 23, 'end_index': 30, 'metadata': {'familiarity': 1.0}}
        #     ],
        #     'metrics': {'readability': {'name': 'familiarity', 'value': 0.8, 'metadata': {}}},
        #     'metadata': {'source': 'https://en.wikipedia.org/wiki/Austria'}
        # }

        See Also
        --------
        from_dict :
            Creates a Question instance from a dictionary.

        """

        ret_dict = {'question': self.question}
        ret_dict.update({'question_type': self.question_type})
        ret_dict.update({'entities': [entity.to_dict() for entity in self.entities]})
        ret_dict.update({'metrics': {key: val.to_dict() for key, val in self.metrics.items()}})
        ret_dict.update({'metadata': self.metadata})
        return ret_dict

    @classmethod
    def from_dict(cls, data):
        """
        Creates a Question instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing all necessary attributes to instantiate a Question object.

        Returns
        -------
        Question
            A new instance of Question initialized from the provided dictionary.

        Examples
        -------
        >>> from hinteval.cores import Question
        >>>
        >>> data = {
        ...     'question': 'What is the capital of Austria?',
        ...     'question_type': {'type': 'LOC:LOCATION'},
        ...     'entities': [{'entity': 'Austria', 'ent_type': 'LOCATION',
        ...     'start_index': 23, 'end_index': 30, 'metadata': {'familiarity': 1.0}}],
        ...     'metrics': {'readability': {'name': 'readability', 'value': 0.8}},
        ...     'metadata': {'source': 'https://en.wikipedia.org/wiki/Austria'}
        ... }
        >>> question = Question.from_dict(data)
        >>> print(question.question)
        # What is the capital of Austria?

        See Also
        --------
        to_dict :
            Converts the Question instance into a dictionary.

        Raises
        ------
        KeyError
            If required keys are missing in the dictionary.
        """

        question = data['question']
        question_type = data['question_type'] if 'question_type' in data else None
        metadata = data['metadata'] if 'metadata' in data else None
        entities = []
        if 'entities' in data:
            for entity in data['entities']:
                entities.append(Entity.from_dict(entity))
        metrics = dict()
        if 'metrics' in data:
            for key, value in data['metrics'].items():
                metrics[key] = Metric.from_dict(value)
        return cls(question=question, question_type=question_type, entities=entities, metrics=metrics,
                   metadata=metadata)

    def __eq__(self, other):
        return self.question == other.question and self.question_type == other.question_type and self.entities == other.entities and self.metadata == other.metadata

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return json.dumps(self.to_dict(), cls=_CoreEncoder, indent=4)


class Answer:
    """
    A class to represent an answer with associated entities, metrics, and optional metadata.

    Attributes
    ----------
    answer : str
        The text of the answer.
    entities : list[Entity]
        A list of :class:`Entity` instances associated with the answer.
    metrics : dict[str, Metric]
        A dictionary of :class:`Metric` instances associated with the answer, keyed by their names.
    metadata : dict[str, Union[str,int, float]]
        Optional additional metadata about the answer.
    """

    def __init__(self, answer: str, entities: List[Entity] = None, metrics: Dict[str, Metric] = None,
                 metadata: Dict[str, Union[str,int, float]] = None):
        """
        Initializes a new instance of the Answer class.

        Parameters
        ----------
        answer : str
            The text of the answer.
        entities : list[Entity], optional
            A list of :class:`Entity` objects associated with the answer (default is an empty list).
        metrics : dict[str, Metric], optional
            A dictionary where keys are metric names and values are :class:`Metric` objects (default is an empty dictionary).
        metadata : dict[str, Union[str,int, float]], optional
            A dictionary containing metadata about the answer (default is an empty dictionary).

        Examples
        -------
        >>> from hinteval.cores import Entity, Metric, Answer
        >>>
        >>> answer = "Vienna"
        >>> entities = [Entity("Vienna", "LOCATION", 0, 6)]
        >>> metrics = {"familiarity": Metric("familiarity", 1.0)}
        >>> metadata = {"source": "https://en.wikipedia.org/wiki/Austria"}
        >>> ans = Answer(answer, entities, metrics, metadata)
        >>> print(ans.answer)
        # Vienna

        See Also
        --------
        from_dict :
            Creates an Answer instance from a dictionary.

        """

        self.answer = answer
        self.entities: List[Entity] = entities if entities is not None else []
        self.metrics: Dict[str, Metric] = metrics if metrics is not None else {}
        self.metadata: Dict[str, Union[str,int, float]] = metadata if metadata is not None else {}

    def to_dict(self):
        """
        Converts the Answer instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Answer instance including all its attributes.

        Examples
        -------
        >>> from hinteval.cores import Entity, Metric, Answer
        >>>
        >>> answer = Answer(
        ...     "Vienna",
        ...     [Entity("Vienna", "LOCATION", 0, 6)],
        ...     {"familiarity": Metric("familiarity", 1.0)},
        ...     {"source": "https://en.wikipedia.org/wiki/Austria"}
        ...)
        >>> print(answer.to_dict())
        # {
        #     'answer': 'Vienna',
        #     'entities': [
        #         {'entity': 'Vienna', 'ent_type': 'LOCATION', 'start_index': 0, 'end_index': 6}
        #     ],
        #     'metrics': {'familiarity': {'name': 'familiarity', 'value': 1.0}},
        #     'metadata': {'source': 'https://en.wikipedia.org/wiki/Austria'}
        # }

        See Also
        --------
        from_dict :
            Creates an Answer instance from a dictionary.

        """

        ret_dict = {'answer': self.answer}
        ret_dict.update({'entities': [entity.to_dict() for entity in self.entities]})
        ret_dict.update({'metrics': {key: val.to_dict() for key, val in self.metrics.items()}})
        ret_dict.update({'metadata': self.metadata})
        return ret_dict

    @classmethod
    def from_dict(cls, data):
        """
        Creates an Answer instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing all necessary attributes to instantiate an Answer object.

        Returns
        -------
        Answer
            A new instance of Answer initialized from the provided dictionary.

        Examples
        -------
        >>> from hinteval.cores import Answer
        >>>
        >>> data = {
        ...     'answer': 'Vienna',
        ...     'entities': [{'entity': 'Vienna', 'ent_type':
        ...     'LOCATION', 'start_index': 0, 'end_index': 6}],
        ...     'metrics': {'familiarity': {'name': 'familiarity', 'value': 1.0}},
        ...     'metadata': {'source': 'https://en.wikipedia.org/wiki/Austria'}
        ... }
        >>> answer = Answer.from_dict(data)
        >>> print(answer.answer)
        # Vienna

        See Also
        --------
        to_dict :
            Converts the Answer instance into a dictionary.

        Raises
        ------
        KeyError
            If required keys are missing in the dictionary.
        """

        answer = data['answer']
        metadata = data['metadata'] if 'metadata' in data else None
        entities = []
        if 'entities' in data:
            for entity in data['entities']:
                entities.append(Entity.from_dict(entity))
        metrics = dict()
        if 'metrics' in data:
            for key, value in data['metrics'].items():
                metrics[key] = Metric.from_dict(value)
        return cls(answer=answer, entities=entities, metrics=metrics, metadata=metadata)

    def __eq__(self, other):
        return self.answer == other.answer and self.entities == other.entities and self.metadata == other.metadata

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return json.dumps(self.to_dict(), cls=_CoreEncoder, indent=4)


class Hint:
    """
    A class to represent a hint associated with questions and answers, including sources, entities, metrics, and optional metadata.

    Attributes
    ----------
    hint : str
        The text of the hint.
    source : str, optional
        The source from which the hint was derived.
    entities : list[Entity]
        A list of :class:`Entity` instances related to the hint.
    metrics : dict[str, Metric]
        A dictionary of :class:`Metric` instances associated with the hint, keyed by their names.
    metadata : dict[str, Union[str,int, float]]
        Optional additional metadata about the hint.
    """

    def __init__(self, hint, source: str = None, entities: List[Entity] = None, metrics: Dict[str, Metric] = None,
                 metadata: Dict[str, Union[str,int, float]] = None):
        """
        Initializes a new instance of the Hint class.

        Parameters
        ----------
        hint : str
            The text of the hint.
        source : str, optional
            The source from which the hint was derived.
        entities : list[Entity]
            A list of :class:`Entity` instances related to the hint.
        metrics : dict[str, Metric]
            A dictionary of :class:`Metric` instances associated with the hint, keyed by their names.
        metadata : dict[str, Union[str,int, float]]
            Optional additional metadata about the hint.

        Examples
        -------
        >>> from hinteval.cores import Entity, Metric, Hint
        >>>
        >>> hint_text = "This city, once home to Mozart and Beethoven, is famous for its music and culture."
        >>> source = "https://en.wikipedia.org/wiki/Vienna"
        >>> entities = [
        ...     Entity("Mozart", "PERSON", 29, 35, {"url": "https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart"}),
        ...     Entity("Beethoven", "PERSON", 40, 49, {"url": "https://en.wikipedia.org/wiki/Ludwig_van_Beethoven"})
        ... ]
        >>> metrics = {"relevance": Metric("relevance", 0.9)}
        >>> hint = Hint(hint_text, source, entities, metrics)
        >>> print(hint.hint)
        # This city, once home to Mozart and Beethoven, is famous for its music and culture.

        See Also
        --------
        from_dict :
            Creates a Hint instance from a dictionary.

        """

        self.hint = hint
        self.source: str = source
        self.entities: List[Entity] = entities if entities is not None else []
        self.metrics: Dict[str, Metric] = metrics if metrics is not None else {}
        self.metadata: Dict[str, Union[str,int, float]] = metadata if metadata is not None else {}

    def to_dict(self):
        """
        Converts the Hint instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Hint instance including all its attributes.

        Examples
        -------
        >>> from hinteval.cores import Entity, Metric, Hint
        >>>
        >>> hint = Hint(
        ...     "This city, once home to Mozart and Beethoven, is famous for its music and culture.",
        ...     "https://en.wikipedia.org/wiki/Vienna",
        ...     [Entity("Mozart", "PERSON", 29, 35, {"url": "https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart"}),
        ...     Entity("Beethoven", "PERSON", 40, 49, {"url": "https://en.wikipedia.org/wiki/Ludwig_van_Beethoven"})],
        ...     {"relevance": Metric("relevance", 0.9)}
        ...     )
        >>> print(hint.to_dict())
        # {
        #     'hint': 'This city, once home to Mozart and Beethoven, is famous for its music and culture.',
        #     'source': 'https://en.wikipedia.org/wiki/Vienna',
        #     'entities': [
        #         {'entity': 'Mozart', 'ent_type': 'PERSON', 'start_index': 29, 'end_index': 35, 'metadata': {'url': 'https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart'}},
        #         {'entity': 'Beethoven', 'ent_type': 'PERSON', 'start_index': 40, 'end_index': 49, 'metadata': {'url': 'https://en.wikipedia.org/wiki/Ludwig_van_Beethoven'}}
        #     ],
        #     'metrics': {'relevance': {'name': 'relevance', 'value': 0.9, 'metadata': {}}},
        #     'metadata': {}
        # }

        See Also
        --------
        from_dict :
            Creates a Hint instance from a dictionary.

        """

        ret_dict = {'hint': self.hint}
        ret_dict.update({'source': self.source})
        ret_dict.update({'entities': [entity.to_dict() for entity in self.entities]})
        ret_dict.update({'metrics': {key: val.to_dict() for key, val in self.metrics.items()}})
        ret_dict.update({'metadata': self.metadata})
        return ret_dict

    @classmethod
    def from_dict(cls, data):
        """
        Creates a Hint instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing all necessary attributes to instantiate a Hint object.

        Returns
        -------
        Hint
            A new instance of Hint initialized from the provided dictionary.

        Examples
        -------
        >>> from hinteval.cores import Hint
        >>>
        >>> data = {
        ...     'hint': 'This city, once home to Mozart and Beethoven, is famous for its music and culture.',
        ...     'source': 'https://en.wikipedia.org/wiki/Vienna',
        ...     'entities': [
        ...        {'entity': 'Mozart', 'ent_type': 'PERSON', 'start_index': 29, 'end_index': 35, 'metadata': {'url': 'https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart'}},
        ...         {'entity': 'Beethoven', 'ent_type': 'PERSON', 'start_index': 40, 'end_index': 49, 'metadata': {'url': 'https://en.wikipedia.org/wiki/Ludwig_van_Beethoven'}}
        ...     ],
        ...     'metrics': {'relevance': {'name': 'relevance', 'value': 0.9}},
        ...     'metadata': {}
        ... }
        >>> hint = Hint.from_dict(data)
        >>> print(hint.hint)
        # This city, once home to Mozart and Beethoven, is famous for its music and culture.

        See Also
        --------
        to_dict :
            Converts the Hint instance into a dictionary.

        Raises
        ------
        KeyError
            If required keys are missing in the dictionary.
        """

        hint = data['hint']
        source = data['source'] if 'source' in data else None
        metadata = data['metadata'] if 'metadata' in data else None
        entities = []
        if 'entities' in data:
            for entity in data['entities']:
                entities.append(Entity.from_dict(entity))
        metrics = dict()
        if 'metrics' in data:
            for key, value in data['metrics'].items():
                metrics[key] = Metric.from_dict(value)
        return cls(hint=hint, source=source, entities=entities, metrics=metrics, metadata=metadata)

    def __eq__(self, other):
        return self.hint == other.hint and self.entities == other.entities and self.source == other.source and self.metrics == other.metrics and self.metadata == other.metadata

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return json.dumps(self.to_dict(), cls=_CoreEncoder, indent=4)


class Instance:
    """
    A class to represent a question-and-answer instance along with associated hints and metadata.

    Attributes
    ----------
    question : Question
        The :class:`Question` object representing the primary question.
    answers : list[Answer]
        A list of :class:`Answer` objects providing possible answers to the question.
    hints : list[Hint]
        A list of :class:`Hint` objects providing additional context or clues for the question.
    metadata : dict[str, Union[str,int, float]]
        Optional additional metadata about the instance.
    """

    def __init__(self, question: Question, answers: List[Answer], hints: List[Hint], metadata: Dict[str, Union[str,int, float]] = None):
        """
        Initializes a new instance of the Instance class.

        Parameters
        ----------
        question : Question
            The :class:`Question` object representing the primary question.
        answers : list[Answer]
            A list of :class:`Answer` objects providing possible answers.
        hints : list[Hint]
            A list of :class:`Hint` objects providing additional context or clues.
        metadata : dict[str, Union[str,int, float]], optional
            A dictionary containing metadata about the instance (default is an empty dictionary).

        Examples
        -------
        >>> from hinteval.cores import Instance, Question, Answer, Hint
        >>>
        >>> question = Question("What is the capital of France?")
        >>> answers = [Answer("Paris")]
        >>> hints = [Hint("This city is also known as the City of Lights.")]
        >>> instance = Instance(question, answers, hints)
        >>> print(instance.question.question)
        # What is the capital of France?

        See Also
        --------
        from_dict :
            Creates an Instance object from a dictionary.

        from_strings :
            Creates an Instance object from strings representing the `question`, `answers`, and `hints`.

        """

        self.question: Question = question
        self.answers: List[Answer] = answers
        self.hints: List[Hint] = hints
        self.metadata: Dict[str, Union[str,int, float]] = metadata if metadata is not None else {}

    @classmethod
    def from_strings(cls, question: str, answers: List[str], hints: List[str]):
        """
        Creates an Instance object from strings representing the question, answers, and hints.

        Parameters
        ----------
        question : str
            The text of the primary question.
        answers : list[str]
            A list of strings representing the answers.
        hints : list[str]
            A list of strings representing the hints.

        Returns
        -------
        Instance
            A new Instance object populated with the converted question, answers, and hints.

        Examples
        -------
        >>> from hinteval.cores import Instance
        >>>
        >>> instance = Instance.from_strings(
        ...     "What is the capital of France?",
        ...     ["Paris"],
        ...     ["This city is also known as the City of Lights."]
        ... )
        >>> print(instance.question.question)
        # What is the capital of France?

        See Also
        --------
        from_dict :
            Creates an Instance object from a dictionary.

        to_dict :
            Converts the Instance object into a dictionary.

        """

        question = Question(question)
        answers = [Answer(ans) for ans in answers]
        hints = [Hint(hint) for hint in hints]
        return cls(question, answers, hints)

    def hints_from_strings(self, hints: List[str]):
        """
        Updates the hints of the instance using a list of string representations.

        Parameters
        ----------
        hints : list[str]
            A list of strings where each string represents a hint.

        Examples
        -------
        >>> from hinteval.cores import Instance
        >>>
        >>> instance = Instance.from_strings(
        ...     "What is the capital of France?",
        ...     ["Paris"],
        ...     []
        ... )
        >>> instance.hints_from_strings(["This city is also known as the City of Lights.", "It is a major European city."])
        >>> print([hint.hint for hint in instance.hints])
        # ['This city is also known as the City of Lights.', 'It is a major European city.']

        See Also
        --------
        from_dict :
            Creates an Instance object from a dictionary.

        from_strings :
            Creates an Instance object from strings representing the `question`, `answers`, and `hints`.

        """

        self.hints = [Hint(hint) for hint in hints]

    def answers_from_strings(self, answers: List[str]):
        """
        Updates the answers of the instance using a list of string representations.

        Parameters
        ----------
        answers : list[str]
            A list of strings where each string represents an answer.

        Examples
        -------
        >>> from hinteval.cores import Instance
        >>>
        >>> instance = Instance.from_strings(
        ...     "What is the capital of France?",
        ...     ['Lyon'],
        ...     ["This city is also known as the City of Lights."]
        ... )
        >>> instance.answers_from_strings(["Paris"])
        >>> print([answer.answer for answer in instance.answers])
        # ['Paris']

        See Also
        --------
        from_dict :
            Creates an Instance object from a dictionary.

        from_strings :
            Creates an Instance object from strings representing the `question`, `answers`, and `hints`.

        """

        self.answers = [Answer(ans) for ans in answers]

    def question_from_string(self, question: str):
        """
        Updates the question of the instance using a string representation.

        Parameters
        ----------
        question : str
            The text of the question.

        Examples
        -------
        >>> from hinteval.cores import Instance
        >>>
        >>> instance = Instance.from_strings(
        ...     "What is the capital of Italy?",
        ...     ["Paris"],
        ...     ["This city is also known as the City of Lights."]
        ... )
        >>> instance.question_from_string("What is the capital of France?")
        >>> print(instance.question.question)
        # What is the capital of France?

        See Also
        --------
        from_dict :
            Creates an Instance object from a dictionary.

        from_strings :
            Creates an Instance object from strings representing the `question`, `answers`, and `hints`.

        """

        self.question = Question(question)

    def to_dict(self):
        """
        Converts the Instance object into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Instance object including all its attributes.

        Examples
        -------
        >>> from hinteval.cores import Instance, Question, Answer, Hint
        >>>
        >>> question = Question("What is the capital of France?")
        >>> answers = [Answer("Paris")]
        >>> hints = [Hint("This city is also known as the City of Lights.")]
        >>> instance = Instance(question, answers, hints)
        >>> print(instance.to_dict())
        # {
        #     'question': {'question': 'What is the capital of France?', ...},
        #     'answers': [{'answer': 'Paris', ...}],
        #     'hints': [{'hint': 'This city is also known as the City of Lights.', ...}],
        #     'metadata': {}
        # }

        See Also
        --------
        from_dict :
            Creates an Instance object from a dictionary.

        from_strings :
            Creates an Instance object from strings representing the `question`, `answers`, and `hints`.

        """

        ret_dict = {'question': self.question.to_dict()}
        ret_dict.update({'answers': [answer.to_dict() for answer in self.answers]})
        ret_dict.update({'hints': [hint.to_dict() for hint in self.hints]})
        ret_dict.update({'metadata': self.metadata})
        return ret_dict

    @classmethod
    def from_dict(cls, data):
        """
        Creates an Instance object from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing all necessary attributes to instantiate an Instance object.

        Returns
        -------
        Instance
            A new Instance object initialized from the provided dictionary.

        Examples
        -------
        >>> from hinteval.cores import Instance
        >>>
        >>> data = {
        ...     'question': {'question': 'What is the capital of France?'},
        ...     'answers': [{'answer': 'Paris'}],
        ...     'hints': [{'hint': 'This city is also known as the City of Lights.'}]
        ... }
        >>> instance = Instance.from_dict(data)
        >>> print(instance.question.question)
        # What is the capital of France?

        See Also
        --------
        to_dict :
            Converts the Instance object into a dictionary.

        from_strings :
            Creates an Instance object from strings representing the `question`, `answers`, and `hints`.

        Raises
        ------
        KeyError
            If required keys are missing in the dictionary.

        """

        question = Question.from_dict(data['question'])
        metadata = data['metadata'] if 'metadata' in data else None
        answers = []
        for answer in data['answers']:
            answers.append(Answer.from_dict(answer))
        hints = []
        for hint in data['hints']:
            hints.append(Hint.from_dict(hint))
        return cls(question=question, answers=answers, hints=hints, metadata=metadata)

    def __eq__(self, other):
        return self.question == other.question and self.answers == other.answers and self.hints == other.hints and self.metadata == other.metadata

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return json.dumps(self.to_dict(), cls=_CoreEncoder, indent=4)


class Subset:
    """
    A class to represent a subset of instances, typically used for managing and organizing a collection of instances with associated metadata.

    Attributes
    ----------
    name : str
        The name of the subset.
    metadata : dict[str, Union[str,int, float]]
        Optional additional metadata about the subset.
    """

    def __init__(self, name: str = 'entire', metadata: Dict[str, Union[str,int, float]] = None):
        """
        Initializes a new Subset instance.

        Parameters
        ----------
        name : str, optional
            The name of the subset (default is 'entire').
        metadata : dict[str, Union[str,int, float]], optional
            Additional metadata about the subset (default is an empty dictionary).

        Examples
        -------
        >>> from hinteval.cores import Subset
        >>>
        >>> subset = Subset(name='training_set')
        >>> print(subset.name)
        # training_set

        See Also
        --------
        from_dict :
            Creates a Subset object from a dictionary.

        """

        self.name = name
        self._instances: Dict[str, Instance] = dict()
        self.metadata: Dict[str, Union[str,int, float]] = metadata if metadata is not None else {}

    def _generate_id(self, name):
        return f'{name}_{rnd.randint(1000000, 9999999)}'

    def add_instance(self, instance: Instance, q_id: str = None):
        """
        Adds an instance to the subset.

        Parameters
        ----------
        instance : Instance
            The `instance` to be added to the subset.
        q_id : str, optional
            The unique identifier for the `instance` (default is None, which generates a new random unique ID).

        Raises
        ------
        ValueError
            If the provided `q_id` already exists in the subset.

        Examples
        -------
        >>> from hinteval.cores import Subset, Instance
        >>>
        >>> subset = Subset()
        >>> instance = Instance.from_strings("What is the capital of France?", ["Paris"], [])
        >>> subset.add_instance(instance, "q1")
        >>> # subset["q1"] = instance
        >>> print(subset["q1"].answers[0].answer)
        # Paris

        See Also
        --------
        get_instance :
            Retrieves an instance by its unique identifier.

        remove_instance :
            Removes an instance from the subset.
        """

        if q_id is None:
            q_id = self._generate_id(self.name)
            while q_id in self._instances.keys():
                q_id = self._generate_id(self.name)
        else:
            if q_id in self._instances.keys():
                raise ValueError(f'The id "{q_id}" is already.')
        self._instances[q_id] = instance

    def remove_instance(self, q_id: str):
        """
        Removes an instance from the subset.

        Parameters
        ----------
        q_id : str
            The unique identifier of the instance to be removed.

        Raises
        ------
        ValueError
            If the `q_id` does not exist in the subset.

        Examples
        -------
        >>> from hinteval.cores import Subset, Instance
        >>>
        >>> subset = Subset()
        >>> instance = Instance.from_strings("What is the capital of France?", ["Paris"], [])
        >>> subset.add_instance(instance, "q1")
        >>> print(len(subset))
        # 1
        >>> subset.remove_instance("q1")
        >>> # del subset["q1"]
        >>> print(len(subset))
        # 0

        See Also
        --------
        get_instance :
            Retrieves an instance by its unique identifier.

        add_instance :
            Adds an instance to the subset.
        """

        if q_id not in self._instances.keys():
            raise ValueError(f'The id "{q_id}" is not in the subset.')
        del self._instances[q_id]

    def get_instance(self, q_id: str):
        """
        Retrieves an instance by its unique identifier.

        Parameters
        ----------
        q_id : str
            The unique identifier of the instance to retrieve.

        Returns
        -------
        Instance
            The instance associated with the given `q_id`.

        Raises
        ------
        ValueError
            If the `q_id` does not exist in the subset.

        Examples
        -------
        >>> from hinteval.cores import Subset, Instance
        >>>
        >>> subset = Subset()
        >>> instance = Instance.from_strings("What is the capital of France?", ["Paris"], [])
        >>> subset.add_instance(instance, "q1")
        >>> instance = subset.get_instance("q1")
        >>> # instance = subset["q1"]
        >>> print(instance.question.question)
        # What is the capital of France?

        See Also
        --------
        add_instance :
            Adds an instance to the subset.

        remove_instance :
            Removes an instance from the subset.

        get_instances :
            Retrieves all instances in the subset.

        get_instance_ids :
            Retrieves all instance identifiers in the subset.
        """

        if q_id not in self._instances.keys():
            raise ValueError(f'The id "{q_id}" is not in the subset.')
        return self._instances[q_id]

    def get_instance_ids(self):
        """
        Retrieves all instance identifiers in the subset.

        Returns
        -------
        list[str]
            A list of all instance identifiers in the subset.

        Examples
        -------
        >>> from hinteval.cores import Subset, Instance
        >>>
        >>> subset = Subset()
        >>> instance_1 = Instance.from_strings("What is the capital of France?", ["Paris"], [])
        >>> instance_2 = Instance.from_strings("What is the capital of Austria?", ["Vienna"], [])
        >>> subset["q1"] = instance_1
        >>> subset["q2"] = instance_2
        >>> ids = subset.get_instance_ids()
        >>> print(ids)
        # ['q1', 'q2']

        See Also
        --------
        get_instance :
            Retrieves an instance by its unique identifier.

        get_instances :
            Retrieves all instances in the subset.

        """

        return list(self._instances.keys())

    def get_instances(self):
        """
        Retrieves all instances in the subset.

        Returns
        -------
        list[Instance]
            A list of all instances in the subset.

        Examples
        -------
        >>> from hinteval.cores import Subset, Instance
        >>>
        >>> subset = Subset()
        >>> instance_1 = Instance.from_strings("What is the capital of France?", ["Paris"], [])
        >>> instance_2 = Instance.from_strings("What is the capital of Austria?", ["Vienna"], [])
        >>> subset["q1"] = instance_1
        >>> subset["q2"] = instance_2
        >>> instances = subset.get_instances()
        >>> print([instance.question.question for instance in instances])
        # ['What is the capital of France?', 'What is the capital of Austria?']

        See Also
        --------
        get_instance :
            Retrieves an instance by its unique identifier.

        get_instance_ids :
            Retrieves all instance identifiers in the subset.
        """

        return list(self._instances.values())

    def to_dict(self):
        """
        Converts the Subset instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Subset instance.

        Examples
        -------
        >>> from hinteval.cores import Subset, Instance
        >>>
        >>> instance1 = Instance.from_strings("What is the capital of France?", ["Paris"], [])
        >>> instance2 = Instance.from_strings("What is the capital of Austria?", ["Vienna"], [])
        >>> subset = Subset(name='training_set')
        >>> subset.add_instance(instance1, "q1")
        >>> subset.add_instance(instance2, "q2")
        >>> subset_dict = subset.to_dict()
        >>> print(subset_dict)
        # {
        #     'name': 'training_set',
        #     'metadata': {},
        #     'instances': {
        #         'q1': {
        #             'question': {'question': 'What is the capital of France?', ...},
        #             'answers': [{'answer': 'Paris', ...}],
        #             'hints': [],
        #             'metadata': {}
        #         },
        #         'q2': {
        #             'question': {'question': 'What is the capital of Austria?', ...},
        #             'answers': [{'answer': 'Vienna', ...}],
        #             'hints': [],
        #             'metadata': {}
        #         }
        #     }
        # }

        See Also
        --------
        from_dict :
            Creates a Subset instance from a dictionary.

        """

        ret_dict = {'name': self.name}
        ret_dict.update({'metadata': self.metadata})
        ret_dict.update({'instances': {key: val.to_dict() for key, val in self._instances.items()}})
        return ret_dict

    @classmethod
    def from_dict(cls, data):
        """
        Creates a Subset instance from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary containing all necessary attributes to instantiate a Subset object.

        Returns
        -------
        Subset
            A new Subset object initialized from the provided dictionary.

        Examples
        -------
        >>> from hinteval.cores import Subset
        >>>
        >>> data = {
        ...     'name': 'training_set',
        ...     'metadata': {},
        ...     'instances': {
        ...         'q1': {
        ...             'question': {'question': 'What is the capital of France?'},
        ...             'answers': [{'answer': 'Paris'}],
        ...             'hints': [],
        ...             'metadata': {}
        ...         },
        ...         'q2': {
        ...             'question': {'question': 'What is the capital of Austria?'},
        ...             'answers': [{'answer': 'Vienna'}],
        ...             'hints': [],
        ...             'metadata': {}
        ...         }
        ...     }
        ... }
        >>> subset = Subset.from_dict(data)
        >>> print(subset.name)
        # training_set
        >>> for instance_id in subset.get_instance_ids():
        ...     print(instance_id, subset[instance_id].question.question)
        # q1 What is the capital of France?
        # q2 What is the capital of Austria?

        Raises
        ------
        KeyError
            If required keys are missing in the dictionary.

        See Also
        --------
        to_dict :
            Converts the Subset instance into a dictionary.
        """

        name = data['name']
        metadata = data['metadata'] if 'metadata' in data else None
        new_cls = cls(name=name, metadata=metadata)
        if 'instances' in data:
            for q_id, instance in data['instances'].items():
                new_cls.add_instance(Instance.from_dict(instance), q_id)
        return new_cls

    def __getitem__(self, q_id):
        return self.get_instance(q_id)

    def __setitem__(self, q_id, instance):
        self.add_instance(instance, q_id)

    def __delitem__(self, q_id):
        self.remove_instance(q_id)

    def __len__(self):
        return len(self._instances)

    def __eq__(self, other):
        if isinstance(other, Subset):
            return self.name == other.name and self._instances == other._instances
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return json.dumps(self.to_dict(), cls=_CoreEncoder, indent=4)
