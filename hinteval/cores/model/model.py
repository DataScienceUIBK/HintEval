import itertools
import asyncio
import transformers
from hinteval.cores.model_core import Model
from hinteval.cores.dataset_core import Instance
from hinteval.utils.model.answer_aware.api_based import Hint_Generation as Hint_Generation_Aware_API
from hinteval.utils.model.answer_aware.local import Hint_Generation as Hint_Generation_Aware_Local
from hinteval.utils.model.answer_agnostic.api_based import Hint_Generation as Hint_Generation_Agnostic_API
from hinteval.utils.model.answer_agnostic.local import Hint_Generation as Hint_Generation_Agnostic_Local
from hinteval.utils.model.hint_filtering import Hint_Filtering
from typing import List, Callable
from tqdm import tqdm
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()


class AnswerAware(Model):
    """
    Class for automatically generating hints for questions that are aware of their answers `[39]`_.

    .. _[39]: https://dl.acm.org/doi/10.1145/3626772.3657855

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
    .. [39] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

    See Also
    --------
    :class:`AnswerAgnostic` : Class for automatically generating hints for questions that are unaware of their answers.
    """

    def __init__(self, model_name: str,
                 api_key: str = None,
                 base_url: str = 'https://api.together.xyz/v1',
                 num_of_hints: int = 5,
                 parse_llm_response: Callable[[str], List[str]] = None,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 max_tokens: int = 512,
                 batch_size: int = 2,
                 checkpoint: bool = False,
                 checkpoint_step: int = 1,
                 enable_tqdm=False):
        """
        Initializes the AnswerAware class with the specified large language model and configuration options `[40]`_.

        .. _[40]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        model_name : str
            The large language model to use for hint generation.
        api_key : str, optional
            Specifies the API key required for accessing and interacting with the model.
        base_url : str, default 'https://api.together.xyz/v1'
            Specifies the base URL for the API endpoints. This URL is used to construct full API request URLs.
        num_of_hints : int, default 5
            Number of hints to generate per instance.
        parse_llm_response : Callable[[str], List[str]], optional
            Function to parse the language model's output into a list of strings as hints.
        temperature : float, default 0.7
            Sampling temperature to control the diversity of generated content.
        top_p : float, default 1.0
            Nucleus sampling cutoff to control the randomness of hint generation.
        max_tokens : int, default 512
            Maximum number of tokens to generate for each hint.
        batch_size : int, default 2
            Number of instances processed in one batch during hint generation.
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
        If `api_key` is None, the generator will attempt to download the model and run it locally.

        If `parse_llm_response` is None, the generator will use the default function used in the `[41]`_.

        .. _[41]: https://dl.acm.org/doi/10.1145/3626772.3657855



        Examples
        --------
        >>> from hinteval.model import AnswerAware
        >>> answer_aware = AnswerAware(model_name='llama-3-8b', api_key='your_api_key', base_url='base_url', num_of_hints=5,
        ...                             parse_llm_response=None, temperature=0.7, top_p=1.0, max_tokens=512, batch_size=2,
        ...                             checkpoint=True, checkpoint_step=5, enable_tqdm=True)

        References
        ----------
        .. [40] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`AnswerAgnostic` : Class for automatically generating hints for questions that are unaware of their answers.
        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'answer_aware_{model_name.replace("/", "_")}.pickle'
        self._api_key = api_key
        self._base_url = base_url
        self._num_of_hints = num_of_hints
        self._parse_llm_response = parse_llm_response
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self.batch_size = batch_size
        self._model_name = model_name

        self._hint_filtering = Hint_Filtering()

        if self._api_key is None:
            self._pipeline = transformers.pipeline(
                "text-generation",
                model=self._model_name,
                device_map="auto"
            )
            self._pipeline.tokenizer.pad_token = self._pipeline.tokenizer.eos_token
            self._pipeline.tokenizer.pad_token_id = self._pipeline.tokenizer.eos_token_id

            self._hint_generator = Hint_Generation_Aware_Local(self._pipeline, self._num_of_hints, self._parse_llm_response,
                                                         self._temperature, self._top_p, self._max_tokens)
        else:
            self._hint_generator = Hint_Generation_Aware_API(self._base_url, self._api_key, self._model_name,
                                                       self._num_of_hints, self._parse_llm_response, self._temperature,
                                                       self._top_p, self._max_tokens)

    def generate(self, instances: List[Instance], **kwargs) -> List[List[str]]:
        """
        Generates hints for a list of instances using the configured large language model `[42]`_.

        .. _[42]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        instances : List[Instance]
            A list of instances, where each instance contains a question and its corresponding answer.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[str]]
            A list of lists containing generated hints for each instance.

        Notes
        -----
        This function stores the generated hints as :class:`Hint` objects within the `instances` attribute of the :class:`Subset`.

        Examples
        --------
        >>> from hinteval.cores import Instance, Question, Answer
        >>> from hinteval.model import AnswerAware
        >>>
        >>> answer_aware = AnswerAware(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        ...                           api_key='your_api_key',
        ...                           base_url='base_url',
        ...                           num_of_hints=2)
        >>> instance_1 = Instance(question=Question('What is the capital of Austria?'),
        ...                      answers=[Answer('Vienna')], hints=[])
        >>> instance_2 = Instance(question=Question('Who was the president of USA in 2009?'),
        ...                      answers=[Answer('Barack Obama')], hints=[])
        >>> instances = [instance_1, instance_2]
        >>> results = answer_aware.generate(instances)
        >>> print(results)
        # [
        #     ['The city is celebrated for its cultural landmarks like palaces and museums.',
        #      'It is historically significant, located near the Danube River.'],
        #     ['The president Obama, inaugurated in 2009, was the first African American to hold the office.',
        #      'He won the Nobel Peace Prize in 2009 for enhancing international diplomacy.']
        # ]

        Raises
        ------
        RuntimeError
            If hint generation fails due to model or API issues.

        References
        ----------
        .. [42] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`AnswerAgnostic` : Class for automatically generating hints for questions that are unaware of their answers.
        """

        self._validate_input(instances)
        pairs = []
        for idx, instance in enumerate(instances):
            q = instance.question.question
            a = instance.answers[0].answer
            pairs.append((q, a))
        pairs_batches = [pairs[i:i + self.batch_size] for i in range(0, len(pairs), self.batch_size)]

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        question_answer_pairs_stream = tqdm(pairs_batches, total=len(pairs_batches),
                                            desc=f'Generating hints using {self._model_name}') if self.enable_tqdm else pairs_batches
        for pairs_batch in question_answer_pairs_stream:
            _idx = next(item_counter)
            final_step = len(results) // self.batch_size
            if _idx <= final_step:
                continue
            if self._api_key is None:
                generated_hints = self._hint_generator.generate(pairs_batch)
            else:
                generated_hints = asyncio.run(self._hint_generator.generate(pairs_batch))
            for pair_idx, pair in enumerate(pairs_batch):
                hints = generated_hints[pair_idx]
                cleared_hints = self._hint_filtering.filtering(hints)
                results.append(cleared_hints)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    question_answer_pairs_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        for idx, instance in enumerate(instances):
            instance.hints_from_strings(results[idx])
        return results


class AnswerAgnostic(Model):
    """
    Class for automatically generating hints for questions that are unaware of their answers `[43]`_.

    .. _[43]: https://dl.acm.org/doi/10.1145/3626772.3657855

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
    .. [43] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

    See Also
    --------
    :class:`AnswerAware` : Class for automatically generating hints for questions that are aware of their answers.
    """

    def __init__(self, model_name: str,
                 api_key: str = None,
                 base_url: str = 'https://api.together.xyz/v1',
                 num_of_hints: int = 5,
                 parse_llm_response: Callable[[str], List[str]] = None,
                 temperature: float = 0.7,
                 top_p: float = 1.0,
                 max_tokens: int = 512,
                 batch_size: int = 2,
                 checkpoint: bool = False,
                 checkpoint_step: int = 1,
                 enable_tqdm=False):
        """
        Initializes the AnswerAgnostic class with the specified large language model and configuration options `[44]`_.

        .. _[44]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        model_name : str
            The large language model to use for hint generation.
        api_key : str, optional
            Specifies the API key for the Together.ai platform, required for accessing and interacting with the model.
        base_url : str, default 'https://api.together.xyz/v1'
            Specifies the base URL for the API endpoints. This URL is used to construct full API request URLs.
        num_of_hints : int, default 5
            Number of hints to generate per instance.
        parse_llm_response : Callable[[str], List[str]], optional
            Function to parse the language model's output into a list of strings as hints.
        temperature : float, default 0.7
            Sampling temperature to control the diversity of generated content.
        top_p : float, default 1.0
            Nucleus sampling cutoff to control the randomness of hint generation.
        max_tokens : int, default 512
            Maximum number of tokens to generate for each hint.
        batch_size : int, default 2
            Number of instances processed in one batch during hint generation.
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
        If `api_key` is None, the generator will attempt to download the model and run it locally.

        If `parse_llm_response` is None, the generator will use the default function used in the `[45]`_.

        .. _[45]: https://dl.acm.org/doi/10.1145/3626772.3657855



        Examples
        --------
        >>> from hinteval.model import AnswerAgnostic
        >>> answer_agnostic = AnswerAgnostic(model_name='llama-3-8b', api_key='your_api_key', base_url='base_url', num_of_hints=5,
        ...                                     parse_llm_response=None, temperature=0.7, top_p=1.0, max_tokens=512, batch_size=2,
        ...                                     checkpoint=True, checkpoint_step=5, enable_tqdm=True)

        References
        ----------
        .. [44] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`AnswerAware` : Class for automatically generating hints for questions that are aware of their answers.
        """

        super().__init__(checkpoint, checkpoint_step, enable_tqdm)
        self._file_name = f'answer_agnostic_{model_name.replace("/", "_")}.pickle'
        self._api_key = api_key
        self._base_url = base_url
        self._num_of_hints = num_of_hints
        self._parse_llm_response = parse_llm_response
        self._temperature = temperature
        self._top_p = top_p
        self._max_tokens = max_tokens
        self.batch_size = batch_size
        self._model_name = model_name

        self._hint_filtering = Hint_Filtering()

        if self._api_key is None:
            self._pipeline = transformers.pipeline(
                "text-generation",
                model=self._model_name,
                device_map="auto"
            )
            self._pipeline.tokenizer.pad_token = self._pipeline.tokenizer.eos_token
            self._pipeline.tokenizer.pad_token_id = self._pipeline.tokenizer.eos_token_id

            self._hint_generator = Hint_Generation_Agnostic_Local(self._pipeline, self._num_of_hints, self._parse_llm_response,
                                                         self._temperature, self._top_p, self._max_tokens)
        else:
            self._hint_generator = Hint_Generation_Agnostic_API(self._base_url, self._api_key, self._model_name,
                                                       self._num_of_hints, self._parse_llm_response, self._temperature,
                                                       self._top_p, self._max_tokens)

    def generate(self, instances: List[Instance], **kwargs) -> List[List[str]]:
        """
        Generates hints for a list of instances using the configured large language model `[46]`_.

        .. _[46]: https://dl.acm.org/doi/10.1145/3626772.3657855

        Parameters
        ----------
        instances : List[Instance]
            A list of instances, where each instance contains a question.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        List[List[str]]
            A list of lists containing generated hints for each instance.

        Notes
        -----
        This function stores the generated hints as :class:`Hint` objects within the `instances` attribute of the :class:`Subset`.

        Examples
        --------
        >>> from hinteval.cores import Instance, Question
        >>> from hinteval.model import AnswerAgnostic
        >>>
        >>> answer_agnostic = AnswerAgnostic(model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        ...                                 api_key='your_api_key',
        ...                                 num_of_hints=2)
        >>> instance_1 = Instance(question=Question('What is the capital of Austria?'), answers=[], hints=[])
        >>> instance_2 = Instance(question=Question('Who was the president of USA in 2009?'), answers=[], hints=[])
        >>> instances = [instance_1, instance_2]
        >>> results = answer_agnostic.generate(instances)
        >>> print(results)
        # [
        #     ["The city you're looking for is located along the Danube River.",
        #      "It's a city famous for its grand palaces, opera houses, and classical music heritage."],
        #     ['The person who held the office in 2009 was the first African American to hold the position.',
        #      'The president who took office in 2009 was a member of the Democratic Party and served two consecutive terms from 2009 to 2017.']
        # ]

        Raises
        ------
        RuntimeError
            If hint generation fails due to model or API issues.

        References
        ----------
        .. [46] Jamshid Mozafari, Anubhav Jangra, and Adam Jatowt. 2024. TriviaHG: A Dataset for Automatic Hint Generation from Factoid Questions. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24). Association for Computing Machinery, New York, NY, USA, 2060–2070. https://doi.org/10.1145/3626772.3657855

        See Also
        --------
        :class:`AnswerAware` : Class for automatically generating hints for questions that are aware of their answers.
        """

        self._validate_input(instances)
        pairs = []
        for idx, instance in enumerate(instances):
            q = instance.question.question
            pairs.append(q)
        pairs_batches = [pairs[i:i + self.batch_size] for i in range(0, len(pairs), self.batch_size)]

        checkpoint_content = self._load_content_checkpoint()
        results = [] if checkpoint_content is None else checkpoint_content['results']
        item_counter = itertools.count(1)

        question_answer_pairs_stream = tqdm(pairs_batches, total=len(pairs_batches),
                                            desc=f'Generating hints using {self._model_name}') if self.enable_tqdm else pairs_batches
        for pairs_batch in question_answer_pairs_stream:
            _idx = next(item_counter)
            final_step = len(results) // self.batch_size
            if _idx <= final_step:
                continue
            if self._api_key is None:
                generated_hints = self._hint_generator.generate(pairs_batch)
            else:
                generated_hints = asyncio.run(self._hint_generator.generate(pairs_batch))
            for pair_idx, pair in enumerate(pairs_batch):
                hints = generated_hints[pair_idx]
                cleared_hints = self._hint_filtering.filtering(hints)
                results.append(cleared_hints)

            if (_idx % self.checkpoint_step == 0 or _idx == len(
                    question_answer_pairs_stream)) and self.checkpoint_path is not None:
                self._store_checkpoint({'results': results})

        for idx, instance in enumerate(instances):
            instance.hints_from_strings(results[idx])
        return results
