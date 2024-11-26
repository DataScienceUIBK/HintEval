import os
import json
import pickle
import gzip
import random as rnd
from prettytable import PrettyTable
from typing import Dict, Literal, Union
from hinteval.cores.dataset_core import Entity, Metric, Question, Answer, Hint, Instance, Subset
from hinteval.utils.identify_functions import identify_entities, identify_question_type
from hinteval.utils.functions.download_manager import DatasetDownloader


class _CoreEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Entity, Metric, Question, Answer, Hint, Instance, Subset)):
            return obj.to_dict()
        return super().default(obj)


class Dataset:
    """
    A class to represent a dataset, including its subsets and associated metadata.

    Attributes
    ----------
    name : str
        The name of the dataset.
    url : str
        The URL where the dataset can be accessed.
    version : str
        The version of the dataset.
    description : str
        A description of the dataset.
    metadata : dict[str, Union[str,int, float]]
        Additional metadata about the dataset.
    """

    def __init__(self, name: str = None, url: str = None, version: str = None, description: str = None,
                 metadata: Dict[str, Union[str, int, float]] = None):
        """
        Initializes a new Dataset instance.

        Parameters
        ----------
        name : str, optional
            The name of the dataset (default is a random name).
        url : str, optional
            The URL where the dataset can be accessed (default is None).
        version : str, optional
            The version of the dataset (default is None).
        description : str, optional
            A description of the dataset (default is None).
        metadata : dict[str, Union[str,int, float]], optional
            Additional metadata about the dataset (default is an empty dictionary).

        Examples
        --------
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset(name='example_dataset', version='1.0', description='An example dataset.')
        >>> print(dataset.name)
        # example_dataset

        See Also
        --------
        from_dict :
            Creates a Dataset object from a dictionary.

        load_json :
            Loads a Dataset instance from a JSON file.

        load_or_download_dataset :
            Loads a dataset from a local cache or downloads it if not available locally.
        """

        if name is None:
            self.name = f'dataset_{rnd.randint(10000, 99999)}'
        else:
            self.name = name
        self.url = url
        self.version = version
        self.description = description
        self.metadata: Dict[str, Union[str, int, float]] = metadata if metadata is not None else {}
        self._subsets: Dict[str, Subset] = dict()

    @classmethod
    def available_datasets(cls, show_info=False, update=False) -> Dict:
        """
        Retrieves a list of available datasets to download.

        Parameters
        ----------
        update : bool, optional
            Whether to update the dataset list from the remote source (default is False).
        show_info : bool, optional
            Whether to show the information about the available datasets (default is False).

        Returns
        -------
        dict
            A dictionary of available datasets.

        Raises
        ------
        Exception
            If failed to load datasets from the remote source.

        Examples
        --------
        >>> from hinteval import Dataset
        >>>
        >>> available_datasets = Dataset.available_datasets()
        >>> print(available_datasets['triviahg']['description'])
        # TriviaHG is an extensive dataset crafted specifically for hint generation in question answering.

        See Also
        --------
        download_and_load_dataset :
            Loads a dataset from a local cache or downloads it if not available locally.

        """

        available_datasets_json = DatasetDownloader.available_datasets(update)
        if show_info:
            for dataset in available_datasets_json:
                print()
                print(f"Name: {available_datasets_json[dataset]['name']}")
                print(f"Version: {available_datasets_json[dataset]['version']}")
                print(f"URL: {available_datasets_json[dataset]['url']}")
                print(f"Description: {available_datasets_json[dataset]['description']}")
                table = PrettyTable(['Subset', 'Num. of Questions', 'Num. of Hints'])
                for subset in available_datasets_json[dataset]['subsets']:
                    table.add_row([subset, available_datasets_json[dataset]['subsets'][subset]['questions'],
                                   available_datasets_json[dataset]['subsets'][subset]['hints']])
                print(table)
        return available_datasets_json

    def add_subset(self, subset: Subset):
        """
        Adds a subset to the dataset.

        Parameters
        ----------
        subset : Subset
            The subset to be added to the dataset.

        Raises
        ------
        ValueError
            If the subset name already exists in the dataset.

        Examples
        --------
        >>> from hinteval.cores import Subset
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset('Hint_Dataset')
        >>> subset = Subset(name='training_set')
        >>> dataset.add_subset(subset)
        >>> print(dataset["training_set"].name)
        # training_set

        See Also
        --------
        remove_subset :
            Removes a subset from the dataset.

        get_subset :
            Retrieves a subset by name.
        """

        if subset.name in self._subsets.keys():
            raise ValueError(f'The subset "{subset.name}" is already.')
        subset._dataset_object = self
        self._subsets[subset.name] = subset

    def remove_subset(self, name: str):
        """
        Removes a subset from the dataset.

        Parameters
        ----------
        name : str
            The name of the subset to be removed.

        Raises
        ------
        ValueError
            If the subset name does not exist in the dataset.

        Examples
        --------
        >>> from hinteval.cores import Subset
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset('Hint_Dataset')
        >>> subset = Subset(name='training_set')
        >>> dataset.add_subset(subset)
        >>> print(dataset.get_subsets_name())
        # ['training_set']
        >>> dataset.remove_subset('training_set')
        >>> # del dataset['training_set']
        >>> print(dataset.get_subsets_name())
        # []

        See Also
        --------
        add_subset :
            Adds a subset to the dataset.

        get_subset :
            Retrieves a subset by name.
        """

        if name not in self._subsets.keys():
            raise ValueError(f'The subset "{name}" is not in the subsets.')
        del self._subsets[name]

    def get_subset(self, name: str):
        """
        Retrieves a subset by name.

        Parameters
        ----------
        name : str
            The name of the subset to retrieve.

        Returns
        -------
        Subset
            The subset associated with the given name.

        Raises
        ------
        ValueError
            If the subset name does not exist in the dataset.

        Examples
        --------
        >>> from hinteval.cores import Subset
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset('Hint_Dataset')
        >>> subset = Subset(name='training_set')
        >>> dataset.add_subset(subset)
        >>> subset = dataset.get_subset('training_set')
        >>> # subset = dataset['training_set']
        >>> print(subset.name)
        # training_set

        See Also
        --------
        add_subset :
            Adds a subset to the dataset.

        remove_subset :
            Removes a subset from the dataset.

        get_subsets_name :
            Retrieves the names of all subsets in the dataset.

        get_subsets :
            Retrieves all subsets in the dataset.

        """

        if name not in self._subsets.keys():
            raise ValueError(f'The subset "{name}" is not in the subsets.')
        return self._subsets[name]

    def get_subsets_name(self):
        """
        Retrieves the names of all subsets in the dataset.

        Returns
        -------
        list[str]
            A list of all subset names in the dataset.

        Examples
        --------
        >>> from hinteval.cores import Subset
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset('Hint_Dataset')
        >>> training = Subset(name='training_set')
        >>> validation = Subset(name='validation_set')
        >>> dataset.add_subset(training)
        >>> dataset.add_subset(validation)
        >>> subset_names = dataset.get_subsets_name()
        >>> print(subset_names)
        # ['training_set', 'validation_set']

        See Also
        --------
        add_subset :
            Adds a subset to the dataset.

        remove_subset :
            Removes a subset from the dataset.

        get_subset :
            Retrieves a subset by name.

        get_subsets :
            Retrieves all subsets in the dataset.

        """

        return list(self._subsets.keys())

    def get_subsets(self):
        """
        Retrieves all subsets in the dataset.

        Returns
        -------
        list[Subset]
            A list of all subsets in the dataset.

        Examples
        --------
        >>> from hinteval.cores import Subset
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset('Hint_Dataset')
        >>> training = Subset(name='training_set')
        >>> validation = Subset(name='validation_set')
        >>> dataset.add_subset(training)
        >>> dataset.add_subset(validation)
        >>> subsets = dataset.get_subsets()
        >>> print([subset.name for subset in subsets])
        # ['training_set', 'validation_set']

        See Also
        --------
        add_subset :
            Adds a subset to the dataset.

        remove_subset :
            Removes a subset from the dataset.

        get_subset :
            Retrieves a subset by name.

        get_subsets_name:
            Retrieves the names of all subsets in the dataset.

        """

        return list(self._subsets.values())

    def to_dict(self):
        """
        Converts the Dataset instance into a dictionary.

        Returns
        -------
        dict[str, Any]
            A dictionary representation of the Dataset instance.

        Examples
        --------
        >>> from hinteval.cores import Subset
        >>> from hinteval import Dataset
        >>>
        >>> subset1 = Subset(name='training_set')
        >>> subset2 = Subset(name='validation_set')
        >>> dataset = Dataset(name='example_dataset')
        >>> dataset.add_subset(subset1)
        >>> dataset.add_subset(subset2)
        >>> dataset_dict = dataset.to_dict()
        >>> print(dataset_dict)
        # {
        #     'name': 'example_dataset',
        #     'version': None,
        #     'description': None,
        #     'url': None,
        #     'metadata': {},
        #     'subsets': {
        #         'training_set': {'name': 'training_set', ...},
        #         'validation_set': {'name': 'validation_set', ...}
        #     }
        # }

        See Also
        --------
        from_dict :
            Creates a Dataset instance from a dictionary.

        store_json :
            Stores the Dataset instance as a JSON file.

        store :
            Stores the Dataset instance.
        """

        ret_dict = {'name': self.name}
        ret_dict.update({'version': self.version})
        ret_dict.update({'description': self.description})
        ret_dict.update({'url': self.url})
        ret_dict.update({'metadata': self.metadata})
        ret_dict.update({'subsets': {key: val.to_dict() for key, val in self._subsets.items()}})
        return ret_dict

    def store_json(self, path):
        """
        Stores the Dataset instance as a JSON file.

        Parameters
        ----------
        path : str
            The file path to store the JSON representation of the Dataset instance.

        Examples
        --------
        >>> from hinteval.cores import Subset
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset('Hint_Dataset')
        >>> training = Subset(name='training_set')
        >>> validation = Subset(name='validation_set')
        >>> dataset.add_subset(training)
        >>> dataset.add_subset(validation)
        >>> dataset.store_json('./dataset.json')

        See Also
        --------
        load_json :
            Loads a Dataset instance from a JSON file.

        load :
            Loads a Dataset instance from a file.
        """

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load_json(cls, path):
        """
        Loads a Dataset instance from a JSON file.

        Parameters
        ----------
        path : str
            The file path to load the JSON representation of the Dataset instance.

        Returns
        -------
        Dataset
            A new Dataset object initialized from the JSON file.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        KeyError
            If required keys are missing in the JSON file.

        Examples
        --------
        >>> from hinteval import Dataset
        >>>
        >>> dataset_loaded = Dataset.load_json('./dataset.json')
        >>> print(dataset_loaded.name)
        # Hint_Dataset

        See Also
        --------
        store_json
            Stores the Dataset instance as a JSON file.

        store :
            Stores the Dataset instance.
        """

        with open(path, 'r') as f:
            data = json.load(f)
        name = data['name']
        version = data['version']
        description = data['description']
        url = data['url']
        metadata = data['metadata']
        new_cls = cls(name=name, url=url, version=version, description=description, metadata=metadata)
        for subset in data['subsets'].values():
            new_cls.add_subset(Subset.from_dict(subset))
        return new_cls

    def store(self, path):
        """
        Stores the Dataset instance.

        Parameters
        ----------
        path : str
            The file path to store the Dataset instance.

        Examples
        --------
        >>> from hinteval.cores import Subset
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset('Hint_Dataset')
        >>> training = Subset(name='training_set')
        >>> validation = Subset(name='validation_set')
        >>> dataset.add_subset(training)
        >>> dataset.add_subset(validation)
        >>> dataset.store('./dataset.pickle')

        See Also
        --------
        load_json :
            Loads a Dataset instance from a JSON file.

        load :
            Loads a Dataset instance from a file.
        """

        with gzip.open(path, 'wb') as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        """
        Loads a Dataset instance from a file.

        Parameters
        ----------
        path : str
            The file path to load the Dataset instance.

        Returns
        -------
        Dataset
            A new Dataset object initialized from the loaded file.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        Exception
            If the specified file is correpted.

        Examples
        --------
        >>> from hinteval import Dataset
        >>>
        >>> dataset_loaded = Dataset.load('./dataset.pickle')
        >>> print(dataset_loaded.name)
        # Hint_Dataset

        See Also
        --------
        store_json :
            Stores the Dataset instance as a JSON file.

        store :
            Stores the Dataset instance.
        """

        with gzip.open(path, 'rb') as f:
            data = pickle.load(f)
        name = data['name']
        version = data['version']
        description = data['description']
        url = data['url']
        metadata = data['metadata']
        new_cls = cls(name=name, url=url, version=version, description=description, metadata=metadata)
        for subset in data['subsets'].values():
            new_cls.add_subset(Subset.from_dict(subset))
        return new_cls

    def prepare_dataset(self, fill_question_types=True, fill_entities=False, batch_size: int = 256,
                        spacy_pipeline: Literal[
                            'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'] = 'en_core_web_sm',
                        qc_model_force_download=False, enable_tqdm=False):
        """
        Prepares the dataset by detecting question types for questions and entities for questions, hints, and answers.

        Parameters
        ----------
        fill_question_types : bool, optional
            Whether to detect question types (default is True).
        fill_entities : bool, optional
            Whether to detect entities (default is False).
        batch_size : int, optional
            The batch size for processing (default is 256).
        spacy_pipeline : {'en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf'}, optional
            The spaCy pipeline to use for entity recognition (default is 'en_core_web_sm').
        qc_model_force_download : bool, optional
            Whether to force download the question classification model (default is False).
        enable_tqdm : bool, optional
            Whether to enable tqdm progress bar (default is False).

        Examples
        --------
        >>> from hinteval.cores import Subset, Instance
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset("Hint_Dataset")
        >>> subset = Subset(name="training_set")
        >>> dataset.add_subset(subset)
        >>> instance = Instance.from_strings("What is the capital of Austria?",
        ...                 ["Vienna"],
        ...                 ["This city, once home to Mozart and Beethoven."])
        >>> subset.add_instance(instance, "q_1")
        >>> dataset.prepare_dataset(fill_question_types=True, fill_entities=True,
        ...                         batch_size=64,
        ...                         spacy_pipeline='en_core_web_sm',
        ...                         qc_model_force_download=False,
        ...                         enable_tqdm=False)
        >>> print(instance.question)
        # {
        #     "question": "What is the capital of Austria?",
        #     "question_type": {
        #         "major": "LOC:LOCATION",
        #         "minor": "other:Other location"
        #     },
        #     "entities": [
        #         {
        #             "entity": "Austria",
        #             "ent_type": "GPE",
        #             "start_index": 23,
        #             "end_index": 30,
        #             "metadata": {}
        #         }
        #     ],
        #     "metrics": {},
        #     "metadata": {}
        # }

        See Also
        --------
        utils.identify_functions.identify_entities :
            Function to detect entities in instances.
        utils.identify_functions.identify_question_type :
            Function to detect question types for questions.
        """

        instances = []
        for subset in self.get_subsets():
            for instance in subset.get_instances():
                instances.append(instance)
        if fill_entities:
            identify_entities(instances, batch_size=batch_size, spacy_pipeline=spacy_pipeline, enable_tqdm=enable_tqdm)
        if fill_question_types:
            questions = []
            for instance in instances:
                questions.append(instance.question)
            identify_question_type(questions, batch_size=batch_size, force_download=qc_model_force_download,
                                   enable_tqdm=enable_tqdm)

    def __getitem__(self, s_name):
        return self.get_subset(s_name)

    def __delitem__(self, s_name):
        self.remove_subset(s_name)

    def __len__(self):
        return len(self._subsets)

    def __eq__(self, other):
        if isinstance(other, Dataset):
            return self.name == other.name and self.version == other.version and self.description == other.description and self.url == other.url and self.metadata == other.metadata and self._subsets == other._subsets
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return json.dumps(self.to_dict(), cls=_CoreEncoder, indent=4)

    @classmethod
    def download_and_load_dataset(cls, name, force_download=False):
        """
        Loads a dataset from a local cache or downloads it if not available locally.

        Parameters
        ----------
        name : str
            The name of the dataset to load.
        force_download : bool, optional
            Whether to force download the dataset even if it already exists locally (default is False).

        Returns
        -------
        Dataset
            A new Dataset object initialized from the loaded data.

        Raises
        ------
        FileNotFoundError
            If the dataset name does not exist in the available datasets.
        Exception
            If the dataset fails to download.

        Examples
        --------
        >>> from hinteval import Dataset
        >>>
        >>> dataset = Dataset.download_and_load_dataset('triviahg', force_download=True)
        >>> print(dataset.description)
        # TriviaHG is an extensive dataset crafted specifically for hint generation in question answering.

        See Also
        --------
        available_datasets : 
            Retrieves a list of available datasets to download.

        load_json :
            Loads a Dataset instance from a JSON file.

        load :
            Loads a Dataset instance from a file.

        """

        name = name.lower()
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'datasets', f'{name}.pickle')
        if not os.path.exists(_path) or force_download:
            DatasetDownloader.download(name, force_download)
        return cls.load(_path)
