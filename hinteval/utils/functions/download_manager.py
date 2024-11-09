import spacy
import json
import os
import requests
import zipfile_deflate64 as zipfile
from tqdm import tqdm


class SpacyDownloader:
    @classmethod
    def download(cls, model_name):
        if model_name not in ['en_core_web_sm', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_trf']:
            raise ValueError(
                f'Invalid model name: "{model_name}".\n'
                'Please choose one of the following valid models:\n'
                '- en_core_web_sm: Small English model\n'
                '- en_core_web_lg: Large English model\n'
                '- en_core_web_md: Medium English model\n'
                '- en_core_web_trf: Transformer-based English model'
            )
        try:
            spacy.load(model_name)
        except Exception:
            spacy.cli.download(model_name)


class QC_Downloader:
    @classmethod
    def _download_model(cls, _path):
        os.makedirs(_path, exist_ok=True)

        url = 'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/question-classification/qc_model.zip?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, 'qc_model.zip'), 'wb') as file, tqdm(
                    desc='Downloading question classification model',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))

            with zipfile.ZipFile(os.path.join(_path, 'qc_model.zip'), 'r') as zip_ref:
                file_list = zip_ref.namelist()

                with tqdm(total=len(file_list), desc="Unzipping downloaded file", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, _path)
                        pbar.update(1)
            os.remove(os.path.join(_path, 'qc_model.zip'))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, force_download):
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'question-classification')
        output_model_name = os.path.join(_path, 'best_qc_model.pickle')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path)


class DatasetDownloader:
    @classmethod
    def _download_dataset(cls, _path, name):
        os.makedirs(_path, exist_ok=True)
        url = f'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/datasets/{name}.pickle?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, f'{name}.pickle'), 'wb') as file, tqdm(
                    desc=f'Downloading {name}',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def available_datasets(cls, update=False):
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'datasets')
        os.makedirs(_path, exist_ok=True)
        if not os.path.exists(os.path.join(_path, 'metadata.json')):
            update = True
        if update:
            url = 'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/datasets/metadata.json?download=true'
            response = requests.get(url)
            if response.status_code == 200:
                with open(os.path.join(_path, 'metadata.json'), 'wb') as file:
                    for data in response.iter_content(chunk_size=1024):
                        file.write(data)
            else:
                raise Exception(
                    f'Failed to download the file from {url}.\n'
                    f'Status code: {response.status_code}\n'
                    'Please check your internet connection and the URL, and try again.'
                )
        with open(os.path.join(_path, 'metadata.json'), mode='r') as f:
            available_datasets = json.load(f)
        return available_datasets

    @classmethod
    def download(cls, name, force_download):
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'datasets')
        available_datasets = cls.available_datasets(True)
        datasets_kv = [f'- {k}: {v["description"]}' for k, v in available_datasets.items()]
        if name not in available_datasets.keys():
            errors = '\n'.join(datasets_kv)
            raise FileNotFoundError(
                f'Invalid dataset name: "{name}".\n'
                'Please choose one of the following valid datasets:\n'
                f'{errors}'
            )
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'datasets')
        output_model_name = os.path.join(_path, f'{name}.pickle')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_dataset(_path, name)


class EmbeddingsDownloader:
    @classmethod
    def _download_vectors(cls, _path, name):
        os.makedirs(_path, exist_ok=True)

        if name == 'glove.42B':
            name = 'glove.42B.300d'
        url = f'https://nlp.stanford.edu/data/{name}.zip'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, f'{name}.zip'), 'wb') as file, tqdm(
                    desc=f'Downloading word embeddings of {name} for relevance evaluation',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))

            with zipfile.ZipFile(os.path.join(_path, f'{name}.zip'), 'r') as zip_ref:
                file_list = zip_ref.namelist()

                with tqdm(total=len(file_list), desc="Unzipping downloaded file", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, _path)
                        pbar.update(1)
            if name == 'glove.6B':
                os.remove(os.path.join(_path, f'glove.6B.50d.txt'))
                os.remove(os.path.join(_path, f'glove.6B.100d.txt'))
                os.remove(os.path.join(_path, f'glove.6B.200d.txt'))
            os.remove(os.path.join(_path, f'{name}.zip'))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, version, force_download):
        if version not in ['glove.6B', 'glove.42B']:
            raise ValueError(
                f'Invalid Glove word embedding version: "{version}".\n'
                'Please choose one of the following valid versions:\n'
                '- glove.6B: Contains 6 billion tokens with 50, 100, 200, and 300-dimensional vectors\n'
                '- glove.42B: Contains 42 billion tokens with 300-dimensional vectors'
            )
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'embeddings')
        output_model_name = os.path.join(_path, f'{version}.300d.txt')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_vectors(_path, version)


class RelevanceNonContextualDownloader:
    @classmethod
    def _download_model(cls, _path):
        os.makedirs(_path, exist_ok=True)

        url = 'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/relevance-non-contextual/APMPCNN.model?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, 'APMPCNN.model'), 'wb') as file, tqdm(
                    desc='Downloading non contextual model for relevance evaluation',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, force_download):
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'relevance-non-contextual')
        output_model_name = os.path.join(_path, 'APMPCNN.model')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path)


class RelevanceContextualDownloader:
    @classmethod
    def _download_model(cls, _path, name):
        os.makedirs(_path, exist_ok=True)

        url = f'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/relevance-contextual/{name}.zip?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, f'{name}.zip'), 'wb') as file, tqdm(
                    desc=f'Downloading {name} model for relevance evaluation',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))

            with zipfile.ZipFile(os.path.join(_path, f'{name}.zip'), 'r') as zip_ref:
                file_list = zip_ref.namelist()

                with tqdm(total=len(file_list), desc="Unzipping downloaded file", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, _path)
                        pbar.update(1)
            os.remove(os.path.join(_path, f'{name}.zip'))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, model, force_download):
        if model not in ['bert-base', 'roberta-large']:
            raise ValueError(
                f'Invalid model name: "{model}".\n'
                'Please choose one of the following valid models:\n'
                '- bert-base: BERT (Bidirectional Encoder Representations from Transformers) base model\n'
                '- roberta-large: RoBERTa (Robustly optimized BERT approach) large model'
            )
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'relevance-contextual', model)
        output_model_name = os.path.join(_path, 'pytorch_model.bin')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path, model)


class ReadabilityMLDownloader:
    @classmethod
    def _download_model(cls, _path, name):
        os.makedirs(_path, exist_ok=True)

        url = f'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/readability-ml/{name}.model?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, f'{name}.model'), 'wb') as file, tqdm(
                    desc=f'Downloading {name} model for readability evaluation',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, method, force_download):
        if method not in ['xgboost', 'random_forest']:
            raise ValueError(
                f'Invalid method name: "{method}".\n'
                'Please choose one of the following valid methods:\n'
                '- xgboost: An optimized gradient boosting algorithm that uses decision trees to improve model accuracy through boosting techniques. It is known for its high performance and flexibility.\n'
                '- random_forest: An ensemble learning method that creates multiple decision trees and merges their outputs to improve accuracy and control overfitting. It is robust and works well with various types of data.'
            )
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'readability-ml')
        output_model_name = os.path.join(_path, f'{method}.model')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path, method)


class ReadabilityNNDownloader:
    @classmethod
    def _download_model(cls, _path, name):
        os.makedirs(_path, exist_ok=True)

        url = f'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/readability-nn/{name}.zip?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, f'{name}.zip'), 'wb') as file, tqdm(
                    desc=f'Downloading {name} model for readability evaluation',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))

            with zipfile.ZipFile(os.path.join(_path, f'{name}.zip'), 'r') as zip_ref:
                file_list = zip_ref.namelist()

                with tqdm(total=len(file_list), desc="Unzipping downloaded file", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, _path)
                        pbar.update(1)
            os.remove(os.path.join(_path, f'{name}.zip'))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, model, force_download):
        if model not in ['bert-base', 'roberta-large']:
            raise ValueError(
                f'Invalid model name: "{model}".\n'
                'Please choose one of the following valid models:\n'
                '- bert-base: BERT (Bidirectional Encoder Representations from Transformers) base model\n'
                '- roberta-large: RoBERTa (Robustly optimized BERT approach) large model'
            )
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'readability-nn', model)
        output_model_name = os.path.join(_path, 'optimizer.pt')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path, model)


class ConvergenceNNDownloader:
    @classmethod
    def _download_model(cls, _path, name):
        os.makedirs(_path, exist_ok=True)

        url = f'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/convergence-nn/{name}.zip?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, f'{name}.zip'), 'wb') as file, tqdm(
                    desc=f'Downloading {name} model for end-to-end convergence evaluation',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))

            with zipfile.ZipFile(os.path.join(_path, f'{name}.zip'), 'r') as zip_ref:
                file_list = zip_ref.namelist()

                with tqdm(total=len(file_list), desc="Unzipping downloaded file", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, _path)
                        pbar.update(1)
            os.remove(os.path.join(_path, f'{name}.zip'))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, model, force_download):
        if model not in ['bert-base', 'roberta-large']:
            raise ValueError(
                f'Invalid model name: "{model}".\n'
                'Please choose one of the following valid models:\n'
                '- bert-base: BERT (Bidirectional Encoder Representations from Transformers) base model\n'
                '- roberta-large: RoBERTa (Robustly optimized BERT approach) large model'
            )
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'convergence-nn', model)
        output_model_name = os.path.join(_path, 'optimizer.pt')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path, model)


class ConvergenceSpecificityDownloader:
    @classmethod
    def _download_model(cls, _path, name):
        os.makedirs(_path, exist_ok=True)

        url = f'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/convergence-specificity/{name}.zip?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, f'{name}.zip'), 'wb') as file, tqdm(
                    desc=f'Downloading {name} model for specificity',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))

            with zipfile.ZipFile(os.path.join(_path, f'{name}.zip'), 'r') as zip_ref:
                file_list = zip_ref.namelist()

                with tqdm(total=len(file_list), desc="Unzipping downloaded file", unit="file") as pbar:
                    for file in file_list:
                        zip_ref.extract(file, _path)
                        pbar.update(1)
            os.remove(os.path.join(_path, f'{name}.zip'))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, model, force_download):
        if model not in ['bert-base', 'roberta-large']:
            raise ValueError(
                f'Invalid model name: "{model}".\n'
                'Please choose one of the following valid models:\n'
                '- bert-base: BERT (Bidirectional Encoder Representations from Transformers) base model\n'
                '- roberta-large: RoBERTa (Robustly optimized BERT approach) large model'
            )
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'convergence-specificity', model)
        output_model_name = os.path.join(_path, 'optimizer.pt')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path, model)


class ConvergenceLLMDownloader:
    @classmethod
    def _download_model(cls, _path):
        os.makedirs(_path, exist_ok=True)

        url = 'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/convergence-llm/together_models.json?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(os.path.join(_path, 'together_models.json'), 'wb') as file:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, force_download):
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'convergence-llm')
        output_model_name = os.path.join(_path, 'together_models.json')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path)


class FamiliarityFrequencyDownloader:
    @classmethod
    def _download_model(cls, _path):
        os.makedirs(_path, exist_ok=True)

        url = 'https://huggingface.co/JamshidJDMY/HintEval/resolve/main/familiarity-freq/word_frequency_normalized.json?download=true'
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))

            with open(os.path.join(_path, 'word_frequency_normalized.json'), 'wb') as file, tqdm(
                    desc='Downloading word frequencies for familiarity evaluation',
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    file.write(data)
                    progress_bar.update(len(data))
        else:
            raise Exception(
                f'Failed to download the file from {url}.\n'
                f'Status code: {response.status_code}\n'
                'Please check your internet connection and the URL, and try again.'
            )

    @classmethod
    def download(cls, force_download):
        _path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'familiarity-freq')
        output_model_name = os.path.join(_path, 'word_frequency_normalized.json')
        if not os.path.exists(output_model_name) or force_download:
            cls._download_model(_path)
