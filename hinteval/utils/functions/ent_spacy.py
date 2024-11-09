import spacy
from tqdm import tqdm
from hinteval.cores.dataset_core import Entity
from hinteval.utils.functions.download_manager import SpacyDownloader


class EntitySpacy:

    def __init__(self, batch_size, model_name, enable_tqdm):

        SpacyDownloader.download(model_name)
        self._nlp = spacy.load(model_name)
        self._valid_entities = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW',
                                'LANGUAGE']
        self.batch_size = batch_size
        self.enable_tqdm = enable_tqdm

    def predict(self, texts, sentences):

        if self.enable_tqdm:
            docs = tqdm(enumerate(self._nlp.pipe(sentences, n_process=8, batch_size=self.batch_size)),
                        desc='Detecting entities', total=len(sentences))
        else:
            docs = enumerate(self._nlp.pipe(sentences, n_process=8, batch_size=self.batch_size))
        for idx, doc in docs:
            for ent in doc.ents:
                if ent.label_ in self._valid_entities:
                    new_entity = Entity(ent.text, ent.label_, ent.start_char, ent.end_char)
                    if new_entity not in texts[idx].entities:
                        texts[idx].entities.append(new_entity)
