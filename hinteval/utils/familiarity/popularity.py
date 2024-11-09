import spacy
import requests
import numpy as np
from faker import Faker
from hinteval.utils.familiarity.pageviews import PageviewsClient
from hinteval.cores.dataset_core import Entity
from datetime import datetime


class Popularity:
    def __init__(self, spacy_pipeline):
        self.nlp = spacy.load(spacy_pipeline)
        self.faker = Faker()
        self.id_counter = 0

    @staticmethod
    def _clear_hint(hint: str):
        idx = hint.find('[^')
        if idx >= 0:
            hint = hint[:idx] + '.'
        return hint

    def _sent_entities(self, sentence: str):
        valid_entities = ['PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW',
                          'LANGUAGE']
        doc = self.nlp(sentence)
        entities = []
        for ent in doc.ents:
            if ent.label_ in valid_entities:
                entities.append(Entity(ent.text, ent.label_, ent.start_char, ent.end_char))
        return entities

    def _init_requests(self, q_id):
        self._session = requests.Session()
        self._user_agent = f'<{self.faker.company_email()}> {q_id} analyzer'

    def _find_similar_titles(self, title):
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": title
        }

        request = self._session.get(url='https://en.wikipedia.org/w/api.php', params=params)
        request_json = request.json()

        if request_json['query']['search']:
            return [page['title'] for page in request_json['query']['search']]
        else:
            return None

    def _extract_views(self, entities):
        if len(entities) == 0:
            return dict()
        most_similar_entities = set()
        for entity in entities:
            top_result = self._find_similar_titles(entity.entity)
            entity.metadata['wikipedia_page_title'] = top_result[0].replace(' ',
                                                                            '_') if top_result is not None else None
            if top_result is None:
                continue
            most_similar_entities.add(top_result[0])

        views_dict = dict()
        if len(most_similar_entities) > 0:
            try:
                p = PageviewsClient(user_agent=self._user_agent, parallelism=len(entities))
                views = p.article_views('en.wikipedia.org', list(most_similar_entities), granularity='monthly',
                                        start='20150101', end='20231231')
            except Exception as e:
                value_dict = dict()
                for entity in list(most_similar_entities):
                    value_dict[entity.replace(' ', '_')] = 0
                views = {datetime(2015, 1, 1): value_dict}

            views_dict_list = {}
            for entity_dict in views.values():
                for key in entity_dict.keys():
                    if key not in views_dict_list:
                        views_dict_list[key] = []
                    views_dict_list[key].append(entity_dict[key] if entity_dict[key] is not None else -1)

            for key in views_dict_list.keys():
                valid_months = [num for num in views_dict_list[key] if num >= 0]
                if len(valid_months) == 0:
                    views_dict[key] = 0
                else:
                    views_dict[key] = int(sum(valid_months) / len(valid_months))

        for entity in entities:
            if entity.metadata['wikipedia_page_title'] is None:
                entity.metadata['wiki_views_per_month'] = -1
            else:
                entity.metadata['wiki_views_per_month'] = views_dict[entity.metadata['wikipedia_page_title']]
        return views_dict

    def _normalize_and_remove_outliers(self, pops):
        pops = [(idx, popularity) for idx, popularity in enumerate(pops)]
        pops = sorted(pops, key=lambda x: x[1])
        pops_arr = np.array([p[1] for p in pops])
        min_val = 0
        max_val = 57837
        scaled_data = (pops_arr - min_val) / (max_val - min_val)
        scaled_data = np.where(scaled_data > 1.0, 1.0, scaled_data)
        scaled_data = np.where(scaled_data < 0.0, 0.0, scaled_data)

        pops = np.array(pops, dtype=np.float64)
        pops = np.insert(pops, 2, scaled_data, axis=1)
        pops_list = pops.tolist()
        pops_list = sorted([(int(p[0]), round(p[2], 3)) for p in pops_list], key=lambda x: x[0])
        pops_list = [p[1] for p in pops_list]
        return pops_list

    def normalize(self, sent_popularity):
        pops = []
        for itm in sent_popularity.keys():
            pops.append(sent_popularity[itm])
        pops = self._normalize_and_remove_outliers(pops)
        results_normalized = dict()
        for itm in sent_popularity.keys():
            results_normalized[itm] = pops.pop(0)
        return results_normalized

    def popularity(self, sentence: str, is_word: bool):
        self.id_counter += 1
        self._init_requests(self.id_counter)
        if is_word:
            sent_entities = [Entity(sentence, 'OTHER', 0, len(sentence))]
        else:
            sent_entities = self._sent_entities(sentence)
        sent_popularity = self._extract_views(sent_entities)
        sent_normalized = dict()
        if len(sent_popularity) > 0:
            sent_normalized = self.normalize(sent_popularity)
        for entity in sent_entities:
            if entity.metadata['wikipedia_page_title'] is None:
                entity.metadata['normalized_views'] = -1
            else:
                entity.metadata['normalized_views'] = sent_normalized[entity.metadata['wikipedia_page_title']]
        return sent_entities, sent_normalized
