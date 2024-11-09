import os.path
import torch
import random
import numpy as np
import torch.nn as nn
import pickle
import logging
from transformers import logging as tr_logging
from transformers import AutoModel, AutoTokenizer, AutoConfig
from torch.utils.data import TensorDataset, DataLoader
from hinteval.utils.functions.download_manager import QC_Downloader
from tqdm import tqdm

logging.disable(logging.WARNING)
tr_logging.set_verbosity_error()


class Classifier(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout_rate=0.1):
        super(Classifier, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.cls_size = int(config.hidden_size)
        self.input_dropout = nn.Dropout(p=dropout_rate)
        self.fully_connected_layer = nn.Linear(self.cls_size, num_labels)

    def forward(self, input_ids, attention_mask):
        model_outputs = self.encoder(input_ids, attention_mask)
        encoded_cls = model_outputs.last_hidden_state[:, 0]
        encoded_cls_dp = self.input_dropout(encoded_cls)
        logits = self.fully_connected_layer(encoded_cls_dp)
        return logits, encoded_cls


class QC:
    def __init__(self, batch_size, force_download, enable_tqdm):

        seed_val = 213
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_val)
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._max_seq_length = 64
        self.batch_size = batch_size
        self.enable_tqdm = enable_tqdm
        self._out_dropout_rate = 0.1
        qc_path = os.path.join(os.environ['HINTEVAL_CACHE_DIR'], 'question-classification')
        self._output_model_name = os.path.join(qc_path, 'best_qc_model.pickle')
        QC_Downloader.download(force_download)
        self._model_name = 'roberta-large'
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        with open(os.path.join(qc_path, 'label_to_id.pickle'), mode='rb') as f:
            self._label_to_id_map = pickle.load(f)
        with open(os.path.join(qc_path, 'id_to_label.pickle'), mode='rb') as f:
            self._id_to_label_map = pickle.load(f)
        with open(os.path.join(qc_path, 'labels_dict.pickle'), mode='rb') as f:
            self._labels = pickle.load(f)

    def _generate_data_loader(self, questions, label_map, tokenizer):
        input_ids = []
        input_mask_array = []
        label_id_array = []

        for (text, label) in questions:
            encoded_sent = tokenizer.encode_plus(text, add_special_tokens=True, max_length=self._max_seq_length,
                                                 padding='max_length', truncation=True)
            input_ids.append(encoded_sent['input_ids'])
            input_mask_array.append(encoded_sent['attention_mask'])

            id = -1
            if label in label_map:
                id = label_map[label]
            label_id_array.append(id)

        input_ids = torch.tensor(input_ids)
        input_mask_array = torch.tensor(input_mask_array)
        label_id_array = torch.tensor(label_id_array, dtype=torch.long)

        dataset = TensorDataset(input_ids, input_mask_array, label_id_array)
        return DataLoader(dataset, batch_size=self.batch_size)

    def predict(self, texts, questions):
        import __main__
        setattr(__main__, "Classifier", Classifier)
        best_model: Classifier = torch.load(self._output_model_name)

        my_list = [(question, '_') for question in questions]
        if self.enable_tqdm:
            my_data_loader = tqdm(self._generate_data_loader(my_list, self._label_to_id_map, self._tokenizer),
                                  desc='Detecting question types')
        else:
            my_data_loader = self._generate_data_loader(my_list, self._label_to_id_map, self._tokenizer)
        for batch_idx, batch in enumerate(my_data_loader):
            b_input_ids = batch[0].to(self._device)
            b_input_mask = batch[1].to(self._device)

            with torch.no_grad():
                logits, _ = best_model(b_input_ids, b_input_mask)

            _, preds = torch.max(logits, 1)
            for ex_id in range(len(b_input_mask)):
                predicted_label = self._id_to_label_map[preds[ex_id].item()]
                coarse_lbl, fine_lbl = tuple(predicted_label.split(':'))
                idx = batch_idx * self.batch_size + ex_id
                texts[idx].question_type['major'] = f'{coarse_lbl}:{self._labels["short_to_desc"][coarse_lbl]}'
                texts[idx].question_type['minor'] = f'{fine_lbl}:{self._labels["labels"][coarse_lbl][fine_lbl]}'
