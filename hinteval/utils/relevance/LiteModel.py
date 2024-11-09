import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PairwiseConv(nn.Module):

    def __init__(self, model):
        super(PairwiseConv, self).__init__()
        self.convModel = model
        self.dropout = nn.Dropout(model.dropout)
        self.linearLayer = nn.Linear(model.n_hidden, 1)
        self.sigmoid = nn.Sigmoid()
        self.posModel = self.convModel
        self.negModel = self.convModel

    def forward(self, input):
        pos = self.posModel(input[0].sentence_1, input[0].sentence_2, input[0].ext_feats)
        neg = self.negModel(input[1].sentence_1, input[1].sentence_2, input[1].ext_feats)
        pos = self.dropout(pos)
        neg = self.dropout(neg)
        pos = self.linearLayer(pos)
        neg = self.linearLayer(neg)
        pos = self.sigmoid(pos)
        neg = self.sigmoid(neg)
        combine = torch.cat([pos, neg], 1)
        return combine


class MPCNNLite(nn.Module):
    def __init__(self, n_word_dim, n_holistic_filters, n_per_dim_filters, filter_widths, hidden_layer_units,
                 dropout, ext_feats, attention, wide_conv, embedding):
        super(MPCNNLite, self).__init__()
        self.arch = 'MPCNN-Lite'
        self.n_word_dim = n_word_dim
        self.n_holistic_filters = n_holistic_filters
        # self.n_per_dim_filters = n_per_dim_filters
        self.filter_widths = filter_widths
        self.ext_feats = ext_feats
        self.attention = attention
        self.wide_conv = wide_conv
        self.n_hidden = hidden_layer_units
        self.dropout = dropout
        self.embedding = embedding

        self.in_channels = n_word_dim if attention == 'none' else 2 * n_word_dim

        self._add_layers()

        # compute number of inputs to first hidden layer
        n_feats = self._get_n_feats()

        self.final_layers = nn.Sequential(
            nn.Linear(n_feats, hidden_layer_units),
            nn.Tanh()
        )

    def _add_layers(self):
        holistic_conv_layers_max = []

        for ws in self.filter_widths:
            if np.isinf(ws):
                continue

            padding = ws - 1 if self.wide_conv else 0

            holistic_conv_layers_max.append(nn.Sequential(
                nn.Conv1d(self.in_channels, self.n_holistic_filters, ws, padding=padding),
                nn.Tanh()
            ))

        self.holistic_conv_layers_max = nn.ModuleList(holistic_conv_layers_max)

    def _get_n_feats(self):
        COMP_1_COMPONENTS_HOLISTIC, COMP_2_COMPONENTS = 2 + self.n_holistic_filters, 2
        n_feats_h = self.n_holistic_filters * COMP_2_COMPONENTS
        n_feats_v = (
            # comparison units from holistic conv for max pooling for non-infinite widths
                ((len(self.filter_widths) - 1) ** 2) * COMP_1_COMPONENTS_HOLISTIC +
                # comparison units from holistic conv for max pooling for infinite widths
                3
        )
        n_feats = n_feats_h + n_feats_v + self.ext_feats
        return n_feats

    def _get_blocks_for_sentence(self, sent):
        block_a = {}
        for ws in self.filter_widths:
            if np.isinf(ws):
                sent_flattened, sent_flattened_size = sent.contiguous().view(sent.size(0), 1, -1), sent.size(
                    1) * sent.size(2)
                block_a[ws] = {
                    'max': F.max_pool1d(sent_flattened, sent_flattened_size).view(sent.size(0), -1)
                }
                continue

            holistic_conv_out_max = self.holistic_conv_layers_max[ws - 1](sent)
            block_a[ws] = {
                'max': F.max_pool1d(holistic_conv_out_max, holistic_conv_out_max.size(2)).contiguous().view(-1,
                                                                                                            self.n_holistic_filters)
            }

        return block_a

    def _algo_1_horiz_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        regM1, regM2 = [], []
        for ws in self.filter_widths:
            x1 = sent1_block_a[ws]['max'].unsqueeze(2)
            x2 = sent2_block_a[ws]['max'].unsqueeze(2)
            if np.isinf(ws):
                x1 = x1.expand(-1, self.n_holistic_filters, -1)
                x2 = x2.expand(-1, self.n_holistic_filters, -1)
            regM1.append(x1)
            regM2.append(x2)

        regM1 = torch.cat(regM1, dim=2)
        regM2 = torch.cat(regM2, dim=2)

        # Cosine similarity
        comparison_feats.append(F.cosine_similarity(regM1, regM2, dim=2))
        # Euclidean distance
        pairwise_distances = []
        for x1, x2 in zip(regM1, regM2):
            dist = F.pairwise_distance(x1, x2).view(1, -1)
            pairwise_distances.append(dist)
        comparison_feats.append(torch.cat(pairwise_distances))

        return torch.cat(comparison_feats, dim=1)

    def _algo_2_vert_comp(self, sent1_block_a, sent2_block_a):
        comparison_feats = []
        for ws1 in self.filter_widths:
            x1 = sent1_block_a[ws1]['max']
            for ws2 in self.filter_widths:
                x2 = sent2_block_a[ws2]['max']
                if (not np.isinf(ws1) and not np.isinf(ws2)) or (np.isinf(ws1) and np.isinf(ws2)):
                    comparison_feats.append(F.cosine_similarity(x1, x2).unsqueeze(1))
                    comparison_feats.append(F.pairwise_distance(x1, x2).unsqueeze(1))
                    comparison_feats.append(torch.abs(x1 - x2))

        return torch.cat(comparison_feats, dim=1)

    def concat_attention(self, sent1, sent2):
        sent1_transposed = sent1.transpose(1, 2)

        attention_dot = torch.bmm(sent1_transposed, sent2)
        sent1_norms = torch.norm(sent1_transposed, p=2, dim=2, keepdim=True)
        sent2_norms = torch.norm(sent2, p=2, dim=1, keepdim=True)

        attention_norms = torch.bmm(sent1_norms, sent2_norms)
        attention_matrix = attention_dot / attention_norms
        attention_matrix = attention_dot / attention_norms

        sum_row = attention_matrix.sum(2)
        sum_col = attention_matrix.sum(1)

        attention_weight_vec1 = F.softmax(sum_row, 1)
        attention_weight_vec2 = F.softmax(sum_col, 1)

        attention_weighted_sent1 = attention_weight_vec1.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent1
        attention_weighted_sent2 = attention_weight_vec2.unsqueeze(1).expand(-1, self.n_word_dim, -1) * sent2
        attention_emb1 = torch.cat((attention_weighted_sent1, sent1), dim=1)
        attention_emb2 = torch.cat((attention_weighted_sent2, sent2), dim=1)
        return attention_emb1, attention_emb2

    def forward(self, sent1, sent2, ext_feats=None):

        sent1 = self.embedding(sent1).transpose(1, 2)
        sent2 = self.embedding(sent2).transpose(1, 2)

        # Attention
        sent1, sent2 = self.concat_attention(sent1, sent2)

        # Sentence modeling module
        sent1_block_a = self._get_blocks_for_sentence(sent1)
        sent2_block_a = self._get_blocks_for_sentence(sent2)

        # Similarity measurement layer
        feat_h = self._algo_1_horiz_comp(sent1_block_a, sent2_block_a)
        feat_v = self._algo_2_vert_comp(sent1_block_a, sent2_block_a)
        combined_feats = [feat_h, feat_v, ext_feats] if self.ext_feats else [feat_h, feat_v]
        feat_all = torch.cat(combined_feats, dim=1)

        preds = self.final_layers(feat_all)
        return preds
