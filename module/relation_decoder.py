import torch
from torch import nn
import torch.nn.functional as F

from module import ModuleWithDevice


class DistMultDecoder(ModuleWithDevice):
    def __init__(self, num_relations, num_dim, embed=None):
        super(DistMultDecoder, self).__init__()
        if embed is not None:
            self.emb_relation = nn.Parameter(torch.from_numpy(embed),
                                             requires_grad=True)
        else:
            self.emb_relation = nn.Parameter(torch.zeros(num_relations, num_dim),
                                             requires_grad=True)
            nn.init.xavier_uniform_(self.emb_relation,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, emb_entity, triples):
        # Make sure data is on right device
        emb_entity, triples = self.assure_device(emb_entity, triples)

        s = emb_entity[triples[:, 0]]
        r = self.emb_relation[triples[:, 1]]
        o = emb_entity[triples[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def inference(self, emb_entity, s, r, o):
        emb_entity, s, r, o = self.assure_device(emb_entity, s, r, o)
        # Calculate scores of triples including corrupted s / o
        emb_s = emb_entity[s]
        emb_o = emb_entity[o]
        emb_r = self.emb_relation[r]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sum(emb_triplet, dim=1)
        return scores

    def get_loss(self, scores, labels):
        scores, labels = self.assure_device(scores, labels)
        return F.binary_cross_entropy_with_logits(scores, labels)

    def reglurization(self):
        return torch.pow(self.emb_relation, 2).mean()


class SimplEDecoder(ModuleWithDevice):
    def __init__(self, num_relations, num_dim):
        super(SimplEDecoder, self).__init__()
        self.emb_relation = nn.Parameter(torch.zeros(num_relations, num_dim),
                                         requires_grad=True)
        self.inv_emb_relation = nn.Parameter(torch.zeros(num_relations, num_dim),
                                             requires_grad=True)
        nn.init.xavier_uniform_(
            self.emb_relation, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(
            self.inv_emb_relation, gain=nn.init.calculate_gain('relu'))

    def forward(self, emb_entity, triples):
        # Make sure data is on right device
        emb_entity, triples = self.assure_device(emb_entity, triples)

        s = emb_entity[triples[:, 0]]
        r = self.emb_relation[triples[:, 1]]
        r_inv = self.inv_emb_relation[triples[:, 1]]
        o = emb_entity[triples[:, 2]]

        return (torch.sum(s * r * o, -1) + torch.sum(s * r_inv * o, -1)) / 2

    def inference(self, emb_entity, s, r, o):
        emb_entity, s, r, o = self.assure_device(emb_entity, s, r, o)
        # Calculate scores of triples including corrupted s / o
        emb_s = emb_entity[s]
        emb_o = emb_entity[o]
        emb_r = self.emb_relation[r]
        emb_r_inv = self.inv_emb_relation[r]
        emb_triplet = emb_s * emb_r * emb_o
        emb_triplet_inv = emb_s * emb_r_inv * emb_o
        scores = (torch.sum(emb_triplet, -1) +
                  torch.sum(emb_triplet_inv, -1)) / 2
        return scores

    def get_loss(self, scores, labels):
        scores, labels = self.assure_device(scores, labels)
        return F.binary_cross_entropy_with_logits(scores, labels)

    def reglurization(self):
        return torch.pow(self.emb_relation, 2).mean() + torch.pow(self.inv_emb_relation, 2).mean()


class TransEDecoder(ModuleWithDevice):
    def __init__(self, num_relations, num_dim, embed=None):
        super(TransEDecoder, self).__init__()
        if embed is not None:
            self.emb_relation = nn.Parameter(torch.from_numpy(embed),
                                             requires_grad=True)
        else:
            self.emb_relation = nn.Parameter(torch.zeros(num_relations, num_dim),
                                             requires_grad=True)
            nn.init.xavier_uniform_(
                self.emb_relation, gain=nn.init.calculate_gain('relu'))

    def forward(self, emb_entity, triples):
        # Make sure data is on right device
        emb_entity, triples = self.assure_device(emb_entity, triples)
        s = emb_entity[triples[:, 0]]
        r = self.emb_relation[triples[:, 1]]
        o = emb_entity[triples[:, 2]]

        return 1 - torch.pow(s + r - o, 2).sum(dim=1)

    def inference(self, emb_entity, s, r, o):
        emb_entity, s, r, o = self.assure_device(emb_entity, s, r, o)
        # Calculate scores of triples including corrupted s / o
        emb_s = emb_entity[s]
        emb_o = emb_entity[o]
        emb_r = self.emb_relation[r]
        emb_triple = emb_s + emb_r - emb_o

        return 1 - torch.pow(emb_triple, 2).sum(dim=1)

    def get_loss(self, scores, labels):
        scores, labels = self.assure_device(scores, labels)
        return F.binary_cross_entropy_with_logits(scores, labels)

    def reglurization(self):
        return torch.pow(self.emb_relation, 2).mean()


class CompDecoder(ModuleWithDevice):
    def __init__(self, score_func):
        super(CompDecoder, self).__init__()
        self.score_func = score_func

    def forward(self, emb_tuple, triples):
        full_emb_node, emb_node, emb_rel = self.assure_device(*emb_tuple)
        triples = self.assure_device(triples)
        s = emb_node
        r = emb_rel
        o = full_emb_node[triples[:, 2]]

        import pdb
        pdb.set_trace()

        # Score functions
        if self.score_func == 'transe':
            return 1 - torch.pow(s + r - o, 2).sum(dim=1)
        elif self.score_func == 'distmult':
            return torch.sum(s * r * o, dim=1)
        else:
            raise NotImplementedError

    def inference(self, emb_tuple, s, r, o):
        emb_node, emb_rel, _, __ = self.assure_device(*emb_tuple)
        s, r, o = self.assure_device(s, r, o)
        emb_s = emb_node[s]
        emb_o = emb_node[o]
        emb_r = emb_rel[r]
        # Score functions
        if self.score_func == 'transe':
            return 1 - torch.pow(emb_s + emb_r - emb_o, 2).sum(dim=1)
        elif self.score_func == 'distmult':
            return torch.sum(emb_s * emb_r * emb_o, dim=1)
        else:
            raise NotImplementedError

    def get_loss(self, scores, labels):
        scores, labels = self.assure_device(scores, labels)
        return F.binary_cross_entropy_with_logits(scores, labels)


def load_relation_decoder(params, dataset, embed=None):
    # Load relation decoder
    relation_decoder_name = params['relation_decoder']['name']
    n_hidden = params['relation_decoder']['n_hidden']

    if relation_decoder_name == 'distmult':
        relation_decoder = DistMultDecoder(
            dataset.num_relations, n_hidden, embed)
    elif relation_decoder_name == 'simple':
        relation_decoder = SimplEDecoder(dataset.num_relations, n_hidden)
    elif relation_decoder_name == 'transe':
        relation_decoder = TransEDecoder(
            dataset.num_relations, n_hidden, embed)
    elif relation_decoder_name == 'comp':
        score_func = params['relation_decoder']['details']['score_func']
        relation_decoder = CompDecoder(score_func)
    else:
        raise NotImplementedError()

    return relation_decoder
