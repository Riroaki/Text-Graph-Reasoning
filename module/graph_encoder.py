import torch
import torch.nn as nn
from torch.nn import functional as F
from dgl.nn.pytorch import RelGraphConv, GATConv
# from sklearn import decomposition

from module import ModuleWithDevice
from module.compgcn import CompGCNCov


class RelGraphConvolutionEncoder(ModuleWithDevice):
    def __init__(self, num_nodes, num_relations, num_hidden, num_bases,
                 num_hidden_layers=2, dropout=0.0, embed=None, embed_connect="residual"):
        super(RelGraphConvolutionEncoder, self).__init__()
        num_bases = None if num_bases < 0 else num_bases

        # Embedding layer
        if embed is not None:
            if embed.shape[1] > num_hidden:
                raise Exception('Pretrain embdding dimension mismatch:'
                                'required {}-d, but got {}-d'.format(num_hidden, embed.shape[1]))
                # svd = decomposition.TruncatedSVD(n_components=num_hidden)
                # embed = svd.fit_transform(embed)
                # embed = torch.tensor(embed, dtype=torch.float)
            self.emb_node = nn.Embedding.from_pretrained(embed)
        else:
            self.emb_node = nn.Embedding(num_nodes, num_hidden)

        # Register layers
        layers = []
        for i in range(num_hidden_layers):
            act = F.relu if i < num_hidden_layers - 1 else None
            layers.append(RelGraphConv(num_hidden, num_hidden, num_relations, "bdd",
                                       num_bases, activation=act, self_loop=True,
                                       dropout=dropout))
        self.layers = nn.ModuleList(layers)

        # Connction from node embedding to relational decoder
        self.emb_connect = embed_connect

    def forward(self, g, node_id, r, norm):
        # Make sure data is on right device
        g, node_id, r, norm = self.assure_device(g, node_id, r, norm)

        h = self.emb_node(node_id.squeeze())

        if self.emb_connect == 'embed':
            return h

        for layer in self.layers:
            h = layer(g, h, r, norm)

        if self.emb_connect == 'residual':
            return h + self.emb_node(node_id.squeeze())
        elif self.emb_connect == 'graph':
            return h
        else:
            raise NotImplementedError


class GraphAttentionEncoder(ModuleWithDevice):
    def __init__(self, num_nodes, num_layers, in_dim, num_hidden,
                 heads, feat_drop, attn_drop, negative_slope, residual,
                 embed=None, embed_connect="residual", graph=None):
        super(GraphAttentionEncoder, self).__init__()

        # Embedding layer
        if embed is not None:
            if embed.shape[1] > num_hidden:
                raise Exception('Pretrain embdding dimension mismatch:'
                                'required {}-d, but got {}-d'.format(num_hidden, embed.shape[1]))
                # svd = decomposition.TruncatedSVD(n_components=num_hidden)
                # embed = svd.fit_transform(embed)
                # embed = torch.tensor(embed, dtype=torch.float)
            self.emb_node = nn.Embedding.from_pretrained(embed)
        else:
            self.emb_node = nn.Embedding(num_nodes, num_hidden)

        # Hidden layers
        layers = []
        for idx in range(num_layers):
            activation = None if idx == num_layers - 1 else F.elu
            if idx == 0:
                layer = GATConv(in_dim, num_hidden, heads[idx], feat_drop, attn_drop,
                                negative_slope, False, activation)
            else:
                layer = GATConv(num_hidden * heads[idx - 1], num_hidden, heads[idx], feat_drop,
                                attn_drop,
                                negative_slope, residual, activation)
            layers.append(layer)
        self.gat_layers = nn.ModuleList(layers)

        # Record full graph for propagation
        self.graph = graph

        # Connction from node embedding to relational decoder
        self.emb_connect = embed_connect

    def forward(self, _, node_id, __, ___):
        # Make sure data is on right device
        g, node_id = self.assure_device(self.graph, h)

        h = self.emb_node(node_id.squeeze())

        if self.emb_connect == 'embed':
            return h

        for layer in self.layers:
            h = layer(g, h).flatten(1)

        if self.emb_connect == 'residual':
            return h + self.emb_node(node_id.squeeze())
        elif self.emb_connect == 'graph':
            return h
        else:
            raise NotImplementedError


class CompGraphConvolutionEncoder(ModuleWithDevice):
    def __init__(self, num_nodes, num_relations, num_bases, in_dim, num_hidden, out_dim,
                 num_layers=2, operation='corr', gcn_drop=0.0, conv_bias=True, embed=None,
                 embed_connect="residual", graph=None, edge_type=None, edge_norm=None):
        super(CompGraphConvolutionEncoder, self).__init__()

        # Embedding layer
        if embed is not None:
            if embed.shape[1] > in_dim:
                raise Exception('Pretrain embdding dimension mismatch:'
                                'required {}-d, but got {}-d'.format(in_dim, embed.shape[1]))
            self.emb_node = nn.Parameter(torch.tensor(embed),
                                         requires_grad=True)
        else:
            self.emb_node = nn.Parameter(torch.zeros(num_nodes, in_dim),
                                         requires_grad=True)
            nn.init.xavier_normal_(self.emb_node,
                                   gain=nn.init.calculate_gain('relu'))

        # Embedding of relation
        if num_bases > 0:
            # linear combination of a set of basis vectors
            self.emb_rel = nn.Parameter(torch.zeros(num_bases, in_dim))
        else:
            # independently defining an embedding for each relation
            self.emb_rel = nn.Parameter(
                torch.zeros(num_relations * 2, in_dim))
        nn.init.xavier_normal_(self.emb_rel,
                               gain=nn.init.calculate_gain('relu'))

        # Hidden layers
        act = torch.tanh
        layers = []
        for idx in range(num_layers):
            if idx == 0:
                # First layer: embedding lookup size to hidden size
                layer = CompGCNCov(in_dim, num_hidden, act, conv_bias, gcn_drop,
                                   operation, num_bases, num_relations)
            elif idx == num_layers - 1:
                # Last layer: hidden size to output size
                layer = CompGCNCov(num_hidden, out_dim, act, conv_bias, gcn_drop,
                                   operation)
            else:
                layer = CompGCNCov(num_hidden, num_hidden, act, conv_bias, gcn_drop,
                                   operation, num_bases, num_relations)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # Record full graph information for propagation
        self.graph = graph
        self.edge_type = edge_type
        self.edge_norm = edge_norm

        # Connction from node embedding to relational decoder
        self.emb_connect = embed_connect

    def forward(self, _, node_id, rel_id, __):
        # Make sure data is on right device
        g, node_id, rel_id = self.assure_device(self.graph, node_id, rel_id)
        h, r = self.assure_device(self.emb_node, self.emb_rel)

        if self.emb_connect == 'embed':
            return (h, r,
                    h[node_id.squeeze()],
                    r[rel_id.squeeze()])

        for layer in self.layers:
            h, r = layer(g, h, r, self.edge_type, self.edge_norm)

        # Get embdding for current sub-graph
        head_emb = h[node_id.squeeze()]
        rel_emb = r[rel_id.squeeze()]

        import pdb
        pdb.set_trace()

        if self.emb_connect == 'residual':
            return (h, head_emb + h[node_id.squeeze()], rel_emb)
        elif self.emb_connect == 'graph':
            return (h, head_emb, rel_emb)
        else:
            raise NotImplementedError


def load_graph_encoder(params, dataset, embed=None, graph=None, edge_type=None, edge_norm=None):
    # Load graph encoder
    graph_encoder_name = params['graph_encoder']['name']
    n_hidden = params['graph_encoder']['n_hidden']
    embed_connect = params['graph_encoder']['embed_connect']

    if graph_encoder_name == 'rgcn':
        model_details = params['graph_encoder']['details']['rgcn']
        n_layers = model_details['n_layers']
        n_bases = model_details['n_bases']
        dropout = model_details['dropout']
        graph_encoder = RelGraphConvolutionEncoder(dataset.num_nodes, dataset.num_relations * 2,
                                                   n_hidden, n_bases, n_layers, dropout, embed,
                                                   embed_connect)
    elif graph_encoder_name == 'gat':
        model_details = params['graph_encoder']['details']['gat']
        n_layers = model_details['n_layers']
        negative_slope = model_details['negative_slope']
        residual = model_details['residual']
        attn_drop = model_details['attn_drop']
        in_drop = model_details['in_drop']
        num_heads = model_details['n_heads']
        # Always return 1 head
        heads = [num_heads] * (n_layers - 1) + [1]
        graph_encoder = GraphAttentionEncoder(dataset.num_nodes, n_layers, n_hidden, n_hidden, heads,
                                              in_drop, attn_drop, negative_slope, residual, embed,
                                              embed_connect, graph)
    elif graph_encoder_name == 'comp':
        model_details = params['graph_encoder']['details']['comp']
        n_layers = model_details['n_layers']
        n_bases = model_details['n_bases']
        dropout = model_details['dropout']
        operation = model_details['operation']
        in_dim = model_details['in_dim']
        out_dim = model_details['out_dim']
        conv_bias = model_details['conv_bias']
        graph_encoder = CompGraphConvolutionEncoder(dataset.num_nodes, dataset.num_relations,
                                                    n_bases, in_dim, n_hidden, out_dim, n_layers,
                                                    operation, dropout, conv_bias, embed, embed_connect,
                                                    graph, edge_type, edge_norm)
    else:
        raise NotImplementedError()

    return graph_encoder
