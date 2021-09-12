import math

import torch
import dgl
from dgl import function as fn


class SGConv(torch.nn.Module):
    def __init__(self, n_filters, order, p, activation):
        super(SGConv, self).__init__()
        self.n_filters = n_filters
        self.order = order          # order of the stochastic graph filter
        self.p = p        # probability to create the input tensor of probability values for the Bernoulli distribution

        # weight initialization with Chebyshev nodes
        h = self.chebyshev_nodes(order + 1)
        self.H = torch.stack([h] * n_filters, dim=1)  # (order+1) x n_filters shape
        self.activation = activation  # nonlinear function

    def chebyshev_nodes(self, K):
        """Return the K Chebyshev nodes in [-1,1]."""
        return torch.cos(math.pi * (torch.arange(K) + 1 / 2) / K)

    def shift_operator(self, S, num_nodes):
        """
        :param S: torch.Tensor
            Shift operator of original graph
        :param num_nodes: int
            Number of nodes in the graph
        :return:
        """

        probabilities = torch.ones(num_nodes, num_nodes) * self.p
        sample = torch.bernoulli(probabilities)
        S_k = torch.where(S.triu() == 0., 0., sample.triu().type(torch.double)).type(torch.float)
        S_k = S_k + S_k.T
        return S_k

    def forward(self, g, x):
        """
        :param g: DGLGraph
            The graph.
        :param x: torch.Tensor
            It represents the input feature of shape (number of nodes, size of input feature)
        :return: torch.Tensor
            The output feature

        Note
        -----
        Shift operator (S) is an adjacency matrix of graph in this implementation.

        """

        if self.n_filters <= 0:
            raise Exception('Number of filters must be greater than 0.')

        if self.order <= 0:
            raise Exception('Order must be greater than 0.')

        with g.local_scope():
            n_nodes = g.num_nodes()
            adj_matrix = g.adjacency_matrix(transpose=True, scipy_fmt=g.formats()['created'][0])
            S = torch.tensor(adj_matrix.todense(), dtype=torch.float)

            x_diffusion = list()

            for f in range(self.n_filters):
                init_g = dgl.graph((g.nodes(), g.nodes()))
                S_k = torch.eye(init_g.num_nodes(), init_g.num_nodes())

                u_node_ids, v_node_ids = torch.where(S_k == 1)
                edge_ids = init_g.edge_ids(u_node_ids, v_node_ids)
                eweight = init_g.adjacency_matrix(transpose=True).coalesce().values()
                eweight[:] = 0
                eweight[edge_ids] = 1

                init_g.ndata['ft'] = x
                init_g.edata['a'] = eweight
                init_g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

                x = init_g.ndata.pop('ft')
                x_f_list = list()
                x_f_list.append(x)

                for k in range(1, self.order + 1):
                    S_k = self.shift_operator(S, n_nodes)
                    u_node_ids, v_node_ids = torch.where(S_k == 1)
                    edge_ids = g.edge_ids(u_node_ids, v_node_ids)
                    eweight = g.adjacency_matrix(transpose=True).coalesce().values()
                    eweight[:] = 0
                    eweight[edge_ids] = 1

                    g.ndata['ft'] = x
                    g.edata['a'] = eweight
                    g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

                    x = g.ndata.pop('ft')
                    x_f_list.append(x)

                x_f = torch.cat(x_f_list, dim=1)
                x_diffusion.append(x_f)
            X = torch.cat(x_diffusion, dim=0)       # (n_filters*n_nodes) x (order+1) shape
            X = X.reshape((self.n_filters, n_nodes, self.order + 1))
            X = X.permute(1, 2, 0).contiguous()     # n_nodes x (order+1) x n_filters shape
            u = X * self.H                          # n_nodes x (order+1) x n_filters shape

            # sum filter outputs
            u = torch.sum(u, axis=1)                # n_nodes x n_filters shape

            # sum over filters
            out = torch.sum(u, axis=1, keepdim=True)  # n_nodes x size of input feature shape

        return self.activation(out)
