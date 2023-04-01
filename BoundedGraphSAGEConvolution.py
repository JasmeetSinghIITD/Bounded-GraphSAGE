import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import math
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score

class GraphSAGEConvolution(Module):
    """GraphSAGE layer, similar to https://arxiv.org/abs/1706.02216
    """

    def __init__(self, in_features, out_features, aggregator='mean', with_bias=True):
        super(GraphSAGEConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator

        # Define weights
        self.weight = Parameter(torch.FloatTensor(in_features * (1 + int(aggregator == 'pool')), out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ GraphSAGE Layer forward function
        """
        if input.data.is_sparse:
            # Convert sparse input to dense format
            input = input.to_dense()

        # Aggregate node embeddings from the neighborhood
        if self.aggregator == 'mean':
            agg_func = torch.mean
        elif self.aggregator == 'max':
            agg_func = torch.max
        elif self.aggregator == 'pool':
            agg_func = lambda x: torch.cat((torch.mean(x, dim=0), torch.max(x, dim=0)[0]), dim=0)
        else:
            raise ValueError('Invalid aggregator type')

        nei_input = torch.sparse.mm(adj, input)
        output = torch.cat((input, agg_func(nei_input)), dim=1)

        # Linear transformation
        output = torch.mm(output, self.weight)

        # Add bias if required
        if self.bias is not None:
            output += self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
class BoundedGraphSAGE(nn.Module):
    """ 2 Layer GraphSAGE Network.
    Parameters
    ----------
    nfeat : int
        size of input feature dimension
    nhid : int
        number of hidden units
    nclass : int
        size of output dimension
    dropout : float
        dropout rate for GraphSAGE
    lr : float
        learning rate for GraphSAGE
    weight_decay : float
        weight decay coefficient (l2 normalization) for GraphSAGE.
        When `with_relu` is True, `weight_decay` will be set to 0.
    with_relu : bool
        whether to use relu activation function. If False, GraphSAGE will be linearized.
    with_bias: bool
        whether to include bias term in GraphSAGE weights.
    device: str
        'cpu' or 'cuda'.
    Examples
    --------
    We can first load dataset and then train GraphSAGE.
    >>> from torch_geometric.datasets import Planetoid
    >>> import torch_geometric.nn as geo_nn
    >>> dataset = Planetoid(root='/tmp/cora', name='cora')
    >>> data = dataset[0]
    >>> model = BoundedGraphSAGE(nfeat=data.num_features,
                                 nhid=16,
                                 nclass=dataset.num_classes,
                                 dropout=0.5,
                                 device='cpu')
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    >>> model = model.to('cpu')
    >>> model.train()
    >>> optimizer.zero_grad()
    >>> output = model(data.x, data.edge_index)
    >>> loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    >>> acc_train = accuracy(output[data.train_mask], data.y[data.train_mask])
    >>> loss_train.backward()
    >>> optimizer.step()
    >>> model.eval()
    >>> output = model(data.x, data.edge_index)
    >>> loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
    >>> acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    >>> print('Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(loss_train.item(), acc_train.item()))
    >>> print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(loss_val.item(), acc_val.item()))
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4,
            with_relu=True, with_bias=True, device=None, bound=0, agg_func='mean'):

        super(BoundedGraphSAGE, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.agg_func = agg_func
        self.sage1 = SAGEConv(nfeat, nhid, agg_func=self.agg_func)
        self.sage2 = SAGEConv(nhid, nclass, agg_func=self.agg_func)
        self.dropout = dropout
        self.lr = lr
        self.bound = bound
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def forward(self, x, adj):
        if self.with_relu:
            x = F.relu(self.sage1(x, adj))
        else:
            x = self.sage1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.sage2(x, adj)
        return F.log_softmax(x, dim=1)
    def initialize(self):
        """Initialize parameters of GraphSage.
        """
        self.sage1.reset_parameters()
        self.sage2.reset_parameters()
    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, normalize=True, patience=500, **kwargs):
        """Train the GraphSAGE model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GraphSAGE training process will not adopt early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        normalize : bool
            whether to normalize the input adjacency matrix.
        patience : int
            patience for early stopping, only valid when `idx_val` is given
        """
        print(" Using GraphSAGE")
        self.device = self.sage1.weight.device
        if initialize:
            self.initialize()

        if type(adj) is not torch.Tensor:
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        if normalize:
            if utils.is_sparse_tensor(adj):
                adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(adj)
        else:
            adj_norm = adj

        self.adj_norm = adj_norm
        self.features = features
        self.labels = labels

        if idx_val is None:
            print("Training without val")
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            if patience < train_iters:
                self._train_with_early_stopping(labels, idx_train, idx_val, train_iters, patience, verbose)
            else:
                self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)
  #checkfrom here 

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        print("Training without val")
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            self.l2_reg = self.bound * torch.square(torch.norm(self.sage1.weight)) + torch.square(torch.norm(self.sage2.weight))  # Added by me

            print(f'L2 reg at iteration {i} = {l2_reg}')
            loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + self.l2_reg
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.adj_norm)
        self.output = output
    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        print("Training with val")
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)

            self.l2_reg = 2 * self.bound * (torch.log(torch.norm(self.sage1.weight)) + torch.log(torch.norm(self.sage2.weight)) )    # Added by me

            if self.l2_reg<0:
                self.l2_reg=0

            loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + self.l2_reg

            if i%10==0:
                print(f'l2 Reg = {self.l2_reg} , Loss = {loss_train}')

            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        print("Training with early stopping")
        if verbose:
            print('=== training GraphSage model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = 100

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            # def eval_class(output, labels):
            #     preds = output.max(1)[1].type_as(labels)
            #     return f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='micro') + \
            #         f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')

            # perf_sum = eval_class(output[idx_val], labels[idx_val])
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
             print('=== early stopping at {0}, loss_val = {1} ==='.format(i, best_loss_val) )
        self.load_state_dict(weights)
    
    def test(self, idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output = self.predict()
        # output = self.output
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()
    def predict(self, features=None, adj=None):
        """By default, the inputs should be unnormalized adjacency
        Parameters
        ----------
        features :
            node features. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        adj :
            adjcency matrix. If `features` and `adj` are not given, this function will use previous stored `features` and `adj` from training to make predictions.
        Returns
        -------
        torch.FloatTensor
            output (log probabilities) of GCN
        """

        self.eval()
        if features is None and adj is None:
            return self.forward(self.features, self.adj_norm)
        else:
            if type(adj) is not torch.Tensor:
                features, adj = utils.to_tensor(features, adj, device=self.device)

            self.features = features
            if utils.is_sparse_tensor(adj):
                self.adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
            else:
                self.adj_norm = utils.normalize_adj_tensor(adj)
            return self.forward(self.features, self.adj_norm)
    


    
