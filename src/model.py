import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectDomainClassifier(nn.Module):

    def __init__(self, encoder, num_object_classes=10, num_domain_classes=3):
        super(ObjectDomainClassifier, self).__init__()
        self.encoder = encoder
        if self.encoder.disentangle_layer == self.encoder.num_layers:
            # Method 1
            object_dim = self.encoder.object_dim
            domain_dim = self.encoder.domain_dim
        else:
            # Method 2/3-1
            object_dim = self.encoder.last_hidden_dim
            domain_dim = self.encoder.last_hidden_dim

        self.object_clf = nn.Linear(object_dim, num_object_classes)
        self.domain_clf = nn.Linear(domain_dim, num_domain_classes)

    def forward(self, x, perm=None):
        x = self.encoder(x, perm=perm)
        if isinstance(x, tuple):
            # Method 1
            logits_object = self.object_clf(x[0])
            logits_domain = self.domain_clf(x[1])
        else:
            # Method 2/3-1
            logits_object = self.object_clf(x)
            logits_domain = self.domain_clf(x)

        return logits_object, logits_domain


class LatentObjectClassifier(nn.Module):

    def __init__(self, encoder, num_object_classes=10):
        super(LatentObjectClassifier, self).__init__()
        self.encoder = encoder
        self.object_clf = nn.Linear(self.encoder.object_dim, num_object_classes)

    def forward(self, x, perm=None):
        # Method 3-2
        object_feat, _ = self.encoder(x, return_feature=True)
        return self.object_clf(object_feat)


class MLPClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_classes=10):
        super(MLPClassifier, self).__init__()
        self.linears = nn.ModuleList()
        self.clf = nn.Linear(hidden_dims[-1], num_classes)
        self.num_layers = len(hidden_dims)

        for i in range(len(hidden_dims)):
            in_features = input_dim if i == 0 else hidden_dims[i - 1]
            self.linears.append(nn.Linear(in_features, hidden_dims[i]))

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.linears[i](x))

        return self.clf(x)


class DomainAwareEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims, domain_dim, disentangle_layer):
        super(DomainAwareEncoder, self).__init__()
        self.object_dim = hidden_dims[disentangle_layer - 1]
        self.domain_dim = domain_dim
        self.num_layers = len(hidden_dims)
        self.disentangle_layer = disentangle_layer

        hidden_dims[self.disentangle_layer - 1] += domain_dim
        self.last_hidden_dim = hidden_dims[-1]

        self.linears = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_features = input_dim if i == 0 else hidden_dims[i - 1]
            self.linears.append(nn.Linear(in_features, hidden_dims[i]))

    def forward(self, x, perm=None, return_feature=False):
        for i in range(self.disentangle_layer):
            x = F.relu(self.linears[i](x))

        object_feat, domain_feat = x[:, :-self.domain_dim], x[:, -self.domain_dim:]
        if self.disentangle_layer == self.num_layers or return_feature:
            # Method 1/3-2
            return object_feat, domain_feat

        # Method 2/3-1
        if self.training:
            x_perm = torch.cat((object_feat, domain_feat[perm]), dim=-1)
            x = torch.cat((x, x_perm))

        for i in range(self.disentangle_layer, self.num_layers):
            x = F.relu(self.linears[i](x))

        return x


class GradReverse(torch.autograd.Function):
    def __init__(self):
        super(GradReverse, self).__init__()

    @ staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.view_as(x)

    @ staticmethod
    def backward(ctx, grad_output):
        lambda_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return - lambda_ * grad_input, None


class GradReverseLayer(torch.nn.Module):

    def __init__(self, lambd):
        super(GradReverseLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        lam = torch.tensor(self.lambd)
        return GradReverse.apply(x, lam)


class MLPClassifierReverse(nn.Module):

    def __init__(self, input_dim, hidden_dims, num_classes=10, num_domains=3):
        super(MLPClassifierReverse, self).__init__()
        self.linears = nn.ModuleList()
        self.object_clf = nn.Linear(hidden_dims[-1], num_classes)
        self.domain_clf = nn.Linear(hidden_dims[-1], num_domains)
        self.grl = GradReverseLayer(1)
        self.num_layers = len(hidden_dims)

        for i in range(len(hidden_dims)):
            in_features = input_dim if i == 0 else hidden_dims[i - 1]
            self.linears.append(nn.Linear(in_features, hidden_dims[i]))

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.linears[i](x))

        logits_object = self.object_clf(x)
        logits_domain = self.domain_clf(self.grl(x))
        return logits_object, logits_domain
