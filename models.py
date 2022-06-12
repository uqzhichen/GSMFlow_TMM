import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn.functional as F


class cINN(nn.Module):
    '''cINN for class-conditional MNISt generation'''
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cinn = self.build_inn(opt)

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        for p in self.trainable_parameters:
            p.data = 0.01 * torch.randn_like(p)

    def build_inn(self, opt):

        def subnet(ch_in, ch_out):
            return nn.Sequential(nn.Linear(ch_in, 2048),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(2048, ch_out)
                                 )

        cond = Ff.ConditionNode(opt.input_dim)
        nodes = [Ff.InputNode(2048)]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(opt.num_coupling_layers):
            # nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':subnet, 'clamp':1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def forward(self, x, c):
        z, jac = self.cinn(x, c, jac=True)
        return z, jac

    def reverse_sample(self, z, c):
        return self.cinn(z, c, rev=True, jac=False)

class LinearModule(nn.Module):
    def __init__(self, vertice, out_dim):
        super(LinearModule, self).__init__()
        self.register_buffer('vertice', vertice.clone())
        self.fc = nn.Linear(vertice.numel(), out_dim)

    def forward(self, semantic_vec):
        input_offsets = semantic_vec - self.vertice
        response = F.relu(self.fc(input_offsets))
        return response

class GSModule(nn.Module):
    def __init__(self, vertices, out_dim):
        super(GSModule, self).__init__()
        self.individuals = nn.ModuleList()
        assert vertices.dim() == 2, 'invalid shape : {:}'.format(vertices.shape)
        self.out_dim     = out_dim
        self.require_adj = False
        for i in range(vertices.shape[0]):
            layer = LinearModule(vertices[i], out_dim)
            self.individuals.append(layer)

    def forward(self, semantic_vec):
        responses = [indiv(semantic_vec) for indiv in self.individuals]
        global_semantic = sum(responses)
        return global_semantic

class RelationNet(nn.Module):
    def __init__(self, args):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(args.C_dim + 2048, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, s, c):

        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        cls_num = c_ext.shape[1]

        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])
        relation = nn.ReLU()(self.fc1(relation_pairs))
        relation = nn.Sigmoid()(self.fc2(relation))
        return relation

class RelationNetV2(nn.Module):
    def __init__(self, args):
        super(RelationNetV2, self).__init__()
        self.fc1 = nn.Linear(args.C_dim + 2048, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, s, c):

        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        cls_num = c_ext.shape[1]

        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])
        h = nn.ReLU()(self.fc1(relation_pairs))
        relation = nn.Sigmoid()(self.fc2(h))
        return relation, h


class Linear_classifier(nn.Module):
    def __init__(self, input_dim, nclass):
        super(Linear_classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x):
        h = nn.ReLU()(self.fc1(x))
        o = self.logic(self.fc2(h))
        return o, h
