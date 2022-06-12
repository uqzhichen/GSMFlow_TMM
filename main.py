import os
import math
import glob
import json
import random
import argparse
import classifier
from utils import *
import torch.nn as nn
import torch.optim as optim
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn.functional as F
from time import gmtime, strftime
import torch.backends.cudnn as cudnn
from dataset import FeatDataLayer, DATA_LOADER
from sklearn.metrics.pairwise import cosine_similarity
# from diversify import diversify_n
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='APY')
parser.add_argument('--dataroot', default='/home/uqzche20/Datasets/SDGZSL_data', help='path to dataset')
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)
parser.add_argument('--zsl', action='store_true', help='evaluation for zsl or gzsl')
parser.add_argument('--gpu', default="0", help='index of GPU to use')

# Augmentation
parser.add_argument('--aug', action='store_true', help='index of GPU to use')
parser.add_argument('--relation_nepoch', type=int, default=50, help='number of epochs to train for relationNet')
parser.add_argument('--relation_batchsize', type=int, default=200, help='relationNet batchsize')
parser.add_argument('--relation_wd', type=float, default=3e-6, help='relationNet  weight decay')
parser.add_argument('--relation_lr', type=float, default=0.001, help='relationNet  weight decay')
parser.add_argument('--mse', type=float, default=1, help='relationNet  weight decay')
parser.add_argument('--aug_lr', type=float, default=30, help='augmentation learning rate')
parser.add_argument('--aug_wd', type=float, default=0, help='augmentation weight decay')
parser.add_argument('--aug_mt', type=float, default=0.0, help='augmentation momentum')
parser.add_argument('--loops_adv', type=int, default=20, help='augmentation steps for each sample')
parser.add_argument('--entropy_weight', type=float, default=10, help='augmentation entropy loss weight')
parser.add_argument('--mse_weight', type=float, default=1, help='augmentation entropy loss weight')

# Flow
parser.add_argument('--gen_nepoch', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate to train generater')
parser.add_argument('--batchsize', type=int, default=200, help='input batch size')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
parser.add_argument('--step_size', type=int, default=20, help='learning rate scheduler step size')
parser.add_argument('--scheduler_gamma', type=float, default=0.55, help='learning rate scheduler gamma')
parser.add_argument('--num_coupling_layers', type=int, default=5, help='number of coupling layers')
parser.add_argument('--prototype',    type=float, default=3, help='weight of the prototype loss')
parser.add_argument('--pi', type=float, default=0.02, help='degree of the perturbation')
parser.add_argument('--dropout', type=float, default=0.0, help='probability of dropping a dimension '
                                                               'in the perturbation noise')
# Cls
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--cls_batch', type=int, default=1200, help='batch size to train softmax classifier')
parser.add_argument('--cls_step', type=int, default=20, help='epochs to train softmax classifier')
parser.add_argument('--nSample', type=int, default=9000, help='number features to generate per class')

# Eval
parser.add_argument('--eval_start', type=int, default=5000, help='evaluation start iteration')
parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=300)
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
parser.add_argument('--input_dim',     type=int, default=1024, help='dimension of the global semantic vectors')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")


def diversify(dataloader, relationNet, opt):
    curent_epoch = dataloader._epoch
    mse = nn.MSELoss().cuda()

    train_feature = opt.dataset.train_feature.clone()
    train_label = opt.dataset.train_label.clone()

    dist_fn = torch.nn.MSELoss()
    while dataloader._epoch == curent_epoch:
        blobs = dataloader.forward()
        feat_data = blobs['data']  # image data
        labels_numpy = blobs['labels'].astype(int)  # class labels
        labels = torch.from_numpy(labels_numpy.astype('int')).cuda()
        # batch_att = [train_att[i, :] for i in labels.unique()]

        batch_att = np.array([opt.dataset.train_att[i, :] for i in labels.unique()])
        batch_att = torch.from_numpy(batch_att.astype('float32')).cuda()
        # sr = sm(batch_att).detach()

        X = torch.from_numpy(feat_data).cuda()


        _, inputs_embedding = relationNet(X, batch_att)
        inputs_embedding = inputs_embedding.detach().clone()

        inputs_embedding.requires_grad_(False)
        inputs_max = X.detach().clone()
        inputs_max.requires_grad_(True)

        batch_att_n = labels.unique().shape[0]
        sample_label = labels.unique().cpu()
        sample_labels = np.array(sample_label)

        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.zeros(opt.batchsize, batch_att_n).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()

        optimier = optim.SGD(params=[inputs_max], lr=opt.aug_lr, momentum=opt.aug_mt, weight_decay=opt.aug_wd)

        for ite_max in range(opt.loops_adv):
            relations, inputs_embedding_pred = relationNet(inputs_max, batch_att)
            relations = relations.view(-1, batch_att_n)
            loss = opt.mse * mse(relations, one_hot_labels) - opt.entropy_weight *entropy_loss(relations)
            relationNet.zero_grad()
            optimier.zero_grad()
            (loss).backward()
            optimier.step()
        X = inputs_max.detach().clone().cpu()
        train_feature = torch.cat((train_feature, X))
        train_label = torch.cat((train_label, labels.cpu()))
        # opt.dataset.train_feature = torch.cat((opt.dataset.train_feature, X))
        # opt.dataset.train_label = torch.cat((opt.dataset.train_label, labels.cpu()))
    # vis(train_feature,train_label)
    print("################finished diversify##########################")
    return FeatDataLayer(train_label, train_feature.cpu().numpy(), opt)
    # return FeatDataLayer(opt.dataset.train_label, opt.dataset.train_feature.cpu().numpy(), opt)

def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.y_dim = dataset.ntrain_class
    out_dir = 'out/{}/seed-{}_mask-{}_c-{}_pi-{}_lr-{}_nS-{}_bs-{}'.format(opt.dataset, opt.manualSeed, opt.dropout,
                            opt.prototype, opt.pi, opt.lr, opt.nSample, opt.batchsize)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))
    opt.dataset = dataset

    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)

    # Training iterations for Flow
    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.gen_nepoch

    # Iterations per epoch (for lr scheduler)
    iters = math.ceil(dataset.ntrain/opt.batchsize)

    # Training iterations for relationNet
    opt.niter_relation = int(dataset.ntrain/opt.batchsize) * opt.relation_nepoch

    # For Saving evaluation results
    result_gzsl_soft = Result()
    result_zsl_soft = Result()

    # Semantic Anchors
    sim = cosine_similarity(dataset.attribute, dataset.attribute)
    min_idx = np.argmin(sim.sum(-1))
    min = dataset.attribute[min_idx]
    max_idx = np.argmax(sim.sum(-1))
    max = dataset.attribute[max_idx]
    medi_idx = np.argwhere(sim.sum(-1)==np.sort(sim.sum(-1))[int(sim.shape[0]/2)])
    medi = dataset.attribute[int(medi_idx)]
    vertices = torch.from_numpy(np.stack((min,max,medi))).float().cuda()

    # Building Models
    flow = cINN(opt).cuda()
    relationNet = RelationNetV2(opt).cuda()
    sm = GSModule(vertices, int(opt.input_dim)).cuda()

    # Optimizers
    optimizer = optim.Adam(list(flow.trainable_parameters)+list(sm.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)
    optimizer_relation = optim.Adam(list(relationNet.parameters()), lr=opt.relation_lr, weight_decay=opt.relation_wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=opt.scheduler_gamma, step_size=opt.step_size)

    # mse loss for relationNet training
    mse = nn.MSELoss()

    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('GSMFlow Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')
    prototype_loss = 0
    x_mean = torch.from_numpy(dataset.tr_cls_centroid).cuda()

    if opt.aug:
        data_layer._opt.batchsize = opt.relation_batchsize
        for it in range(opt.niter_relation + 1):
            relationNet.zero_grad()
            blobs = data_layer.forward()
            feat_data = blobs['data']  # image data
            labels_numpy = blobs['labels'].astype(int)  # class labels
            labels = torch.from_numpy(labels_numpy.astype('int')).cuda()
            _, unique_label_index = np.unique(labels_numpy, return_index=True)
            sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).cuda()
            sample_C_n = labels.unique().shape[0]
            sample_label = labels.unique().cpu()

            X = torch.from_numpy(feat_data).cuda()

            sample_labels = np.array(sample_label)
            re_batch_labels = []
            for label in labels_numpy:
                index = np.argwhere(sample_labels == label)
                re_batch_labels.append(index[0][0])
            re_batch_labels = torch.LongTensor(re_batch_labels)
            one_hot_labels = torch.zeros(opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()

            relations_gt, _ = relationNet(X,sample_C)
            relations_gt = relations_gt.view(-1, labels.unique().cpu().shape[0])
            # print("prediction rate:", str((re_batch_labels.cuda() == relations_gt.argmax(1)).sum().item()/opt.relation_batchsize))
            relation_loss = mse(relations_gt, one_hot_labels)
            relation_loss.backward()
            optimizer_relation.step()
        relationNet.eval()
        data_layer = diversify(data_layer, relationNet, opt)

    for it in range(opt.niter+1):
        flow.zero_grad()
        sm.zero_grad()

        blobs = data_layer.forward()
        feat_data = blobs['data']
        labels_numpy = blobs['labels'].astype(int)
        labels = torch.from_numpy(labels_numpy.astype('int')).cuda()

        C = np.array([dataset.train_att[i, :] for i in labels])
        C = torch.from_numpy(C.astype('float32')).cuda()
        X = torch.from_numpy(feat_data).cuda()

        sr = sm(C)
        z = opt.pi * torch.randn(opt.batchsize, 2048).cuda()
        mask = torch.cuda.FloatTensor(2048).uniform_() > opt.dropout
        z = mask * z
        X = X + z
        z_, log_jac_det = flow(X, sr)
        loss = torch.mean(z_**2) / 2 - torch.mean(log_jac_det) / 2048
        loss.backward(retain_graph=True)

        if opt.prototype > 0:
            with torch.no_grad():
                sr = sm(torch.from_numpy(dataset.train_att).cuda())
            z = torch.zeros(dataset.ntrain_class, 2048).cuda()
            x_, _ = flow.reverse_sample(z, sr)
            prototype_loss = opt.prototype*mse(x_, x_mean)
            prototype_loss.backward()

        optimizer.step()
        if it % iters == 0:
            lr_scheduler.step()
        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}; prototype_loss:{:.5f} '.format(it, opt.niter, loss.item(), prototype_loss.item())
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > opt.eval_start:
            flow.eval()
            sm.eval()
            gen_feat, gen_label = synthesize_feature(flow, sm, dataset, opt)

            train_X = torch.cat((dataset.train_feature, gen_feat), 0)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)

            if opt.zsl:
                """ZSL"""
                cls = classifier.CLASSIFIER(opt, gen_feat, gen_label, dataset, dataset.test_seen_feature, dataset.test_unseen_feature,
                                            dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, opt.cls_step,
                                            opt.cls_batch, False)
                result_zsl_soft.update(it, cls.acc)
                log_print("ZSL Softmax:", log_dir)
                log_print("Acc {:.2f}%  Best_acc [{:.2f}% | Iter-{}]".format(
                    cls.acc, result_zsl_soft.best_acc, result_zsl_soft.best_iter), log_dir)
            else:

                """ GZSL"""
                cls = classifier.CLASSIFIER(opt, train_X, train_Y, dataset,  dataset.test_seen_feature,  dataset.test_unseen_feature,
                                    dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, opt.cls_step, opt.cls_batch, True)

                result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

                log_print("GZSL Softmax:", log_dir)
                log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                    cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                    result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

                if result_gzsl_soft.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model(it, flow, sm, opt.manualSeed, log_text,
                               out_dir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl_soft.best_acc,
                                                                                                 result_gzsl_soft.best_acc_S_T,
                                                                                                 result_gzsl_soft.best_acc_U_T))

            sm.train()
            flow.train()
            if it % opt.save_interval == 0 and it:
                save_model(it, flow, sm, opt.manualSeed, log_text,
                           out_dir + '/Iter_{:d}.tar'.format(it))
                print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))


if __name__ == "__main__":
    train()