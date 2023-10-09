import torch
from transformers import BertPreTrainedModel, BertModel, BertConfig, RobertaPreTrainedModel, RobertaModel
from torch import nn
from einops import einsum, rearrange, reduce, repeat
import numpy as np
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta


def exists(val):
    return val is not None


class BertClassify(BertPreTrainedModel):
    def __init__(self, config, label_num, **kwargs):
        super(BertClassify, self).__init__(config)
        self.bert = BertModel(config)

        self.config = config
        self.classify = nn.Linear(config.hidden_size, label_num)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, x, mask=None, target=None):
        seqlen, device = x.shape[-1], x.device
        # batch, seq, dim
        x = self.bert(x, attention_mask=mask).pooler_output
        x = self.classify(x)

        if exists(target):
            loss = self.loss_fct(x, target)
            return loss, x

        return 0, x


class BertClassifyAug(BertPreTrainedModel):
    def __init__(self, config, label_num, **kwargs):
        super(BertClassifyAug, self).__init__(config)
        self.bert = RobertaModel(config)

        self.config = config
        self.classify = nn.Linear(config.hidden_size, label_num)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.kl_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.m = Dirichlet(torch.tensor([0.5, 0.5, 0.5]))
        self.b = Beta(torch.tensor([0.5]), torch.tensor([0.5]))

        self.classify_aug1 = nn.Linear(config.hidden_size, label_num)
        self.classify_aug2 = nn.Linear(config.hidden_size, label_num)
        self.classify_aug3 = nn.Linear(config.hidden_size, label_num)

    def forward(self, x, aug_x1=None, aug_x2=None, aug_x3=None, mask=None, target=None):
        seqlen, device = x.shape[-1], x.device
        x = self.bert(x, attention_mask=mask)
        if exists(aug_x1):
            # batch, seq, dim
            # eff1 = self.m.sample().cuda()
            # eff2 = self.m.sample().cuda()
            # beta1 = self.b.sample().cuda()
            # beta2 = self.b.sample().cuda()
            # x1 = self.classify_aug1(self.bert(aug_x1, attention_mask=mask).pooler_output)
            # x2 = self.classify_aug2(self.bert(aug_x2, attention_mask=mask).pooler_output)
            # x3 = self.classify_aug3(self.bert(aug_x3, attention_mask=mask).pooler_output)
            #
            # aug1 = eff1[0] * x1 + eff1[1] * x2 + eff1[2] * x3
            # aug1 = (1 - beta1) * aug1 + beta1 * x
            # aug2 = eff2[0] * x1 + eff2[1] * x2 + eff2[2] * x3
            # aug2 = (1 - beta2) * aug2 + beta2 * x
            #
            # log_aug1 = F.log_softmax(aug1, dim=-1)
            # log_aug2 = F.log_softmax(aug2, dim=-1)
            # log_mid = (F.log_softmax(x, dim=-1) + aug1 + aug2) / 3

            # kl_loss = (self.kl_fct(F.log_softmax(x, dim=-1), log_mid) + self.kl_fct(log_aug1, log_mid) + \
            #            self.kl_fct(log_aug2, log_mid)) / 3

            aug1 = self.bert(aug_x1, attention_mask=mask).pooler_output
            kl_loss = self.kl_fct(
                F.log_softmax(x.pooler_output, dim=-1),
                F.log_softmax(aug1, dim=-1)
            )

        x = self.classify(x.pooler_output)

        if exists(target):
            if exists(aug_x1):
                loss = self.loss_fct(x, target) + 1 * kl_loss
            else:
                loss = self.loss_fct(x, target)
            return loss, x

        return 0, x


class DualNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.layer_norm = nn.LayerNorm(size, elementwise_affine=False)
        self.instance_norm = nn.InstanceNorm1d(size, affine=False)
        self.dense_before = nn.Linear(size*2, size*2)
        self.dense_after = nn.Linear(size*2, size*2)

    def forward(self, x):
        x = self.dense_before(x)
        left, right = torch.split(x, self.size, dim=-1)
        left = self.instance_norm(left.transpose(2, 1)).transpose(2, 1)
        right = self.layer_norm(right)
        x = torch.cat([left, right], dim=-1)
        x = self.dense_after(x)

        return x


class RoBertaClassifyAug(RobertaPreTrainedModel):
    """在原先基础上，增加margin"""
    def __init__(self, config, label_num, device='cuda:0', lamb=0.2, **kwargs):
        super(RoBertaClassifyAug, self).__init__(config)
        self.roberta = RobertaModel(config)

        self.config = config
        self.lamb = lamb
        self.classify = nn.Linear(config.hidden_size, label_num)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.kl_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.dist_fct = torch.cdist
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.param = nn.ParameterList([nn.Parameter(torch.tensor(0.5).to(device))] * 10)

        self.dual_norm1 = DualNorm(config.hidden_size // 2)
        self.dual_norm2 = DualNorm(config.hidden_size // 2)
        self.dual_norm3 = DualNorm(config.hidden_size // 2)

    def forward(self, x, aug_x1=None, aug_x2=None, aug_x3=None, mask=None, target=None):
        seqlen, device = x.shape[-1], x.device
        zero = torch.tensor(0.).float().cuda()
        x = self.roberta(x, attention_mask=mask)
        x_norm = self.dual_norm1(x.last_hidden_state)
        x_flat = torch.mean(x_norm, dim=1)
        if exists(aug_x1):

            x1 = self.roberta(aug_x1, attention_mask=mask).last_hidden_state
            x2 = self.roberta(aug_x2, attention_mask=mask).last_hidden_state
            aug1 = self.param[0] * x1 + self.param[1] * x2
            aug2 = self.param[2] * x1 + self.param[3] * x2

            aug1_norm = self.dual_norm2(aug1)
            aug1_flat = torch.mean(aug1_norm, dim=1)
            aug2_norm = self.dual_norm3(aug2)
            aug2_flat = torch.mean(aug2_norm, dim=1)

            Lx_a1 = self.kl_fct(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug1_flat, dim=-1))
            Lx_a2 = self.kl_fct(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))
            La1_a2 = self.kl_fct(F.log_softmax(aug1_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))
            kl_loss = (Lx_a1 + Lx_a2 + La1_a2) / 3

            dx_a1 = torch.diag(self.dist_fct(x_flat, aug1_flat))
            dx_a2 = torch.diag(self.dist_fct(x_flat, aug2_flat))
            da1_a2 = torch.diag(self.dist_fct(aug1_flat, aug2_flat))
            # distant_loss = (self.cal_dist(dx_a1, dx_a2, zero) + \
            #                self.cal_dist(dx_a1, da1_a2, zero) + \
            #                self.cal_dist(dx_a2, da1_a2, zero)) / 3 + \
            #                torch.mean(torch.max(zero, (3.5 - torch.sqrt(dx_a1))**2) + torch.max(zero, (3.5 - torch.sqrt(dx_a2))**2)) / 2
            distant_loss = torch.mean(torch.max(zero, dx_a1**2 + dx_a2**2 - da1_a2**2 + 50.0)) + torch.mean(torch.abs((dx_a1 - 3.5)) + torch.abs((dx_a2 - 3.5))) / 2

        x = self.classify(x_flat)

        if exists(target):
            if exists(aug_x1):
                loss = self.loss_fct(x, target) + self.lamb * kl_loss + self.lamb * distant_loss
            else:
                loss = self.loss_fct(x, target)
            return loss, x

        return 0, x, (x_flat, aug1_flat, aug2_flat)

    def cal_dist(self, x, x1, zero):
        d1 = x - x1
        d1[torch.less_equal(d1, 0)] = zero
        length = torch.sum(torch.greater(d1, 0))
        return torch.sum(d1) / length.clamp(min=1.)

    def cal_dist1(self, x, x1, zero):
        d1 = x - x1
        return torch.sum(torch.max(d1, zero))

class RoBertaClassifyAug4(RobertaPreTrainedModel):
    """在原先基础上，采用欧式距离而非kl散度"""
    def __init__(self, config, label_num, device='cuda:0', lamb=0.2, **kwargs):
        super(RoBertaClassifyAug4, self).__init__(config)
        self.roberta = RobertaModel(config)

        self.config = config
        self.lamb = lamb
        self.classify = nn.Linear(config.hidden_size, label_num)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.kl_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.dist_fct = torch.cdist
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.param = nn.ParameterList([nn.Parameter(torch.tensor(0.5).to(device))] * 10)

        self.dual_norm1 = DualNorm(config.hidden_size // 2)
        self.dual_norm2 = DualNorm(config.hidden_size // 2)
        self.dual_norm3 = DualNorm(config.hidden_size // 2)

    def forward(self, x, aug_x1=None, aug_x2=None, aug_x3=None, mask=None, target=None):
        seqlen, device = x.shape[-1], x.device
        zero = torch.tensor(0.).float().cuda()
        x = self.roberta(x, attention_mask=mask)
        x_norm = self.dual_norm1(x.last_hidden_state)
        x_flat = torch.mean(x_norm, dim=1)
        if exists(aug_x1):

            x1 = self.roberta(aug_x1, attention_mask=mask).last_hidden_state
            x2 = self.roberta(aug_x2, attention_mask=mask).last_hidden_state
            aug1 = self.param[0] * x1 + self.param[1] * x2
            aug2 = self.param[2] * x1 + self.param[3] * x2

            aug1_norm = self.dual_norm2(aug1)
            aug1_flat = torch.mean(aug1_norm, dim=1)
            aug2_norm = self.dual_norm3(aug2)
            aug2_flat = torch.mean(aug2_norm, dim=1)

            Lx_a1 = self.kl_fct(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug1_flat, dim=-1))
            Lx_a2 = self.kl_fct(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))
            La1_a2 = self.kl_fct(F.log_softmax(aug1_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))
            kl_loss = (Lx_a1 + Lx_a2 + La1_a2) / 3

            dx_a1 = torch.diag(self.dist_fct(x_flat, aug1_flat))
            dx_a2 = torch.diag(self.dist_fct(x_flat, aug2_flat))
            da1_a2 = torch.diag(self.dist_fct(aug1_flat, aug2_flat))
            distant_loss = (self.cal_dist(dx_a1, dx_a2, zero) + \
                           self.cal_dist(dx_a1, da1_a2, zero) + \
                           self.cal_dist(dx_a2, da1_a2, zero)) / 3
            # sim = self.similarity(aug1_flat, aug2_flat)
            # distant_loss = torch.mean(torch.ones_like(sim) + torch.abs(sim)) + \
            #                distant_loss

        x = self.classify(x_flat)

        if exists(target):
            if exists(aug_x1):
                loss = self.loss_fct(x, target) + self.lamb * kl_loss + self.lamb * distant_loss
            else:
                loss = self.loss_fct(x, target)
            return loss, x

        return 0, x, (x_flat, aug1_flat, aug2_flat)

    def cal_dist(self, x, x1, zero):
        d1 = x - x1
        d1[torch.less_equal(d1, 0)] = zero
        length = torch.sum(torch.greater(d1, 0))
        return torch.sum(d1) / length.clamp(min=1.)


class RoBertaClassifyAug3(RobertaPreTrainedModel):
    """在原先基础上，加上rank
    理论上掩码越少的样本，越简单，距离真实样本越近"""
    def __init__(self, config, label_num, device='cuda:0', lamb=0.2, **kwargs):
        super(RoBertaClassifyAug3, self).__init__(config)
        self.roberta = RobertaModel(config)

        self.config = config
        self.lamb = lamb
        self.classify = nn.Linear(config.hidden_size, label_num)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.kl_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.kl_fct_single = torch.nn.KLDivLoss(reduction="none", log_target=True)

        self.param = nn.ParameterList([nn.Parameter(torch.zeros(1).to(device))] * 10)

        self.dual_norm1 = DualNorm(config.hidden_size // 2)
        self.dual_norm2 = DualNorm(config.hidden_size // 2)
        self.dual_norm3 = DualNorm(config.hidden_size // 2)

    def forward(self, x, aug_x1=None, aug_x2=None, aug_x3=None, mask=None, target=None):
        seqlen, device = x.shape[-1], x.device
        zero = torch.tensor(0.).float().cuda()
        x = self.roberta(x, attention_mask=mask)
        x_norm = self.dual_norm1(x.last_hidden_state)
        x_flat = torch.mean(x_norm, dim=1)
        if exists(aug_x1):

            x1 = self.roberta(aug_x1, attention_mask=mask).last_hidden_state
            x2 = self.roberta(aug_x2, attention_mask=mask).last_hidden_state
            aug1 = self.param[0] * x1 + self.param[1] * x2
            aug2 = self.param[2] * x1 + self.param[3] * x2

            aug1_norm = self.dual_norm2(aug1)
            aug1_flat = torch.mean(aug1_norm, dim=1)
            aug2_norm = self.dual_norm3(aug2)
            aug2_flat = torch.mean(aug2_norm, dim=1)

            Lx_a1 = self.kl_fct(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug1_flat, dim=-1))
            Lx_a2 = self.kl_fct(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))
            La1_a2 = self.kl_fct(F.log_softmax(aug1_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))
            kl_loss = Lx_a1 + Lx_a2 + La1_a2
            Lx_a1_single = self.kl_fct_single(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug1_flat, dim=-1))
            Lx_a2_single = self.kl_fct_single(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))
            La1_a2_single = self.kl_fct_single(F.log_softmax(aug1_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))

            distant_loss = self.cal_dist_per_sample(Lx_a1_single, Lx_a2_single, zero) + \
                           self.cal_dist_per_sample(Lx_a1_single, La1_a2_single, zero) + \
                           self.cal_dist_per_sample(Lx_a2_single, La1_a2_single, zero)

        x = self.classify(x_flat)

        if exists(target):
            if exists(aug_x1):
                loss = self.loss_fct(x, target) + self.lamb * kl_loss + self.lamb * distant_loss
            else:
                loss = self.loss_fct(x, target)
            return loss, x

        return 0, x

    def cal_dist_per_sample(self, x, x1, zero):
        d1 = x - x1
        d1[torch.less_equal(d1, 0)] = zero
        length = torch.sum(torch.greater(d1, 0))
        return torch.sum(d1) / length.clamp(min=1.)


class RoBertaClassifyAug2(RobertaPreTrainedModel):
    """在原来的基础上，加上了DuLIN"""
    def __init__(self, config, label_num, device='cuda:0', lamb=0.2, **kwargs):
        super(RoBertaClassifyAug2, self).__init__(config)
        self.roberta = RobertaModel(config)

        self.config = config
        self.lamb = lamb
        self.classify = nn.Linear(config.hidden_size, label_num)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.kl_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

        self.param = nn.ParameterList([nn.Parameter(torch.zeros(1).to(device))] * 10)

        self.dual_norm1 = DualNorm(config.hidden_size // 2)
        self.dual_norm2 = DualNorm(config.hidden_size // 2)
        self.dual_norm3 = DualNorm(config.hidden_size // 2)

    def forward(self, x, aug_x1=None, aug_x2=None, aug_x3=None, mask=None, target=None):
        seqlen, device = x.shape[-1], x.device
        x = self.roberta(x, attention_mask=mask)
        x_norm = self.dual_norm1(x.last_hidden_state)
        x_flat = torch.mean(x_norm, dim=1)
        if exists(aug_x1):

            x1 = self.roberta(aug_x1, attention_mask=mask).last_hidden_state
            x2 = self.roberta(aug_x2, attention_mask=mask).last_hidden_state
            aug1 = self.param[0] * x1 + self.param[1] * x2
            aug2 = self.param[2] * x1 + self.param[3] * x2

            aug1_norm = self.dual_norm2(aug1)
            aug1_flat = torch.mean(aug1_norm, dim=1)
            aug2_norm = self.dual_norm3(aug2)
            aug2_flat = torch.mean(aug2_norm, dim=1)

            kl_loss = self.kl_fct(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug1_flat, dim=-1)) + \
                      self.kl_fct(F.log_softmax(x_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1)) + \
                      self.kl_fct(F.log_softmax(aug1_flat, dim=-1), F.log_softmax(aug2_flat, dim=-1))


        x = self.classify(x_flat)

        if exists(target):
            if exists(aug_x1):
                loss = self.loss_fct(x, target) + self.lamb * kl_loss
            else:
                loss = self.loss_fct(x, target)
            return loss, x

        return 0, x


class RoBertaClassifyAug0(RobertaPreTrainedModel):
    def __init__(self, config, label_num, device='cuda:0', **kwargs):
        super(RoBertaClassifyAug0, self).__init__(config)
        self.roberta = RobertaModel(config)

        self.config = config
        self.classify = nn.Linear(config.hidden_size, label_num)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.kl_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        # self.m = Dirichlet(torch.tensor([0.5, 0.5, 0.5]))
        # self.b = Beta(torch.tensor([0.5]), torch.tensor([0.5]))

        self.param = [nn.Parameter(torch.zeros(1).to(device))] * 10
        self.param1 = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
        self.param2 = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
        self.param3 = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)
        self.param4 = nn.Parameter(torch.zeros(1).to(device), requires_grad=True)

        self.dual_norm1 = DualNorm(config.hidden_size // 2)
        self.dual_norm2 = DualNorm(config.hidden_size // 2)
        self.dual_norm3 = DualNorm(config.hidden_size // 2)

    def forward(self, x, aug_x1=None, aug_x2=None, aug_x3=None, mask=None, target=None):
        seqlen, device = x.shape[-1], x.device
        x = self.roberta(x, attention_mask=mask)
        if exists(aug_x1):
            # batch, seq, dim
            # eff1 = self.m.sample().cuda()
            # eff2 = self.m.sample().cuda()
            # beta1 = self.b.sample().cuda()
            # beta2 = self.b.sample().cuda()
            # with torch.no_grad():
            #     x1 = self.roberta(aug_x1, attention_mask=mask).pooler_output
            #     x2 = self.roberta(aug_x2, attention_mask=mask).pooler_output
            #     x3 = self.roberta(aug_x3, attention_mask=mask).pooler_output

            # aug1 = eff1[0] * x1 + eff1[1] * x2 + eff1[2] * x3
            # aug1 = (1 - beta1) * aug1 + beta1 * x.pooler_output
            # aug2 = eff2[0] * x1 + eff2[1] * x2 + eff2[2] * x3
            # aug2 = (1 - beta2) * aug2 + beta2 * x.pooler_output

            # aug1 = self.param[0] * x1 + self.param[1] * x2 + self.param[2] * x3
            # aug1 = self.param[3] * aug1 + self.param[4] * x.pooler_output
            # aug2 = self.param[5] * x1 + self.param[6] * x2 + self.param[7] * x3
            # aug2 = self.param[8] * aug2 + self.param[9] * x.pooler_output
            #
            # x_mid = F.log_softmax((x.pooler_output + aug1 + aug2) / 3, dim=-1)
            # aug1 = F.log_softmax(aug1, dim=-1)
            # aug2 = F.log_softmax(aug2, dim=-1)


            # kl_loss = (self.kl_fct(F.log_softmax(x.pooler_output, dim=-1), x_mid) + self.kl_fct(aug1, x_mid) + \
            #            self.kl_fct(aug2, x_mid)) / 3

            x1 = self.roberta(aug_x1, attention_mask=mask).pooler_output
            x2 = self.roberta(aug_x2, attention_mask=mask).pooler_output
            aug1 = self.param[0] * x1 + self.param[1] * x2
            aug2 = self.param[2] * x1 + self.param[3] * x2

            kl_loss = self.kl_fct(F.log_softmax(x.pooler_output, dim=-1), F.log_softmax(aug1, dim=-1)) + \
                      self.kl_fct(F.log_softmax(x.pooler_output, dim=-1), F.log_softmax(aug2, dim=-1)) + \
                      self.kl_fct(F.log_softmax(aug1, dim=-1), F.log_softmax(aug2, dim=-1))

            # aug1 = self.roberta(aug_x1, attention_mask=mask).pooler_output
            # kl_loss = self.kl_fct(F.log_softmax(x.pooler_output, dim=-1), F.log_softmax(aug1, dim=-1))

        x = self.classify(x.pooler_output)

        if exists(target):
            if exists(aug_x1):
                loss = self.loss_fct(x, target) + 0.2*kl_loss
            else:
                loss = self.loss_fct(x, target)
            return loss, x

        return 0, x

