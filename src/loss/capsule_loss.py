"""
In the documentation I use the following notation to describe tensors shapes

b: batch size
B: number of input capsule types
C: number of output capsule types
ih: input height
iw: input width
oh: output height
ow: output width
is0: first dimension of input capsules
is1: second dimension of input capsules
os0: first dimension of output capsules
os1: second dimension of output capsules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginLoss(nn.Module):

    def __init__(self, batch_averaged=True, margin_loss_lambda=0.5, m_plus=0.9, m_minus=0.1, device="cpu"):
        """
        Batch margin loss for class existence.

        :param batch_averaged: Should the losses be averaged (True) or summed (False) over observations
                               for each minibatch.
        :param margin_loss_lambda: Hyperparameter for down-weighting the loss for missing classes.
        :param m_plus: Vector capsule length threshold for correct class.
        :param m_minus: Vector capsule length threshold for incorrect class.
        :param device: cpu or gpu tensor.
        """
        super(MarginLoss, self).__init__()
        self.batch_averaged = batch_averaged
        self.margin_loss_lambda = margin_loss_lambda
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.device = device

    def forward(self, class_caps_activations, targets):
        """
        The class batch margin loss is defined as:

                Eq. (4): L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2

        where T_k = 1 iff a class k is present.
        The lambda down-weighting the loss for absent classes stops the initial learning from shrinking the lengths
        of the activity vectors of all the class capsules.
        The batch margin loss is simply the sum of the losses of all class capsules.

        :param class_caps_activations: The capsule activations of the last capsule layer, shape [b, B].
        :param targets: One-hot encoded labels tensor, shape [b, B].

        :return: The margin loss (scalar).
        """
        #class_caps_activations = torch.exp(class_caps_activations)
        t_k = targets.type(torch.FloatTensor)
        if targets.ndim == 1:
            t_k = F.one_hot(targets, class_caps_activations.size()[1])
        zeros = torch.zeros(class_caps_activations.size())  # zeros: [b, B]
        # Use GPU if available
        zeros = zeros.to(self.device)
        t_k = t_k.to(self.device)

        margin_loss_correct_classes = t_k * (torch.max(zeros, self.m_plus - class_caps_activations) ** 2)
        margin_loss_incorrect_classes = (1 - t_k) * self.margin_loss_lambda * \
                                        (torch.max(zeros, class_caps_activations - self.m_minus) ** 2)
        margin_loss = margin_loss_correct_classes + margin_loss_incorrect_classes  # margin_loss: [b, B]
        margin_loss = torch.sum(margin_loss, dim=-1)  # margin_loss: [b]

        if self.batch_averaged:
            margin_loss = torch.mean(margin_loss)
        else:
            margin_loss = torch.sum(margin_loss)

        # margin_loss: [1]
        return margin_loss

class CapsLoss(nn.Module):

    def __init__(self, caps_loss_type, margin_loss_lambda=0.5, batch_averaged=True, 
                 m_plus=0.9, m_minus=0.1, device="cpu", writer=None):
        """
        Capsule loss.

        :param caps_loss_type: The encoder loss type (margin)
        :param margin_loss_lambda: Hyperparameter for down-weighting the loss for missing classes.
        :param batch_averaged: should the losses be averaged (True) or summed (False) over observations
                               for each mini-batch.
        :param m_plus: Vector capsule length threshold for correct class.
        :param m_minus: Vector capsule length threshold for incorrect class.
        """
        super(CapsLoss, self).__init__()
        self.caps_loss_type = caps_loss_type
        self.device = device
        self.writer = writer

        if caps_loss_type == "margin":
            self.caps_loss = MarginLoss(batch_averaged, margin_loss_lambda, m_plus, m_minus, self.device)

    def forward(self, class_caps_activations, targets):
        """
        :param class_caps_activations: The capsule activations of the last capsule layer, shape [b, B].
        :param targets: One-hot encoded labels tensor, shape [b, B].

        :return: margin loss
        """
        if self.caps_loss_type == "margin":
            caps_loss = self.caps_loss(class_caps_activations, targets)

        return caps_loss
