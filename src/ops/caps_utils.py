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
import ops.utils as ops
from torch.nn.modules.utils import _pair

def routing(routing_method, num_iterations, votes, logits, routing_bias, input_caps_activations,
            beta_a=None, beta_u=None, squashing="hinton", coupl_coeff=False, binning=False, bins=None):
    """
    Active capsules at one level make predictions, via transformation matrices, for the instantiation parameters of
    higher-level capsules. When multiple predictions agree, a higher level capsule becomes active.
    To achieve these results we use an iterative routing-by-agreement mechanism.
    :param routing_method: The iterative routing-by-agreement mechanism method: dynamic or EM.
    :param num_iterations: The number of routing iterations.
    :param votes: The votes from lower-level capsules to higher-level capsules,
                  shape [b, C, oh, ow, B, kh, kw, os0, os1].
    :param logits: The coupling coefficients that are determined by the routing process, shape [b, C, oh, ow, B, kh, kw]
    :param routing_bias: The routing biases (only for dynamic routing), shape [B, oh, ow, os0, os1].
    :param input_caps_activations: The capsules activations tensor of layer L, shape [b, B, ih, iw].
    :param beta_a: Parameter (one for each output caps type, only for EM routing), shape [C].
    :param beta_u: Parameter (one for each output caps type, only for EM routing), shape [C].
    :param squashing: The non-linear function to ensure that short vectors get shrunk to almost zero length and
                      long vectors get shrunk to a length slightly below 1 (only for dynamic routing).
    :return: (output_caps_poses, output_caps_activations)
             The capsules poses and activations tensors of layer L + 1.
             output_caps_poses: [b, C, oh, ow, os0, os1], output_caps_activations: [b, C, oh, ow]
    """
    assert num_iterations > 0

    if routing_method == "dynamic":
        return dynamic_routing(num_iterations, votes, logits, routing_bias, squashing, coupl_coeff,binning, bins)

def dynamic_routing(num_iterations, votes, logits, routing_bias, squashing="hinton", coupl_coeff=False, binning=False, bins=None):
    """
    A lower-level capsule prefers to send its output to higher level capsules whose activity vectors have a big scalar
    product with the prediction coming from the lower-level capsule.

    :param num_iterations: The number of routing iterations.
    :param votes: The votes from lower-level capsules to higher-level capsules,
                  shape [b, C, oh, ow, B, kh, kw, os0, os1].
    :param logits: The coupling coefficients that are determined by the routing process, shape [b, C, oh, ow, B, kh, kw]
    :param routing_bias: The routing biases (only for dynamic routing), shape [C, oh, ow, os0, os1].
    :param squashing: The non-linear function to ensure that short vectors get shrunk to almost zero length and
                      long vectors get shrunk to a length slightly below 1 (only for dynamic routing).

    :return: (output_caps_poses, output_caps_activations)
             The capsules poses and activations tensors of layer L + 1.
             output_caps_poses: [b, C, oh, ow, os0, os1], output_caps_activations: [b, C, oh, ow]
    """
    batch_size = votes.size(0)
    output_height = logits.size(2)
    output_width = logits.size(3)
    output_caps_types = logits.size(1)
    input_caps_types = votes.size(4)
    kernel_size = (votes.size(5), votes.size(6))
    input_caps = input_caps_types * votes.size(6) * votes.size(6)

    votes_detached = votes.detach()
    # Dynamic routing core
    for it in range(num_iterations):
        logits_r = logits.view(batch_size, output_caps_types, output_height, output_width, -1)
        coupling_coeff = torch.softmax(logits_r, dim=-1)  # logits: [b, C, oh, ow, B, kh, kw]
        coupling_coeff = coupling_coeff.view(logits.size())
        if it == num_iterations-1:
            weighted_votes = votes * coupling_coeff[:, :, :, :, :, :, :, None, None]
        else:
            weighted_votes = votes_detached * coupling_coeff[:, :, :, :, :, :, :, None, None]
        # weighted_votes: [b, C, oh, ow, B, kh, kw, os0, os1]
        output_caps_poses = torch.sum(weighted_votes, dim=(4, 5, 6)) + routing_bias
        # output_caps_poses: [b, C, oh, ow, os0, os1]
        output_caps_poses = squash(output_caps_poses, squashing)
        # output_caps_poses: [b, C, oh, ow, os0, os1]

        # similarities: [b, C, oh, ow, B, kh, kw]
        if num_iterations > 1 and it < num_iterations - 1:
            similarities = torch.matmul(output_caps_poses.view(batch_size, output_caps_types, output_height, output_width,
                                                1, 1, 1, 1, -1),
                                        votes_detached.view(batch_size, output_caps_types, output_height, output_width,
                                                input_caps_types, kernel_size[0], kernel_size[1], -1, 1))
            # similarities: [b, C, oh, ow, B, kh, kw, 1, 1]
            similarities = similarities.squeeze(-1).squeeze(-1)
            logits = logits + similarities
            # logits: [b, C, oh, ow, B, kh, kw]

    output_caps_activations = caps_activations(output_caps_poses)

    return output_caps_poses, output_caps_activations

def caps_activations(caps_poses):
      return caps_poses.norm(dim=(-2,-1))

def squash(caps_poses, squashing_type):
    """
    The non-linear function to ensure that short vectors get shrunk to almost zero length and
    long vectors get shrunk to a length slightly below 1 (only for dynamic routing).

    :param caps_poses: The capsules poses, shape [b, B, ih, iw, is0, is1]
    :param squashing_type: The squashing type

    :return: The capsules poses squashed, shape [b, B, ih, iw, is0, is1]
    """
    if squashing_type == "hinton":
        squared_norm = torch.sum(caps_poses ** 2, dim=(-1, -2), keepdim=True)
        norm = torch.sqrt(squared_norm+1e-6)
        #print(torch.any(norm.isnan()))
        scale = squared_norm / (1 + squared_norm)
        caps_poses = scale * caps_poses / norm
        return caps_poses

def compute_votes(input_caps_poses, transform_matr, kernel_size, stride, output_caps_shape, device):
    """
    The convolution operation between capsule layers useful to compute the votes.
    :param input_caps_poses: The input capsules poses, shape [b, B, ih, iw, is0, is1]
    :param transform_matr: The transformation matrices, shape [B, kh, kw, C, os0, is0]
    :param kernel_size: The size of the receptive fields, a single number or a tuple.
    :param stride: The stride with which we slide the filters, a single number or a tuple.
    :param output_caps_shape: The shape of the higher-level capsules.
    :param device: cpu or gpu tensor.
    :return: The votes from lower-level capsules to higher-level capsules, shape [b, C, oh, ow, B, kh, kw, os0, os1].
    """
    batch_size = input_caps_poses.size(0)
    input_caps_types = input_caps_poses.size(1)
    input_height = input_caps_poses.size(2)
    input_width = input_caps_poses.size(3)
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)

    output_height, output_width = ops.conv2d_output_shape((input_height, input_width), kernel_size, stride)
    input_caps_shape = (input_caps_poses.size(-2), input_caps_poses.size(-1))

    # used to store every capsule i's poses in each capsule c's receptive field
    poses = torch.stack([input_caps_poses[:, :, stride[0] * i:stride[0] * i + kernel_size[0], stride[1] * j:stride[1] * j + kernel_size[1], :, :]
                         for i in range(output_height) for j in range(output_width)], dim=-1)
    # poses: [b, B, kh, kw, is0, is1, oh * ow]
    poses = poses.permute(0, 1, 2, 3, 6, 4, 5)
    # poses: [b, B, kh, kw, oh * ow, is0, is1]
    poses = poses.view(batch_size, input_caps_types, kernel_size[0], kernel_size[1],
                       1, output_height, output_width, input_caps_shape[0], input_caps_shape[1])
    # poses: [b, B, kh, kw, 1, oh, ow, is0, is1]

    transform_matr = transform_matr[None, :, :, :, :, None, None, :, :]
    # transform_matr: [1, B, kh, kw, C, 1, 1, os0, is0]
    votes = torch.matmul(transform_matr, poses)
    # votes: [b, B, kh, kw, C, oh, ow, os0, os1] is1 and os1 should be equals
    return votes.permute(0, 4, 5, 6, 1, 2, 3, 7, 8)  # votes: [b, C, oh, ow, B, kh, kw, os0, os1]