import torch
import numpy as np
import math
import itertools
import torch.nn.functional as F
import scipy
import sys
import matplotlib.pyplot as plt
from common import *
from high_dim_filter import HighDimFilterFunction


class CrfRnnLayer(torch.nn.Module):
    def __init__(self, num_class, num_iter, theta_alpha, theta_beta, theta_gamma):
        super(CrfRnnLayer, self).__init__()
        self.num_class = num_class
        self.num_iter = num_iter
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma

        self.spatial_ker_weights = torch.nn.Parameter(torch.FloatTensor(1))
        self.bilateral_ker_weights = torch.nn.Parameter(torch.FloatTensor(1))
        self.compatibility_matrix = torch.nn.Parameter(torch.FloatTensor(2))
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1))
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1))
        self.w3 = torch.nn.Parameter(torch.FloatTensor(1))
        self.w4 = torch.nn.Parameter(torch.FloatTensor(1))
        self.high_dim_filter = HighDimFilterFunction.apply
        # self.spatial_ker_weights = torch.FloatTensor(1).cuda()
        # self.bilateral_ker_weights = torch.FloatTensor(1).cuda()
        # self.compatibility_matrix = torch.FloatTensor(2).cuda()
        # self.w1 = torch.FloatTensor(1).cuda()
        # self.w2 = torch.FloatTensor(1).cuda()
        # self.w3 = torch.FloatTensor(1).cuda()

    def instanciate(self, num_instance):
        """
        Dynamically instantiate kernel weights and compatibility matrix
        num_instance: number of instances given by the detector
        """
        spatial_ker_weights = torch.zeros(num_instance, num_instance).cuda()
        bilateral_ker_weights = torch.zeros(num_instance, num_instance).cuda()
        compatibility_matrix = torch.zeros(num_instance, num_instance).cuda()

        for i in range(num_instance):
            for j in range(num_instance):
                if i == j:
                    # spatial_ker_weights[i, j] = self.spatial_ker_weights[0]
                    # bilateral_ker_weights[i, j] = self.bilateral_ker_weights[0]
                    compatibility_matrix[i, j] = self.compatibility_matrix[0]
                else:
                    # spatial_ker_weights[i, j] = self.spatial_ker_weights[1]
                    # bilateral_ker_weights[i, j] = self.bilateral_ker_weights[1]
                    compatibility_matrix[i, j] = self.compatibility_matrix[1]

        spatial_ker_weights = self.spatial_ker_weights
        bilateral_ker_weights = self.bilateral_ker_weights

        return spatial_ker_weights, bilateral_ker_weights, compatibility_matrix

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.uniform_(0., 0.05)
        print('[CRFRNN Layer] reset paramter')
        self.spatial_ker_weights.data[0] = 2.
        # self.spatial_ker_weights.data[1] = 0
        self.bilateral_ker_weights.data[0] = 2.
        # self.bilateral_ker_weights.data[1] = 0
        self.compatibility_matrix.data[0] = 0
        self.compatibility_matrix.data[1] = 1.
        self.w1.data[0] = 0.1
        self.w2.data[0] = 0.1
        self.w3.data[0] = 0.4
        self.w4.data[0] = 0.5

    def forward(self, unary, rgb):
        """
        unary: the per class probability, unary classification, [batch, num_class, H, W]
        rgb: original image, [batch, C, H, W]

        only support one image at a time
        """

        assert unary.shape[0] == 1
        assert rgb.shape[0] == 1
       

        # Reshape to [H, W, num_class]
        unary = unary[0].permute([1, 2, 0]).contiguous()
        rgb = rgb[0].permute([1, 2, 0]).contiguous()

        H, W, num_class = unary.shape
        _, _, channel = rgb.shape

        spatial_ker_weights, bilateral_ker_weights, compatibility_matrix = self.instanciate(num_class)


        # Get two pairwise kernels, bilateral filter is the apperance kernel, spatial filter is the smoothness kernel
        ones = torch.ones(H, W, num_class)
        bilateral_norm = self.high_dim_filter(ones, rgb, True,
                                              self.theta_alpha, self.theta_beta, self.theta_gamma)
        spatial_norm = self.high_dim_filter(ones, rgb, False,
                                              self.theta_alpha, self.theta_beta, self.theta_gamma)

        unaries = unary
        q_values = unaries
        for i in range(self.num_iter):
            # Normalize unary potential
            softmax_out = F.softmax(q_values, dim=-1)

            # Spatial filtering
            # [num_class, N] dot [N, N]
            spatial_out = self.high_dim_filter(softmax_out, rgb, False,
                                              self.theta_alpha, self.theta_beta, self.theta_gamma)
            spatial_out = spatial_out / spatial_norm

            # Bilateral filtering
            # [num_class, N] dot [N, N]
            bilateral_out = self.high_dim_filter(softmax_out, rgb, True,
                                              self.theta_alpha, self.theta_beta, self.theta_gamma)
            bilateral_out = bilateral_out / bilateral_norm

            # Weighting filter outputs
            # [num_class, num_class] dot [num_class, N] + num_class, num_class] dot [num_class, N]
            message = bilateral_ker_weights * (bilateral_out.view(-1, num_class)) + \
                      spatial_ker_weights * (spatial_out.view(-1, num_class))
            # message = bilateral_ker_weights.mm(bilateral_out) + spatial_ker_weights.mm(spatial_out)

            # Compatability transform
            # [num_class, num_class] dot [num_class, N]
            pairwise = message.mm(compatibility_matrix)
            pairwise = pairwise.view(H, W, num_class)

            # Adding unary potentials
            q_values = unaries - pairwise
            # q_values = unaries

        self.bilateral_out = bilateral_out
        self.spatial_out = spatial_out
        self.message = message
        self.pairwise = pairwise

        # q_values = F.softmax(q_values, dim=0)
        return q_values.permute([2, 0, 1]).contiguous()

# Multiple unary
def logit(p, eps=1e-6):
    p[p == 0] = eps
    return torch.log(p)


def get_shape_template(boxes, shapes, segmentation):
    '''
    :param boxes: [num_box, 4]
    :param shapes: [num_aspect, num_template_per_scale]
    :param segmentation: [num_class, H, W]
    :return: [npy_array_template] * num_box
    '''
    segmentation = segmentation.cpu().data.numpy()
    res = []
    cross_corelation = np.zeros((len(boxes), n_template_per_scale))
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        H, W = y1 - y0, x1 - x0
        dist = np.abs(aspect_ratio - float(H) / W)
        idx = np.argmin(dist)
        for j in range(len(shapes[idx])):
            template = scipy.misc.imresize(shapes[idx][j], (H, W), interp='bilinear')
            template = template / 255.0  # Since scipy.resize would convert it to 0 ~ 255
            # template = (template > 0.5).astype(np.int32)
            cross_corelation[i, j] = (template * segmentation[1][y0:y1, x0:x1]).sum() / \
                                     (template.sum() + segmentation[1][y0:y1, x0:x1].sum())
        res.append(shapes[idx][np.argmax(cross_corelation[i])])

    return res


def get_unary(segmentation, boxes, shapes, mask, w1, w2, w3, w4):
    assert segmentation.shape[0] == 1

    segmentation = segmentation[0]
    num_class, H, W = segmentation.shape
    num_instance = len(boxes) + 1
    glob_term = segmentation[1, :, :]
    glob_term = glob_term.repeat(num_instance, 1, 1)
    glob_term[0, :, :] = segmentation[0, :, :]
    box_term = torch.zeros(num_instance, H, W).cuda()
    mask_term = torch.zeros(num_instance, H, W).cuda()

    boxes = boxes.astype(np.int32)
    for i, box in enumerate(boxes):
        x0, y0, x1, y1, s = box[0], box[1], box[2], box[3], box[4]
        box_term[i + 1, y0:y1, x0:x1] = segmentation[1][y0:y1, x0:x1] * float(s)
        mask_term[i + 1, y0:y1, x0:x1] = segmentation[1][y0:y1, x0:x1] * mask[i, y0:y1, x0:x1]

    if shapes == None:
        shape_term = torch.zeros(num_instance, H, W).cuda()
    else:
        shape_templates = get_shape_template(boxes, shapes, segmentation)
        shape_term = torch.zeros(num_instance, H, W).cuda()
        for i, box in enumerate(boxes):
            x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
            H, W = y1 - y0, x1 - x0
            shape = shape_templates[i]
            shape = scipy.misc.imresize(shape, (H, W), interp='bilinear')
            # shape = (shape > 0.5).astype(np.int32)
            shape = torch.from_numpy(shape).float().cuda()
            shape = shape / 255.0 # Since scipy.resize would convert it to 0 ~ 255
            # print(shape.max())
            shape_term[i + 1, y0:y1, x0:x1] = segmentation[1][y0:y1, x0:x1] * shape
            # plt.subplot(121)
            # plt.imshow(segmentation.cpu().data.numpy()[1][y0:y1, x0:x1])
            # plt.title('Original nuclei')
            # plt.subplot(122)
            # plt.imshow(shape)
            # plt.title('Matched shape')
            # plt.show()


    # unary = torch.exp(w1 * box_term + w2 * glob_term + w3 * shape_term)
    unary = torch.log(w2 * glob_term + w3 * shape_term + w4 * mask_term)
    # unary = w1 * box_term + w2 * glob_term + w3 * shape_term
    # unary = w1 * logit(box_term) + w2 * logit(glob_term) + w3 * logit(shape_term)


    return unary.unsqueeze(0)

def get_spatial_filter(input, gamma):
    """
    input: original image, [C, H, W] 
    gamma:

    return: spatial kernel matrix, [N, N] where N = H * W, each element in the matrix equals k(i, j)
    """
    C, H, W = input.shape
    oh = np.arange(H)
    ow = np.arange(W)
    vec = []

    for h, w in itertools.product(oh, ow):
        vec.append([h, w])

    vec = torch.FloatTensor(vec).cuda()
    kernel_output = pairwise_distances(vec, vec)
    kernel_output = torch.exp(- kernel_output / (2 * gamma ** 2))
    return kernel_output

def get_bilateral_filter(input, alpha, beta):
    """
    input: original image, [C, H, W] 
    alpha:
    beta:

    return: bilateral kernel matrix, [N, N] where N = H * W, each element in the matrix equals k(i, j)
    """
    C, H, W = input.shape
    oh = np.arange(H)
    ow = np.arange(W)
    spatial = []
    rgb = []

    for h, w in itertools.product(oh, ow):
        spatial.append([h, w])
        rgb.append([input[0, h, w], input[1, h, w], input[2, h, w]])

    spatial = torch.FloatTensor(spatial).cuda()
    spatial_output = pairwise_distances(spatial, spatial)
    rgb = torch.FloatTensor(rgb).cuda()
    rgb_output = pairwise_distances(rgb, rgb)

    kernal_output = torch.exp(- spatial_output / (2 * alpha ** 2) - rgb_output / (2 * beta ** 2))
    return kernal_output


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def check():
    num_class, num_iter, theta_alpha, theta_alpha, theta_gamma = 2, 10, 160.0, 3., 3.
    l = CrfRnnLayer(num_class, num_iter, theta_alpha, theta_alpha, theta_gamma)
    l = l.cuda()

    C, H, W = 3, 128, 128
    input = torch.ones((1, C, H, W)).cuda()
    score = torch.ones((1, num_class, H, W)).cuda()
    target = torch.zeros((H*W)).long().cuda()
    target[:300] = 1
    target[300:] = 0

    output = l.forward(score, input)
    output = output.view(num_class, H*W).transpose(0, 1).contiguous()

    loss = F.cross_entropy(output, target)
    loss.backward

def check_spatial():
    data_dir = '/home/htang6/workspace/UNIT/outputs/stage1_train_transfered/0525_1500'
    id = '1c8b905c9519061d6d091e702b45274f4485c80dcf7fb1491e6b2723f5002180.jpg'
    import matplotlib.pyplot as plt
    import os
    image = plt.imread(os.path.join(data_dir, id))

    image_torch = torch.from_numpy(image).permute([2,0,1])[:, 100:228, 100:228].float()

    input = image_torch.contiguous()
    plt.imshow(input.permute([1,2,0]))
    input = input.view(3, -1)

    spatial_kernel = get_spatial_filter(image_torch, gamma=3.0)
    # spatial_kernel[spatial_kernel < 0.5] = 0
    # spatial_kernel = (spatial_kernel > 0).float() / 4.0
    plt.imshow(spatial_kernel[0, :].contiguous().view(128, 128))
    plt.colorbar()

    output = input.mm(torch.t(spatial_kernel))
    output = output.view(3, 128, 128)
    plt.imshow(output.permute([1,2,0]))


    return 0

def check_loss():
    data_dir = '/home/htang6/workspace/UNIT/outputs/stage1_train_transfered/0525_1500'
    id = '1c8b905c9519061d6d091e702b45274f4485c80dcf7fb1491e6b2723f5002180.jpg'
    import matplotlib.pyplot as plt
    import os
    image = plt.imread(os.path.join(data_dir, id))

    image_torch = torch.from_numpy(image).permute([2,0,1])[:, 100:228, 100:228].float()
    image_torch = image_torch.unsqueeze(0)

    unary = - torch.rand(1,2,128,128)
    crf = CrfRnnLayer(2, 5, 160., 3., 3.)

    output = crf(unary, image_torch)
    output = F.softmax(output, 0)

    label = (torch.rand(128*128) > 0.5).long()
    loss = F.cross_entropy(output.permute([1,2,0]).view(-1, 2), label)

    return

def main():
    check_loss()


if __name__ == '__main__':
    main()
