import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import models
from torch import optim
import torch.nn as nn
import cv2
import sys
import os
from torchvision import transforms
from operator import itemgetter

unit_idx = 256  # the neuron index to visualize
rmslist = []

for unit in range(unit_idx):
    ckpt = torch.load("./regression_Plant subset_Dataset 1_ResNet34_300epochs.pth") # regression model
    #ckpt = torch.load("./pretrained_ResNet34_008epochs.pth") # classification model
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k.replace("resnet.", "")
        new_state_dict[name] = v
    new_state_dict['fc.weight'] = torch.zeros([1000, 512]) # comment out when visualizing classification model
    new_state_dict['fc.bias'] = torch.zeros([1000]) # comment out when visualizing classification model
    model = models.resnet34(pretrained=True)
    model.load_state_dict(new_state_dict)

    # names of the different layers in the model
    print(list(map(lambda x: x[0], model.named_children())))

    # We now freeze the parameters of our pretrained model
    for param in model.parameters():
        param.requires_grad_(False)

    # components of the layer: inception4a
    model.avgpool

    # We will register a forward hook to get the output of the layers
    activation = {}  # to store the activation  of a layer

    def create_hook(name):
        def hook(m, i, o):
            # copy the output of the given layer
            activation[name] = o

        return hook

    # register a forward hook for layer inception4a i.e. the first inception layer
    model.layer3.register_forward_hook(create_hook('layer3'))

    # normalize the input image to have appropriate mean and standard deviation as specified by pytorch
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # undo the above normalization if and when the need arises
    denormalize = transforms.Normalize(mean = [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                       std = [1/0.229, 1/0.224, 1/0.225])


    # function to create an image with random pixels
    # and move it to the specified device i.e. cpu or gpu
    # by default we will move the image to the 'cpu'
    # the default will be to turn-off the requires_grad_ attribute of the image
    def random_image(Height=28, Width=28, device='cpu', requires_grad=False, optimizer=None, lr=0.01):
        img = np.single(np.random.uniform(0, 1, (3, Height, Width)))  # we need the pixel values to be of type float32
        im_tensor = normalize(torch.from_numpy(img)).to(device).requires_grad_(
            requires_grad)  # normalize the image to have requisite mean and std. dev.
        print("img_shape:{}, img_dtype: {}".format(im_tensor.shape, im_tensor.dtype))

        if optimizer:
            if requires_grad:
                return im_tensor, optimizer([im_tensor], lr=lr)
            else:
                print('Error: Optimizer cannot be used on an image without setting its requires_grad_  ')

        return im_tensor


    # function to massage img_tensor for using as input to plt.imshow()
    def image_converter(im):
        # move the image to cpu
        im_copy = im.cpu()

        # for plt.imshow() the channel-dimension is the last
        # therefore use transpose to permute axes
        im_copy = denormalize(im_copy.clone().detach()).numpy().transpose(1, 2, 0)
        # clip negative values as plt.imshow() only accepts
        # floating values in range [0,1] and integers in range [0,255]
        im_copy = im_copy.clip(0, 1)

        return im_copy

    # Use cv2 to compute image gradients
    # This is to verify our implementation of image gradients using conv. layers
    # We can not be use these during backpropagation, hence the need to define our own function
    # Due to differences in how boundary pixels are handled we only expect the results to match for non-boundary pixels


    # function to compute image gradients using cv2
    # we will make use of the Scharr filters provided in cv2
    # the image passed to this function is supposed to have the color channels as its first dimension
    def image_gradients(img):
        num_channels = img.shape[0]
        ddepth = cv2.CV_64F
        x_grad = []
        y_grad = []
        for chnl in range(num_channels):
            x_grad.append(cv2.Scharr(img[chnl], ddepth, dx = 1, dy = 0))
            y_grad.append(cv2.Scharr(img[chnl], ddepth, dx = 0, dy = 1))
        x_grad = np.single(np.array(x_grad))
        y_grad = np.single(np.array(y_grad))
        return x_grad, y_grad

    # Scharr Filters
    # for e.g. see https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html#formulation
    filter_x = np.array([[-3, 0, 3],
                         [-10, 0, 10],
                         [-3, 0, 3]])

    filter_y = filter_x.T
    grad_filters = np.array([filter_x, filter_y])

    grad_filters.shape[1:]


    # class to compute image gradients in pytorch
    class gradients(nn.Module):
        def __init__(self, weight):
            super().__init__()
            k_height, k_width = weight.shape[2:]
            # assuming that the height and width of the kernel are always odd numbers
            padding_x = int((k_height - 1) / 2)
            padding_y = int((k_width - 1) / 2)

            # convolutional layer with 2 output channels corresponding to the x and the y gradients
            self.conv = nn.Conv2d(1, 2, (k_height, k_width), bias=False,
                                  padding=(padding_x, padding_y))
            # initialize the weights of the convolutional layer to be the one provided
            if self.conv.weight.shape == weight.shape:
                self.conv.weight = nn.Parameter(weight)
                self.conv.weight.requires_grad_(False)
            else:
                print('Error: The shape of the given weights is not correct')

        def forward(self, x):
            return self.conv(x)

    gradLayer = gradients(torch.from_numpy(grad_filters).unsqueeze(1).type(torch.FloatTensor))
    gradLayer

    # generating the random image with random pixel values between 0 and 1
    H = 5 # height of input image
    W = 5 # width of input image
    img = np.single(np.random.uniform(0,1, (1, H, W))) # we need the pixel values to be of type float32 hence np.single
    print("img_shape:{}, img_dtype: {}".format(img.shape, img.dtype ))
    # convert the image to a torch tensor with the requisite mean and std. dev.
    img_tensor = torch.from_numpy(img)
    img_tensor

    image_gradients(img)[0][0, 1:4, 1:4]-gradLayer(img_tensor.unsqueeze(0)).numpy()[0, 0, 1:4, 1:4]
    image_gradients(img)[1][0, 1:4, 1:4]-gradLayer(img_tensor.unsqueeze(0)).numpy()[0, 1, 1:4, 1:4]


    # class to compute image gradients in pytorch
    class RGBgradients(nn.Module):
        def __init__(self, weight):  # weight is a numpy array
            super().__init__()
            k_height, k_width = weight.shape[1:]
            # assuming that the height and width of the kernel are always odd numbers
            padding_x = int((k_height - 1) / 2)
            padding_y = int((k_width - 1) / 2)

            # convolutional layer with 3 in_channels and 6 out_channels
            # the 3 in_channels are the color channels of the image
            # for each in_channel we have 2 out_channels corresponding to the x and the y gradients
            self.conv = nn.Conv2d(3, 6, (k_height, k_width), bias=False,
                                  padding=(padding_x, padding_y))
            # initialize the weights of the convolutional layer to be the one provided
            # the weights correspond to the x/y filter for the channel in question and zeros for other channels
            weight1x = np.array([weight[0],
                                 np.zeros((k_height, k_width)),
                                 np.zeros((k_height, k_width))])  # x-derivative for 1st in_channel

            weight1y = np.array([weight[1],
                                 np.zeros((k_height, k_width)),
                                 np.zeros((k_height, k_width))])  # y-derivative for 1st in_channel

            weight2x = np.array([np.zeros((k_height, k_width)),
                                 weight[0],
                                 np.zeros((k_height, k_width))])  # x-derivative for 2nd in_channel

            weight2y = np.array([np.zeros((k_height, k_width)),
                                 weight[1],
                                 np.zeros((k_height, k_width))])  # y-derivative for 2nd in_channel

            weight3x = np.array([np.zeros((k_height, k_width)),
                                 np.zeros((k_height, k_width)),
                                 weight[0]])  # x-derivative for 3rd in_channel

            weight3y = np.array([np.zeros((k_height, k_width)),
                                 np.zeros((k_height, k_width)),
                                 weight[1]])  # y-derivative for 3rd in_channel

            weight_final = torch.from_numpy(np.array([weight1x, weight1y,
                                                      weight2x, weight2y,
                                                      weight3x, weight3y])).type(torch.FloatTensor)

            if self.conv.weight.shape == weight_final.shape:
                self.conv.weight = nn.Parameter(weight_final)
                self.conv.weight.requires_grad_(False)
            else:
                print('Error: The shape of the given weights is not correct')

        # Note that a second way to define the conv. layer here would be to pass group = 3 when calling torch.nn.Conv2d

        def forward(self, x):
            return self.conv(x)

    # create an instance of the class 'gradients' with the given filters
    gradLayer = RGBgradients(grad_filters)


    # function to compute gradient loss of an image using the above defined gradLayer
    def grad_loss(img, beta=1, device='cpu'):
        # move the gradLayer to cuda
        gradLayer.to(device)
        gradSq = gradLayer(img.unsqueeze(0)) ** 2

        # The following was an earlier definition of grad_loss based on Mahendran & Vedaldi's paper
        # For me this seems to give rise to Nan's whose source I am unable to locate
        # so using a simpler grad_loss instead
        # grad_loss = 1/3 * (torch.pow(gradSq[0,0]+gradSq[0,1], beta/2)
        #                   + torch.pow(gradSq[0,2]+gradSq[0,3], beta/2)
        #                   + torch.pow(gradSq[0,4]+gradSq[0,5], beta/2)).mean()

        grad_loss = torch.pow(gradSq.mean(), beta / 2)

        return grad_loss

    # move the model and input image to the GPU (if available)
    # This should be done before defining an optimizer
    # This is mentioned in the following pytorch page: https://pytorch.org/docs/master/optim.html
    # Also see the discussion here: https://discuss.pytorch.org/t/effect-of-calling-model-cuda-after-constructing-an-optimizer/15165/7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('Calculations being executed on {}'.format(device))

    model.to(device)

    # initial random image
    H = 28 # height of image
    W = 28 # width of image
    img_tensor = random_image(Height = H, Width = W, device = device, requires_grad = True ) # we will define an optimizer later, inside the training loop

    # check that the model and img_tensor are on cuda or not
    # https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180
    print('googlenet is on cuda: {}'.format(next(model.parameters()).is_cuda))
    print('img_tensor is on cuda: {}'.format(img_tensor.is_cuda))

    #####new######

    act_wt = 0.5  # factor by which to weigh the activation relative to the regulizer terms

    upscaling_steps = 45  # no. of times to upscale
    upscaling_factor = 1.05
    optim_steps = 20  # no. of times to optimize an input image before upscaling


    model.eval()


    for mag_epoch in range(upscaling_steps + 1):
        optimizer = optim.Adam([img_tensor], lr=0.4)

        for opt_epoch in range(optim_steps):
            optimizer.zero_grad()
            model(img_tensor.unsqueeze(0))
            layer_out = activation['layer3']
            rms = torch.pow((layer_out[0, unit] ** 2).mean(), 0.5)
            # terminate if rms is nan
            if torch.isnan(rms):
                print('Error: rms was Nan; Terminating ...')
                sys.exit()

            # pixel intensity
            pxl_inty = torch.pow((img_tensor ** 2).mean(), 0.5)
            # terminate if pxl_inty is nan
            if torch.isnan(pxl_inty):
                print('Error: Pixel Intensity was Nan; Terminating ...')
                sys.exit()

            # image gradients
            im_grd = grad_loss(img_tensor, beta=1, device=device)
            # terminate is im_grd is nan
            if torch.isnan(im_grd):
                print('Error: image gradients were Nan; Terminating ...')
                sys.exit()

            loss = -act_wt * rms + pxl_inty + im_grd
            # print activation at the beginning of each mag_epoch
            if opt_epoch == 0:
                print('begin mag_epoch {}, activation: {}'.format(mag_epoch, rms))
            loss.backward()
            optimizer.step()

        # view the result of optimising the image
        print('end mag_epoch: {}, activation: {}'.format(mag_epoch, rms))

        img = image_converter(img_tensor)
        if mag_epoch == upscaling_steps:
            print('end mag_epoch: {}, activation: {}'.format(mag_epoch, rms))
            #plt.imshow(img)
            #plt.title('image at the end of mag_epoch: {}'.format(mag_epoch))
            #plt.show()
            plt.imsave('./layer3_cl/{}.png'.format(unit+1), img)
            rmslist.append(rms.item())

        img = cv2.resize(img, dsize=(0, 0), fx=upscaling_factor, fy=upscaling_factor).transpose(2, 0, 1)  # scale up and move the color axis to be the first
        img_tensor = normalize(torch.from_numpy(img)).to(device).requires_grad_(True)

indices, rmslist_sorted = zip(*sorted(enumerate(rmslist), key=itemgetter(1)))
print(list(rmslist_sorted))
print(list(indices))
