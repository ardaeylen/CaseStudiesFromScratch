import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 reduced_dim):
        super(ResidualBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduced_dim = reduced_dim
        #Convolutional Layers of the residual block.
        # Remember that we first take the input and store it for further identity mapping
        # while convolve it two times.
        # Let g be the ReLU activation function and f is the convolution operation. Then,

        # output <- g(x + f(g(f(x)))) is the output of the residual connection.

        #Convolutional layers should not change the input resolution for further identity mapping.
        # If a convolutional layer or a pooling layer has changed the input resolution, then it
        # then padding or linear projection must be applied before addition.
        # General approach is that using a same convolution in order not to handle further upscaling.

        # Dimension reduction to gain efficiency on computational complexity and further preparation
        # For the bottleneck layer.
        self.reducer = nn.Conv2d(in_channels = self.in_channels,
                                 out_channels = self.reduced_dim,
                                 kernel_size = (1,1),
                                 stride = 1,
                                 padding = 0)

        # Bottleneck (Same Convolution is applied to input feature map.)
        self.conv1 = nn.Conv2d(in_channels = self.reduced_dim,
                               out_channels = self.reduced_dim,
                               kernel_size= (3,3),
                               stride = 1,
                               padding = 1)

        # Projection Step for the channel of the resulting feature map to be increased.
        self.projector = nn.Conv2d(in_channels = self.reduced_dim,
                               out_channels = self.out_channels,
                               kernel_size = (1,1),
                               stride = 1,
                               padding = 0)

        # Conv2 is used when stride is 2 to reduce the input resolution and it approximately halves the resolution.
        # Paper claims that when reducing the input resolution they doubled the channels size (width of the network).
        # The idea that doubling the number of feature maps and halving the feature map resolution comes from VGG-16 paper.
        self.conv2 = nn.Conv2d(in_channels = self.reduced_dim,
                               out_channels = self.reduced_dim,
                               kernel_size = (3,3),
                               stride = 2,
                               padding = 1)

        # When input resolution and channel size of the feature map changes, input dimensions should also be changed for identity mapping.
        self.conv_half = nn.Conv2d(in_channels = in_channels,
                                   out_channels = out_channels,
                                   kernel_size = (1,1),
                                   stride = 2,
                                   padding = 0)

    def forward(self,x):

        f = F.relu(self.reducer(x))
        if self.in_channels != self.out_channels:
            # Resolution would be halved and channel size will be doubled in paper at each dimension changes.
            f = F.relu(self.conv2(f)) # This convolution layer halves the resolution of the input feature map.

            x = F.relu(self.conv_half(x)) #This convolution layer halves the resolution of the input and doubles the channel size for skip connection.

        else:
            f = F.relu(self.conv1(f))

        f = F.relu(self.projector(f))

        # Identity mapping step
        output = torch.add(x, f)
        return output