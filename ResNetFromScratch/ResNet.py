from SkipConnection import ResidualBlock
import torch
import torch.nn as nn
import torch.nn.functional as F

# 18 layer ResNet implementation same as 100 and 152 layered architecture (Bottleneck Architecture).
# With dimension reduction.
class ResNet(nn.Module):

    def __init__(self,
                 in_channels,
                 num_classes):

        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = 64,
                               kernel_size = (7,7),
                               stride = 2,
                               padding = 3) #Padding is 3 because this convolutional layer halves the input resolution (224 -> 112)

        self.maxPool2d = nn.MaxPool2d(kernel_size = (3,3), #This Maximum Pooling also halves the input feature map's resolution.
                                      stride = 2,
                                      padding=1) # Padding is 1 because we want the dimensions of the output to be (N, 64, 56, 56)

        # Now the dimensions of the feature map is (N, 64, 56, 56)

        self.skip_connection1 = ResidualBlock(in_channels = 64, # First 1x1 convolution layer is used for dimension reduction.
                                              out_channels=64, # Then 3x3 convolution is applied to the feature map with the same filter size.
                                              reduced_dim=64)# Lastly, dimension of the feature map is increased and padding would be required
                                               # for the skip connection.

        # Now the dimensions of the feature map is (N, 64, 56, 56)

        self.skip_connection2 = ResidualBlock(in_channels = 64, # First 1x1 convolution layer is used for dimension reduction.
                                              out_channels=64, # Then 3x3 convolution is applied to the feature map with the same filter size.
                                              reduced_dim=64)# Lastly, dimension of the feature map is increased and padding would be required
                                                    # for the skip connection.

        # Now the dimensions of the feature map is (N, 64, 56, 56)

        self.skip_connection3 = ResidualBlock(in_channels = 64, # First 1x1 convolution layer is used for dimension reduction.
                                              out_channels=128, # Then 3x3 convolution is applied to the feature map with the same filter size.
                                              reduced_dim=64)# Lastly, dimension of the feature map is increased and padding would be required
                                                    # for the skip connection.

        # Now the dimensions of the feature map is (N, 128, 28, 28)

        self.skip_connection4 = ResidualBlock(in_channels = 128, # First 1x1 convolution layer is used for dimension reduction.
                                              out_channels=128, # Then 3x3 convolution is applied to the feature map with the same filter size.12
                                              reduced_dim=64)# Lastly, dimension of the feature map is increased and padding would be required
                                                    # for the skip connection.

        # Now the dimensions of the feature map is (N, 128, 28, 28)

        self.skip_connection5 = ResidualBlock(in_channels = 128, # First 1x1 convolution layer is used for dimension reduction.
                                              out_channels=256, # Then 3x3 convolution is applied to the feature map with the same filter size.
                                              reduced_dim=64)# Lastly, dimension of the feature map is increased and padding would be required
                                                    # for the skip connection.

        # Now the dimensions of the feature map is (N, 256, 14, 14)

        self.skip_connection6 = ResidualBlock(in_channels = 256, # First 1x1 convolution layer is used for dimension reduction.
                                              out_channels=256, # Then 3x3 convolution is applied to the feature map with the same filter size.12
                                              reduced_dim=64)# Lastly, dimension of the feature map is increased and padding would be required
                                                    # for the skip connection.

        # Now the dimensions of the feature map is (N, 256, 14, 14)

        self.skip_connection7 = ResidualBlock(in_channels = 256, # First 1x1 convolution layer is used for dimension reduction.
                                              out_channels=512, # Then 3x3 convolution is applied to the feature map with the same filter size.
                                              reduced_dim=64)# Lastly, dimension of the feature map is increased and padding would be required
                                                    # for the skip connection.

        # Now the dimensions of the feature map is (N, 512, 7, 7)

        self.skip_connection8 = ResidualBlock(in_channels = 512, # First 1x1 convolution layer is used for dimension reduction.
                                              out_channels=512, # Then 3x3 convolution is applied to the feature map with the same filter size.12
                                              reduced_dim=64)# Lastly, dimension of the feature map is increased and padding would be required
                                                    # for the skip connection.


        self.conv_last = nn.Conv2d(in_channels = 512,
                                   out_channels = self.num_classes,
                                   kernel_size = (1,1),
                                   stride = 1,
                                   padding = 0)

        self.global_average_pooling = nn.AvgPool2d(kernel_size = (7,7),
                                                   stride = 1,
                                                   padding = 0) #Kernel sizes of average pooling is 7 because this is the last resolution
                                                                # of the feature map.

    def forward(self, x):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = x.to(device)
        x = self.conv1(x)
        x = self.maxPool2d(x)
        x = self.skip_connection1(x)
        x = self.skip_connection2(x)
        x = self.skip_connection3(x)
        x = self.skip_connection4(x)
        x = self.skip_connection5(x)
        x = self.skip_connection6(x)
        x = self.skip_connection7(x)
        x = self.skip_connection8(x)
        x = self.conv_last(x)
        x = self.global_average_pooling(x)
        #x = torch.flatten(x)
        x = F.softmax(x, dim=1)
        return x