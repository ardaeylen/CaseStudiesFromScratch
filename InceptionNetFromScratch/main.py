# Inception Net Architecture
# Input image has 224x224 resolution and  RGB images.
import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self,
                 in_resolution,
                 in_channels,
                 dim_reduction_for3,
                 dim_reduction_for5,
                 out_channels_3,
                 out_channels_5,
                 out_channels_1,
                 out_channels_pool_proj,
                 prediction = False):
        super(InceptionModule,self).__init__()

        # Number of output channels for each scale after the convolution operation of multiple scales.
        self.out_channels_1x1  = out_channels_1
        self.out_channels_3x3 = out_channels_3
        self.out_channels_5x5 = out_channels_5
        self.out_channels_pool_proj = out_channels_pool_proj
        # To calculate the padding for different kernel sized convolutional layers to produce same sized feature maps as input.
        # In other words, to apply the feature maps same convolution.
        self.input_resolution = in_resolution
        # Number of output channels for the previous layer's activation before input to the multi-scale convolution
        # for the purpose of dimension reduction to gain efficiency.
        self.reduced_3x3_dim = dim_reduction_for3
        self.reduced_5x5_dim = dim_reduction_for5

        self.max_pool_3x3 = nn.MaxPool2d(kernel_size = (3,3), stride=1, padding = (1,1) ) # Since the filter size of max pooling is 3,
                                                                                    # padding should be ( (f - 1) / 2 ) to produce
                                                                                    # same size feature map.


        self.reduce_dim_for_3x3Conv = nn.Conv2d(in_channels = in_channels, # There is no need to be worry about the output resolution
                                                out_channels = self.reduced_3x3_dim,# because it is a 1x1 Conv Layer with stride 1.
                                                kernel_size= (1,1),
                                                stride = 1)
        self.reduce_dim_for_5x5Conv = nn.Conv2d(in_channels=in_channels, # There is no need to be worry about the output resolution
                                                out_channels=self.reduced_5x5_dim,# because it is a 1x1 Conv Layer with stride 1.
                                                kernel_size=(1, 1),
                                                stride=1)

        # Multi-scale convolution filters for the Inception Module.
        self.conv1x1  = nn.Conv2d(in_channels = in_channels,
                                           out_channels = self.out_channels_1x1,
                                           kernel_size = (1,1),
                                           stride = 1)

        self.conv5x5 = nn.Conv2d(in_channels = self.reduced_5x5_dim, # Padding should be (5 - 1) / 2 = 2 to produce same resolution
                                 out_channels = self.out_channels_5x5,# feature map.
                                 kernel_size = (5,5),
                                 stride = (1, 1),
                                 padding= (2,2))

        self.conv3x3 = nn.Conv2d(in_channels = self.reduced_3x3_dim,# Padding should be (3 - 1) / 2 = 1 to produce same resolution
                                 out_channels = self.out_channels_3x3,# feature map.
                                 kernel_size = (3, 3),
                                 stride = (1,1),
                                 padding = (1,1))


        self.pool_proj = nn.Conv2d(in_channels = in_channels, out_channels = self.out_channels_pool_proj, kernel_size=(1,1))

    def forward(self, previous_activations):
        # 1x1 convolution is applied to the previous activation to reduce the dimension of the input before 3x3 convolution.
        reduced_input_for_3x3 = F.relu(self.reduce_dim_for_3x3Conv(previous_activations))
        # 1x1 convolution is applied to the previous activation to reduce the dimension of the input before 5x5 convolution.
        reduced_input_for_5x5 = F.relu(self.reduce_dim_for_5x5Conv(previous_activations))
        # External 1x1 convolution is applied to be concatenated with multi-scale output as well as taking advantage of ReLU activation.
        output_1x1 = F.relu(self.conv1x1(previous_activations))
        # Maximum Pooling  (3x3) operation is applied before 1x1.
        pooled_input = self.max_pool_3x3(previous_activations)


        # Multi-Scale Convolution will be applied to the reduced dimension input.

        output_3x3 = F.relu(self.conv3x3(reduced_input_for_3x3))
        output_5x5 = F.relu(self.conv5x5(reduced_input_for_5x5))
        pool_proj = F.relu(self.pool_proj(pooled_input))

        # Concatenation of different feature maps obtained from convolutional layers that have different (except for the two 1x1 conv layer)
        # kernel/filter sizes (scales) for the next layer to abstract features from multiple scales simultaneously.
        # So the dimension of the input feature map is (batch_size, num_channels, height, width) and the num_channels would
        # change depend on the convolution layer of different scales. So we want different feature maps gathered from different
        # convolution layer to be concatenated on the channel of the feature maps.
        output = torch.cat([output_1x1, output_3x3, output_5x5, pool_proj], dim=1)

        return output

class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()

        # Inıtial layers are same as traditional convolutional layers.
        self.conv1 = nn.Conv2d(in_channels = 3, # Shape of the output of first conv layer is (112, 112, 64) ->(width, height, num_channels)
                                 out_channels = 64, # In order to produce the output we should have a padding of 3 because
                                 kernel_size=(7,7), # (224 - 7 + 3*2) / 2 = 112
                                 stride=(2,2),
                                 padding=3)

        self.max_pool1 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1)
        # This max pooling produces: (112, 112, 64) -> (56, 56, 64)

        self.resp_norm1 = nn.LocalResponseNorm(16)
        self.reduce1 = nn.Conv2d(in_channels = 64,
                                 out_channels =64,
                                 kernel_size = (1,1),
                                 stride=1,
                                 padding = 0)

        self.conv2 = nn.Conv2d(in_channels = 64, #Same Convolution is applied with 192 filters with patch/kernel size = 3x3
                               out_channels = 192,
                               kernel_size = (3,3),
                               stride = 1,
                               padding = 1)

        self.resp_norm2 = nn.LocalResponseNorm(32)

        self.max_pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1) # (56,56,192) -> (28,28,192)

        self.inception_module1 = InceptionModule(in_resolution=28,
                                                 in_channels = 192,
                                                 dim_reduction_for3=96,
                                                 dim_reduction_for5=16,
                                                 out_channels_3 = 128, # After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=32,
                                                 out_channels_1=64,
                                                 out_channels_pool_proj=32)
        # Now the dimensions (28, 28, 256)
        self.inception_module2 = InceptionModule(in_resolution=28,
                                                 in_channels = 256,
                                                 dim_reduction_for3=128,
                                                 dim_reduction_for5=32,
                                                 out_channels_3 = 192, # After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=96,
                                                 out_channels_1=128,
                                                 out_channels_pool_proj=64)
        # Now the dimensions (28,28,480)
        # Pooling for reducing resolution.
        self.max_pool3 = nn.MaxPool2d(kernel_size = (3,3), stride = 2, padding=1) # Padding is set to 1 in order to get 14x14 resolution.

        #Now the dimensions (14,14,480)

        self.inception_module3 = InceptionModule(in_resolution=14,
                                                 in_channels = 480,
                                                 dim_reduction_for3=96,
                                                 dim_reduction_for5=16,
                                                 out_channels_3 = 208, # After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=48,
                                                 out_channels_1=192,
                                                 out_channels_pool_proj=64)
        # Now the dimensions (14,14, 512)

        self.inception_module4 = InceptionModule(in_resolution=14,
                                                 in_channels = 512,
                                                 dim_reduction_for3=112,
                                                 dim_reduction_for5=24,
                                                 out_channels_3 = 224, # After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=64,
                                                 out_channels_1=160,
                                                 out_channels_pool_proj=64)

        # Now the dimensions (14, 14, 512)


        self.inception_module5 = InceptionModule(in_resolution=14,
                                                 in_channels = 512,
                                                 dim_reduction_for3=128,
                                                 dim_reduction_for5=24,
                                                 out_channels_3 = 256, # After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=64,
                                                 out_channels_1=128,
                                                 out_channels_pool_proj=64)
        # Now the dimensions (14, 14, 512)

        self.inception_module6 = InceptionModule(in_resolution=14,
                                                 in_channels = 512,
                                                 dim_reduction_for3=144,
                                                 dim_reduction_for5=32,
                                                 out_channels_3 = 288, # After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=64,
                                                 out_channels_1=112,
                                                 out_channels_pool_proj=64)
        #Now the dimensions (14, 14, 528)

        self.inception_module7 = InceptionModule(in_resolution=14,
                                                 in_channels = 528,
                                                 dim_reduction_for3=160,
                                                 dim_reduction_for5=32,
                                                 out_channels_3 = 320, # After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=128,
                                                 out_channels_1=256,
                                                 out_channels_pool_proj=128)


        # Now the dimensions (14,14,832)
        # Maximum Pooling for resolution reduction
        self.max_pool4 = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1) #Padding is 1 in order to get (7,7,832)

        self.inception_module8 = InceptionModule(in_resolution=7,
                                                 in_channels=832,
                                                 dim_reduction_for3=160,
                                                 dim_reduction_for5=32,
                                                 out_channels_3=320,    # After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=128,
                                                 out_channels_1=256,
                                                 out_channels_pool_proj=128)
        # Now the dimensions (7, 7, 832)

        self.inception_module9 = InceptionModule(in_resolution=7,
                                                 in_channels=832,
                                                 dim_reduction_for3=192,
                                                 dim_reduction_for5=48,
                                                 out_channels_3=384,# After inception module, channel dim = 128 + 32 + 64 + 32 = 256
                                                 out_channels_5=128,
                                                 out_channels_1=384,
                                                 out_channels_pool_proj=128)
        # Now the dimensions (7, 7, 1024)
        #Average Pooling
        self.avg_pool1 = nn.AvgPool2d(kernel_size = (7,7), stride=1)
        self.dropout = nn.Dropout(p=0.4)

        # Linear layer after flatten the feature map.
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=1024, out_features=1000)

        # Prediction

    def forward(self, x):
            # Initial network configuration is same as traditional ConvNets (e.g. AlexNet) with Local Response Normalization.
            x = self.conv1(x)
            x = self.max_pool1(x)
            x = self.resp_norm1(x)
            x = self.reduce1(x)
            x = self.conv2(x)
            x = self.resp_norm2(x)
            x = self.max_pool2(x)
            x = self.inception_module1(x)
            x = self.inception_module2(x)
            x = self.max_pool3(x)
            x = self.inception_module3(x)
            x = self.inception_module4(x)
            x = self.inception_module5(x)
            x = self.inception_module6(x)
            x = self.inception_module7(x)
            x = self.max_pool4(x)
            x = self.inception_module8(x)
            x = self.inception_module9(x)
            x = self.avg_pool1(x)
            x = self.dropout(x)
            x = self.flatten(x)
            x = F.softmax(self.linear(x), dim=1)
            return x

if __name__ == '__main__':
    image = torch.rand((2,3,224,224))
    print(image.shape)

    model = InceptionNet()
    output = model(image)
    print(output)
