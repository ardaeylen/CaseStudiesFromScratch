from ResNet import ResNet
import torch
if __name__ == "__main__":
    resNet18_bottleneck = ResNet(in_channels = 3,
                                 num_classes = 1000)

    foo = torch.rand(size = (32, 3, 224, 224))

    print(foo.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    resNet18_bottleneck.to(device)


    result = resNet18_bottleneck(foo)
    print(result.shape)
    print(result[0])