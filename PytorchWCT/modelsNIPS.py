import torch.nn as nn
import torch

class encoder(nn.Module):
    def __init__(self, enc_number, vgg):
        super(encoder,self).__init__()

        num_layers = [2, 4, 6, 10, 14]
        num_layers = num_layers[enc_number - 1]
        convs = []
        reflecPads = [nn.ReflectionPad2d((1,1,1,1)) for _ in range(num_layers - 1)]
        maxPools = [nn.MaxPool2d(kernel_size=2,stride=2,return_indices = True) for _ in range([0, 1, 2, 3, 4][enc_number - 1])]
        relus = [nn.ReLU(inplace=True) for _ in range(num_layers - 1)]
        sizes = [3, 3, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512]
        nums_load = [0, 2, 5, 9, 12, 16, 19, 22, 25, 29, 32, 35, 38, 42]

        for i, num_load in enumerate(nums_load[:num_layers]):
            kernel_size = 1 if i == 0 else 3
            conv = torch.nn.Conv2d(sizes[i], sizes[i + 1], kernel_size, 1, 0)
            # load weights and biases
            conv.weight = torch.nn.Parameter(torch.tensor(vgg.modules[num_load].weight).float())
            conv.bias = torch.nn.Parameter(torch.tensor(vgg.modules[num_load].bias).float())
            convs += [conv]

        self.enc_number = enc_number
        self.num_layers = num_layers
        self.convs = nn.ModuleList(convs)
        self.reflecPads = nn.ModuleList(reflecPads)
        self.relus = nn.ModuleList(relus)
        self.maxPools = nn.ModuleList(maxPools)

    def forward(self,x):
        out = x
        pool_index = 0
        for i in range(self.num_layers):
            if i > 1:
                out = self.reflecPads[i - 1](out)
            out = self.convs[i](out)
            if i == 0:
                out = self.reflecPads[i](out)
            else:
                out = self.relus[i - 1](out)
            if i in [2, 4, 8, 12]:
                out, _ = self.maxPools[pool_index](out)
                pool_index += 1
        return out

class decoder(nn.Module):
    def __init__(self, dec_number, d):
        super(decoder,self).__init__()

        num_layers = [1, 3, 5, 9, 13]
        num_layers = num_layers[dec_number - 1]
        convs = []
        reflecPads = [nn.ReflectionPad2d((1,1,1,1)) for _ in range(num_layers)]
        unPools = [nn.UpsamplingNearest2d(scale_factor=2) for _ in range([0, 1, 2, 3, 4][dec_number - 1])]
        relus = [nn.ReLU(inplace=True) for _ in range(num_layers - 1)]
        sizes = [3, 64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512]
        nums_load = [1, 5, 8, 11, 14, 18, 21, 24, 27, 31, 34, 38, 41]

        for i, num_load in enumerate(nums_load[:num_layers]):
            conv = torch.nn.Conv2d(sizes[num_layers - i], sizes[num_layers - i - 1], 3, 1, 0)
            # load weights and biases
            if num_load >= 11 and dec_number == 3:
                num_load += 1
            elif num_load >= 24 and dec_number == 4:
                num_load += 1
            conv.weight = torch.nn.Parameter(torch.tensor(d.modules[num_load].weight).float())
            conv.bias = torch.nn.Parameter(torch.tensor(d.modules[num_load].bias).float())
            convs += [conv]

        self.dec_number = dec_number
        self.num_layers = num_layers
        self.convs = nn.ModuleList(convs)
        self.reflecPads = nn.ModuleList(reflecPads)
        self.relus = nn.ModuleList(relus)
        self.unPools = nn.ModuleList(unPools)

    def forward(self,x):
        out = x
        pool_index = 0
        for i in range(self.num_layers):
            out = self.reflecPads[i](out)
            out = self.convs[i](out)
            if i != self.num_layers - 1:
                out = self.relus[i](out)
            if self.num_layers - i - 1 in [2, 4, 8, 12]:
                out = self.unPools[pool_index](out)
                pool_index += 1
        return out