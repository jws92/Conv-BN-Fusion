import torch
import torch.nn as nn

from functools import partial
import time


# Fuse convolution and batchnorm layers https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
def fuse_conv_and_bn(conv, bn):
    device = conv.weight.device
    
    fusedconv = nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
        groups=conv.groups,
        dilation=conv.dilation,
		bias=True).requires_grad_(False).to(device) 
        # Since this will only be used during inference, the gradient must be disabled.
    
    # fused weight between Conv and BN layer
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
    fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
    
    # fused bias between Conv and BN layer
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros( conv.weight.size(0), device=device )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_( torch.matmul(w_bn, b_conv) + b_bn )
 
    return fusedconv


class ConvBNAct(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1,
                 act_layer=None):
        super().__init__()
        
        # When using BN, it is recommended to set the bias of the Conv layer to False, 
        # as the effect of the bias would be eliminated during BN computation.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    # training or inference w/o fusing
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

    # must be used for inference
    def fuseforward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64):
        super().__init__()
        
        width = int(planes * (base_width / 64.0)) * groups
        
        self.conv1_bn1_relu = ConvBNAct(inplanes, width, kernel_size=1,
                                        act_layer=partial(nn.ReLU, inplace=True))
        self.conv2_bn2_relu = ConvBNAct(width, width, kernel_size=3, stride=stride, padding=1,
                                        act_layer=partial(nn.ReLU, inplace=True))
        self.conv3_bn3_relu = ConvBNAct(width, planes * self.expansion, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1_bn1_relu(x)
        out = self.conv2_bn2_relu(out)
        out = self.conv3_bn3_relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        groups=1,
        width_per_group=64):
        super().__init__()

        self.inplanes = 64
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1_bn1_relu = ConvBNAct(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                        act_layer=partial(nn.ReLU, inplace=True))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1):
        
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBNAct(self.inplanes, planes*block.expansion, 
                                   kernel_size=1, stride=stride)

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width)
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1_bn1_relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
    
    def fuse(self):
        num_fused = 0
        
        for m in self.modules():
            if isinstance(m, ConvBNAct):
                # for SyncBN when training with DDP
                if isinstance(m.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.conv = fuse_conv_and_bn(m.conv, m.norm)
                    delattr(m, 'norm')
                    m.forward = m.fuseforward
                    
                    num_fused += 1
        
        print("Fused %d Conv-BN layers." % num_fused)


def throughput(model, x, num_iters=30):
    batch_size = x.size(0)
    
    for i in range(50):
        model(x)
    torch.cuda.synchronize()

    total_time = 0
    for i in range(num_iters):
        starter, ender = torch.cuda.Event(enable_timing=True), \
            torch.cuda.Event(enable_timing=True)
        # measure time
        starter.record()
        model(x)
        ender.record()
        
        torch.cuda.synchronize()
        
        curr_time = starter.elapsed_time(ender) / 1000
        total_time += curr_time
        
    avg_throughput = (num_iters * batch_size) / total_time
    print(f'batch_size {batch_size} throughput: {avg_throughput}')

    return avg_throughput


def latency(model, x, num_iters=30):
    batch_size = x.size(0)
    
    for i in range(50):
        model(x)
    torch.cuda.synchronize()

    total_time = 0
    for i in range(num_iters):
        starter, ender = torch.cuda.Event(enable_timing=True), \
            torch.cuda.Event(enable_timing=True)
        # measure time
        starter.record()
        model(x)
        ender.record()
        
        torch.cuda.synchronize()
        
        curr_time = starter.elapsed_time(ender) / 1000
        total_time += curr_time
        
    avg_time = total_time / num_iters
    print(f'batch_size {batch_size} latency: {avg_time*1000} ms')

    return avg_time


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=1000)
    
    model.eval()
    model.fuse()
    model.to(device)
    
    measure = 'latency'
    # measure = 'throughput'
    
    num_iters = 30
    
    if measure == 'latency':
        x = torch.randn((1, 3, 224, 224), device=device)
        latency(model, x, num_iters)
    else:
        x = torch.randn((64, 3, 224, 224), device=device)
        throughput(model, x, num_iters)
    
    
if __name__ == "__main__":
    main()