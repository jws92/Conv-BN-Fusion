# Fusion between convolution and batch normalization layer
Implementation of fusion between convolution and batch normalization layer for speed up inference

## Abstract
* In many CNNs, the layer order such as convolution-BN has been used for stable convergence when training.   
* BN requires a batch size of at least 2 during training, so the mean and variance values are stored in advance to be used during inference when the batch size is 1.   
  * That is, the stored mean and variance in advance can be used with the weight of BN.
* Here, the weights and bias in the BN layer can be fused with those of the convolution layer.
  * Since the two layers are fused into a single layer, inference can be performed faster.


## Fusing Conv-BN layers
* The convolution layer is computed as follows:   
$$\hat x = x \cdot W_{conv} + B_{conv} \ …\ (1)$$   
* $x$: input  
  $W_{conv}$: Conv weight  
  $B_{conv}$: Conv bias  
  $\hat x$: output  

* Then, BN is computed as follows:   
$$\hat x = \gamma \dfrac{x - \mu} {\sqrt{\sigma ^ 2 + \epsilon}} + \beta \ …\ (2)$$
* $x$: input  
  $\mu$: mean  
  $\sigma ^2$: variance   
  $\gamma$: BN trainable weights   
  $\beta$: BN trainable bias   
  $\epsilon$: constant (to prevent the denominator from becoming zero)   
  $\hat x$: output   

* When the layers are ordered as Conv followed by BN, they can be fused into a single layer using the following equation:   

$$
\hat x = \gamma \dfrac{(x \cdot W_{conv} + B_{conv}) - \mu} {\sqrt{\sigma ^2 + \epsilon}} + \beta \ …\ (3)
$$

$$
\hat x = \dfrac{W_{conv} \cdot \gamma} {\sqrt{\sigma ^2 + \epsilon}} x + \dfrac{B_{conv} - \mu} {\sqrt{\sigma ^2 + \epsilon}} \gamma + \beta \ …\ (4)
$$
  
* In Eq. (4), it is equivalent to Eq. (1).
  * After multiplying the $W_{conv}$ by the $\gamma$ and dividing by the square root of the sum of the $\sigma ^2$ and $\epsilon$, it can be used as the weight of the convolution layer.
  * Also, $B_{conv}$, $\mu$, and $\beta$ can be fused as a single bias term.
* By updating the weights and bias of the convolution layer respectively, the BN effect can be achieved even with just the convolution layer.

* **Implementation of PyTorch**   
```python
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
```

* For fusing the Conv-BN layers, the model should be initialized first.   
  * A simple example of the fusion of Conv-BN layers
    ```python
    import torch
    import torch.nn as nn

    from functools import partial

    torch.manual_seed(0)


    class ConvBNAct(nn.Module):
        def __init__(self, 
                     in_channels, out_channels, kernel_size, 
                     stride=1, padding=1, dilation=1, groups=1,
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


    class SimpleCNN(nn.Module):
        def __init__(self, 
                     in_channels=3, out_channels=64, num_classes=10,
                     num_blocks=5):
            super().__init__()

            self.stem = ConvBNAct(in_channels, out_channels, 3, 1, 1,
                                  act_layer=partial(nn.ReLU, inplace=True))
            
            self.modules = nn.ModuleList([
                ConvBNAct(out_channels, out_channels, 3, 1, 1,
                          act_layer=partial(nn.ReLU, inplace=True))
            ])

            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(out_channels, num_classes)

            self.apply(self._init_weights)
        
        def _init_weights(self, m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                torch.nn.init.normal_(layer.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        def forward(self, x):
            x = self.stem(x)
            for module in self.modules:
                x = module(x)

            x = self.pool(x)
            x = self.classifier(x)

            return x

        def fuse(self):
            for m in self.modules():
            if isinstance(m, ConvBNAct):
                # for SyncBN when training with DDP
                if isinstance(m.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.conv = fuse_conv_and_bn(m.conv, m.norm)
                    delattr(m, 'norm')
                    m.forward = m.fuseforward

    @torch.no_grad()
    def main():
        # model initialization
        model = SimpleCNN().cuda()
        model.eval()

        # fuse Conv-BN layer
        model.fuse()

        # ...
    ```


## Fusing BN-Conv layers
* On the contrary, we introduce the method for fusing the weights and biases of the BN and Conv layers when the order is BN followed by Conv.   
* Note that the convolution layer with padding cannot work ([see in detail](https://leimao.github.io/blog/Neural-Network-Batch-Normalization-Fusion/)).   
   * Here, we only deal with the fusion BN-Conv1x1.   

* The equation below represents the fusion of the BN-Conv1x1 layers:   

$$
\hat x = \gamma \dfrac{x - \mu} {\sqrt{\sigma ^ 2 + \epsilon}} + \beta
$$

$$
\hat x = \hat x \cdot W_{conv} + B_{conv}
$$

$$
\hat x = (\gamma \dfrac{x - \mu} {\sqrt{\sigma ^ 2 + \epsilon}} + \beta) \cdot W_{conv} + B_{conv}
$$

$$
\hat x = \dfrac{\gamma \cdot W_{conv}} {\sqrt{\sigma ^ 2 + \epsilon}} x + (\beta - \dfrac{\gamma \mu} {\sqrt{\sigma ^ 2 + \epsilon}}) \cdot W_{conv} + B_{conv}
$$

* **Implementation of PyTorch**
```python
def fuse_bn_conv1x1(bn, conv):
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
    
    w_bn = bn.weight.clone()
    b_bn = bn.bias.clone()
    mean = bn.running_mean.clone()
    var = bn.running_var.clone()
    eps = bn.eps
    
    w_conv = conv.weight.clone()
    b_conv = conv.bias
    
    # fused weight between BN and Conv layer
    new_w = (w_bn * torch.rsqrt(var + eps) * w_conv.flatten(1)).view(fusedconv.weight.size())
    fusedconv.weight.copy_(new_w)
    
    # fused bias between BN and Conv layer
    new_b = b_bn - (w_bn * mean * torch.rsqrt(var + eps))
    new_b = nn.functional.conv2d(new_b.unsqueeze(-1).unsqueeze(-1), w_conv, b_conv) # perform convolution here
    fusedconv.bias.copy_(new_b.squeeze())
 
    return fusedconv
```

* A simple example of the fusion of BN-Conv1x1 layers
```python
import torch
import torch.nn as nn

class BNConvAct(nn.Module):
    def __init__(self, 
                 in_channels, out_channels, kernel_size, 
                 stride=1, padding=1, dilation=1, groups=1, bias=True,
                 act_layer=None):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)
        self.act = act_layer() if act_layer is not None else nn.Identity()

    # training or inference w/o fusing
    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.act(x)

        return x

    # must be used for inference
    def fuseforward(self, x):
        x = self.conv(x)
        x = self.act(x)

        return x


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_classes=10,
                 num_blocks=5):
        super().__init__()

    # define some blocks
    # ...

    def fuse(self):
        print("Fusing BN and Conv layers...")
        for m in self.modules():
            if isinstance(m, BNConvAct):
                if isinstance(m.norm, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.conv = fuse_bn_conv1x1(m.norm, m.conv)
                    delattr(m, 'norm')
                    m.forward = m.fuseforward

    # define some methods
    # ...

@torch.no_grad()
def main():
    # model initialization
    model = SimpleCNN().cuda()
    model.eval()

    # fuse BN-Conv1x1 layer
    model.fuse()
```

## Experiments
* ResNet-50   
  * RTX 3060 is used for the experiment with FP32.   

|Methods|Latency (bs=1)|Improved latency|Throughputs (bs=64)|Improved throughputs|
|-------|--------------|----------------|-------------------|--------------------|
|Normal|5.61 ms|-|606.75|-|
|Fusing ConvBN|3.82 ms|**+31.91%**|610.40|**+0.6%**|


## References
* Fusing Conv-BN layers (w/ PyTorch codes) - [LINK](https://nenadmarkus.com/p/fusing-batchnorm-and-conv/)
* Fusing BN-Conv layers (w/o PyTorch codes) - [LINK](https://leimao.github.io/blog/Neural-Network-Batch-Normalization-Fusion/)