# VAE

Variational Auto-Encoder (VAE), 变分自编码器. 这个的原论文叫Auto-Encoding Variational Bayes, 我也不知道为什么Variational提前了, 可能是因为比较符合英文语序? 

## 概念介绍

这个终归不是NLP, 现在的NLP任务基本离不开transformer. 然而, VAE是和概率图模型相关. 因此, 这里需要科普一些基础概念.

### Auto-Encoder

虽然所谓Auto-Encoder, 自动编码器听起来很新颖吓人, 实际上就是能把输入$x$降维并尽可能保证信息不丢失的映射. 在transformer中我们早就见过了encoder, 实际上就是auto-encoder.

而与之相对应的decoder, 则是希望从encoder的输出中尝试还原原始信息. 我们熟知的PCA就是一种线性的数据降维方式, 但当今使用的基于NN的encoder, 其能力要比PCA强上许多.

### 概率图模型

先略过.

## VAE的理论

首先, VAE是个生成模型, 假设其要生成的内容为$X$. 现实中我们要基于一些样本来获取, 显然其是服从某些分布的, 如下

$$
\{x_1,..,x_n\}\sim P(X)
$$

而生成模型中, 本质上我们就是希望知道$P(X)$的具体结果, 这样就能基于此生成一些满足$X$分布的数据. 然而现实中这个目标是非常难实现的, 因为一些具体的生成内容, 如图像之类的, 你很难说它是一个服从什么分布的东西. 对此, 无论是GANs还是VAE, 其目的都是希望基于给定分布的$Z$, 并训练一个NN, 其会输出满足$X$分布的结果, 如下

$$
z\sim P(Z)\\
\hat{x}=G(z)\sim P(X)
$$

为了实现这个目标, GANs采用的是对抗网络, 当检测用的分类器已经看不出生成的内容和真实内容的区别时, 则可以认为生成网络生成的数据和真实数据分布相同了. 而VAE这边采用的是更加数学化的处理.

其中, $P(X)$可以写作

$$
P(X)=\int_{z\in Z} P(X|Z=z)P(Z=z)dz
$$

在许多教程中这个公式被写作求和的样式, 但不管是哪种样式, 其想表达的意思都差不多. 但这个公式有什么用呢? 实际上就目前而言就是没什么用, 因为$P(X|Z)$看上去是一个比$P(X)$更难求的式子. 但假设我们已经求得了$P(X|Z)$, 那么我们可以随机生成一个具体的$z$, 然后根据这个概率公式生成一个$x$.

VAE中的创新点在此, 上面的式子有类似的写法如下

$$
P(Z)=\int_{x\in X} P(Z|X=x)P(X=x)dz
$$

真就只是换皮而已... 但实际上这个公式乍一看比前面那个还难用, 因为$X$到了条件项! 但模型的核心公式其实还真就是这个. 此前我们是假设$Z$服从**标准正态分布**, 这里进一步假设, 它假设针对每一个不同的样本, 其$Z$都符合一个不同的正态分布. 如下

$$
P(Z|X=x_i)\sim\mathcal{N}(\mu_i,\sigma^2_i)
$$

这里就是可以训练的点了! 假设有一个NN, 其能根据$x_i$给出$\mu_i$和$\sigma_i$的值, 如下

$$
(\mu_i,\log\sigma_i^2)=f(x_i)
$$

这里有个小细节, 这里的$\sigma$要取对数, 原因在于后面的loss. 模型的结构图也就可以开始看了

![](https://pic2.zhimg.com/80/v2-beab2fbf9b9913f243b2eb5db3048b49_720w.webp)

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-07_at_4.47.56_PM_Y06uCVO.png)

显然, 根据此前的分析加上模型插图, 我们这里就很能理解其到底在干一件什么事情. 然而, 还有很多数学问题尚待解决. 这里就不做QA了, 直接继续. 简单而言, 前面说的NN倘若训练好了, 实际上就是一个$P(Z|X=x_i)$的概率生成器. 论文这做了进一步简化, 它希望无论输入$x_i$是什么, $P(Z|X=x_i)$都是个标准正态分布, 这样条件分布求和后的$P(Z)$也是个标准正态分布. 这显然是一个相对摆烂的做法, 但这种方法很简单, 就是loss上做文章

$$
\begin{aligned}
\mathcal{L}_\mu&=|\mu|^2\\
\mathcal{L}_\sigma&=|\log\sigma^2|^2
\end{aligned}
$$

这样如果上述loss越小, 那么分布越接近于$\mathcal{N}(0,1)$, 这里也解释了为什么取对数的原因, 这样都趋于0就好了. 其中, 原文仍然不放心这种做法, 还引入了KL散度

$$
\begin{aligned}
&L_{KL}\\
=&KL(\mathcal{N}(\mu,\sigma^2),\mathcal{N}(0,1))\\
=&\frac{1}{2}(-\log\sigma^2+\mu^2+\sigma^2-1)
\end{aligned}
$$

用KL散度来度量分布的差异显然是更数学的. 但为什么有了前面的loss这里还要额外引入loss? 原因是前面提出的两个loss在比例上难以把握, 因为这两个变量在对概率分布的贡献上并不是一样的.

最后是重参数技巧, 这一点非常重要, 因为我们针对采样结果本身是不可导的, 这会导致我们无法使用常规的NN算法. 这一点要重参数化, 其思路是很简单的, 就是从

$$
\mathcal{N}(\mu,\sigma)
$$

中采样一个值, 等价于从标准正态分布中采样

$$
\mathcal{N}(0, 1)
$$

一个值$\epsilon$, 而后经过变换

$$
Z=\sigma\epsilon+\mu
$$

得到. 这样, 一整个模型就串联了起来.

## 代码

github上有各种VAE的实现, 实际的实现比说起来难一点, [基础](https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py)的如下

```python
import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
```

代码的过程和解释的一样, 这个是基于图像的一个VAE, 其定义的`encoder`实际上就是个CNN, 而后会有一个`fc_mu`和`fc_var`就是要根据`encoder`的输出, 拍平, 然后进行一个线性映射, 得到$\mu$和$\log\sigma$. 而后, 在`reparameterize`中, 通过标准正态分布采样得到的值与$\mu$和$\log\sigma$结合, 就可以得到一个带有其梯度的采样. 最后, 通过一个`decoder`方法 (不是`decoder`变量) 就可以获得结果. 这里代码的`forward`有点紧凑, 其第一个返回值就是`decoder(z)`后的结果, 换言之, 也就是生成的结果. 最后就是`loss_function`部分, 其接受重建图像`recons`以及其他参数, 用前面提到的loss来计算结果.

## reference

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

- [知乎-VAE](https://zhuanlan.zhihu.com/p/34998569)

- [知乎-VAE](https://zhuanlan.zhihu.com/p/567818131)

- [PyTorch-VAEs](https://github.com/AntixK/PyTorch-VAE/tree/master)