## GAN



1. 什么是生成?

   生成就是模型通过学习一些数据，然后生成类似的数据



2. GAN原理

   GAN是如何生成图片？GAN有两个网络，一个是generator（生成图片的网络），还有一个是discriminator（判别网络）。

   在我们训练过程当中，生成网络G的目标就是尽量生成真实的图片去"欺骗"网络D。

   网络D的目标就是区分生成的图片与真实的图片。

   这样就构成的一个"博弈过程"。

   在最理想的情况下，G可以生成足以“以假乱真”的图片G(z)。对于D来说，它难以判定G生成的图片究竟是不是真实的，因此D(G(z)) = 0.5。

   ![开发者自述：我是这样学习 GAN 的](https://static.leiphone.com/uploads/new/article/740_740/201707/5966d1ca9dcec.png?imageMogr2/format/jpg/quality/90)

   如图，首先有一个Generator v1，它生成一些很差的图片，然后有一个Discriminator v1

   ，它能准确的把生成的图片，和真实的图片分类。

   换一句话说，这个Discriminator就相当于一个二分类器，对生成的图片输出0，对真实的图片输出1.

   接着，开始训练出Generator v2，它能生成稍好一点的图片，能够让Discriminator  v1认为这些生成的图片是真实的图片。然后会训练出 Discriminator v2，它能准确的识别出真实的图片，和 Generator v2 生成的图片。以此类推，会有三代，四代。。。n 代的 generator 和 discriminator，最后 discriminator 无法分辨生成的图片和真实图片，这个网络就拟合了。

3. GAN数学描述

   >  根据paper  Generative Adversarial Networks 中的公式

   ![img](https://pic2.zhimg.com/50/v2-f98f1d3caabbca9b6baa4235c40150b4_hd.jpg)

   - 整个式子由两项构成。x表示真实图片，z表示输入G网络的噪声，而G(z)表示G网络生成的图片。
   - D(x)表示D网络判断真实图片是否真实的概率（因为x就是真实的，所以对于D来说，这个值越接近1越好）。而D(G(z))是D网络判断G生成的图片的是否真实的概率。
   - 真实图片集的分布Pdata(x)，x 是一个真实图片，可以想象成一个向量，这个向量集合的分布就是 Pdata。
   - G的目的：上面提到过，D(G(z))是D网络判断G生成的图片是否真实的概率，G应该希望自己生成的图片“越接近真实越好”。也就是说，G希望D(G(z))尽可能得大，这时V(D, G)会变小。因此我们看到式子的最前面的记号是min_G。
   - D的目的：D的能力越强，D(x)应该越大，D(G(x))应该越小。这时V(D,G)会变大。因此式子对于D来说是求最大(max_D)



4. GAN训练

   起初有一个 G0 和 D0，先训练 D0 找到

   ![开发者自述：我是这样学习 GAN 的](https://static.leiphone.com/uploads/new/article/740_740/201707/5966d6b71a9c4.png?imageMogr2/format/jpg/quality/90)

   然后固定 D0 开始训练 G0， 训练的过程都可以使用 gradient descent，以此类推，训练 D1,G1,D2,G2,...

   但是这里有个问题就是，你可能在 D0* 的位置取到了：

   ![开发者自述：我是这样学习 GAN 的](https://static.leiphone.com/uploads/new/article/740_740/201707/5966d72e64539.png?imageMogr2/format/jpg/quality/90)

   然后更新 G0 为 G1，可能：

   ![开发者自述：我是这样学习 GAN 的](https://static.leiphone.com/uploads/new/article/740_740/201707/5966d7617e528.png?imageMogr2/format/jpg/quality/90)

   但是并不保证会出现一个新的点 D1* 使得

   ![开发者自述：我是这样学习 GAN 的](https://static.leiphone.com/uploads/new/article/740_740/201707/5966d788afdf1.png?imageMogr2/format/jpg/quality/90)

   这样更新 G 就没达到它原来应该要的效果，如图：

   ![开发者自述：我是这样学习 GAN 的](https://static.leiphone.com/uploads/new/article/740_740/201707/5966d79e1ea1e.png?imageMogr2/format/jpg/quality/90)

   为避免上述情况，我们并不需要更新G太多。

   我们还需要设定两个 loss function，一个是 D 的 loss，一个是 G 的 loss。

   下图是完整的GAN训练流程：

   ![开发者自述：我是这样学习 GAN 的](https://static.leiphone.com/uploads/new/article/740_740/201707/5966d7bdc95a8.png?imageMogr2/format/jpg/quality/90)





5. 代码实现 （MNIST数据集）

   1. 简单的网络结构：生成网络和对抗网络

      判别网络结构，就是一个分类器

      - 全连接(784 -> 256)
      - leakyrelu, α 是 0.2
      - 全连接(256 -> 256)
      - leakyrelu, α 是 0.2
      - 全连接(256 -> 1)

      代码实现如下

      ``` python
      def discriminator():
          net = nn.Sequential(        
                  nn.Linear(784, 256),
                  nn.LeakyReLU(0.2),
                  nn.Linear(256, 256),
                  nn.LeakyReLU(0.2),
                  nn.Linear(256, 1)
              )
          return net
      ```

      

      生成网络

      网络结构：随机噪声生成一个和数据维度一样的张量

      - 全连接(噪音维度 -> 1024)
      - relu
      - 全连接(1024 -> 1024)
      - relu
      - 全连接(1024 -> 784)
      - tanh 将数据裁剪到 -1 ~ 1 之间

``` python
def generator(nosie_dim=NOISE_DIM):
    net = nn.Sequential(
        nn.Linear(nosie_dim, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 1024),
        nn.ReLU(True),
        nn.Linear(1024, 784),
        nn.Tanh()
    )
    return net
```



接下来定义生成对抗网络的loss。

对于对抗网络，相当于二分类问题。

参考公式：

> ℓD=Ex∼pdata[logD(x)]+Ez∼p(z)[log(1−D(G(z)))]

D(x) 看成真实数据的分类得分，那么 D(G(z)) 就是假数据的分类得分,以上面判别器的 loss 就是将真实数据的得分判断为 1，假的数据的得分判断为 0

对于生成网络，需要去骗过对抗网络，也就是将假的也判断为真的。

参考公式

> ℓG=Ez∼p(z)[logD(G(z))]

生成器的 loss 就是将假的数据判断为 1

代码实现

``` python
bce_loss = nn.BCEWithLogitsLoss()

def discriminator_loss(logits_real, logits_fake): # 判别器的 loss
    size = logits_real.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float().cuda()
    false_labels = Variable(torch.zeros(size, 1)).float().cuda()
    loss = bce_loss(logits_real, true_labels) + bce_loss(logits_fake, false_labels)
    return loss
```

``` python
def generator_loss(logits_fake): # 生成器的 loss  
    size = logits_fake.shape[0]
    true_labels = Variable(torch.ones(size, 1)).float().cuda()
    loss = bce_loss(logits_fake, true_labels)
    return loss
```

使用adam来进行训练

```python
# 使用 adam 来进行训练，学习率是 3e-4, beta1 是 0.5, beta2 是 0.999
def get_optimizer(net):
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, betas=(0.5, 0.999))
    return optimizer
```





训练生成对抗网络

```python
def train_a_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, show_every=250, 
              noise_size=96, num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            real_data = Variable(x).view(bs, -1).cuda() # 真实数据
            logits_real = D_net(real_data) # 判别网络得分
            
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5 # -1 ~ 1 的均匀分布
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据
            logits_fake = D_net(fake_images) # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake) # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step() # 优化判别网络
            
            # 生成网络
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake) # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step() # 优化生成网络

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data, g_error.data))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
```

使用GPU训练

```python
D = discriminator().cuda()
G = generator().cuda()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)
train_a_gan(D, G, D_optim, G_optim, discriminator_loss, generator_loss)
```

训练得到的结果如图：

![image-20210412185658084](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210412185658084.png)

可以看到训练得到的结果并不是很理想，图片比较模糊。



### Least Squares GAN

[Least Squares GAN](https://arxiv.org/abs/1611.04076) 比最原始的 GANs 的 loss 更加稳定，通过名字我们也能够看出这种 GAN 是通过最小平方误差来进行估计，而不是通过二分类的损失函数，下面我们看看 loss 的计算公式：

> ![image-20210413094534756](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413094534756.png)

可以看到 Least Squares GAN 通过最小二乘代替了二分类的 loss，下面定义loss函数

```python
def ls_disrimiator_loss(scores_real, scores_fake):
    loss = 0.5 * ((scores_real - 1) ** 2).mean() + 0.5 * (scores_fake ** 2).mean()
    return loss

def ls_generator_loss(scores_fake):
    loss = 0.5 * ((scores_fake - 1) ** 2).mean()
    return loss

```

开始训练

``` python
D = discriminator().cuda()
G = generator().cuda()

D_optim = get_optimizer(D)
G_optim = get_optimizer(G)

train_a_gan(D, G, D_optim, G_optim, ls_discriminator_loss, ls_generator_loss)
```

训练得到的结果：

![image-20210413095021776](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413095021776.png)

效果也不是很理想



### 深度卷积生成对抗网络

顾名思义，深度卷积生成对抗网络，就是将生成网络和对抗网络都改成了卷积网络的形式。

卷积判别网络结构

- 32 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
- Max Pool 2x2, Stride 2
- 64 Filters, 5x5, Stride 1, Leaky ReLU(alpha=0.01)
- Max Pool 2x2, Stride 2
- Fully Connected size 4 x 4 x 64, Leaky ReLU(alpha=0.01)
- Fully Connected size 1

代码实现：

```python
class build_dc_classifier(nn.Module):
    def _init_(self):
        super(bulid_dc_classifier, self)._init_()
        
        #keras.layers.Conv2D(filters, kernel_size, strides,padding)参数讲解
        #filters: 整数，输出空间的维度 （即卷积中滤波器的数量）。
        #kernel_size: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值
        #strides: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。 可以是一个整数，为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 dilation_rate 值 != 1 两者不兼容。
        #padding: "valid" 或 "same" (大小写敏感)。   valid padding就是不padding，而same padding就是指padding完尺寸与原来相同
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, 1),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2)
        )
        
        self.fc = nn.Sequential(
       	   nn.Linear(1024, 1024),
           nn.LeakyReLU(0.01),
           nn.Linear(1024, 1)
        )
          
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
```



卷积生成网络

卷积生成网络需要将一个低维的噪声向量变成一个图片数据，结构如下

- Fully connected of size 1024, ReLU
- BatchNorm
- Fully connected of size 7 x 7 x 128, ReLU
- BatchNorm
- Reshape into Image Tensor
- 64 conv2d^T filters of 4x4, stride 2, padding 1, ReLU
- BatchNorm
- 1 conv2d^T filter of 4x4, stride 2, padding 1, TanH



代码实现：

```python
class build_dc_generator(nn.Module): 
    def __init__(self, noise_dim=NOISE_DIM):
        super(build_dc_generator, self)._init_()
        
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.Relu(True),
            nn.BatchNormld(1024),
            nn.Linear(1024, 7 * 7 * 128),
            nn.Relu(True),
            nn.BatchNormld(7 * 7 * 128)
        )
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1),
            nn.Relu(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1),
            NN.Tanh()
        )
        
        def forward(self, x):
        	x = self.fc(x)
       		x = x.view(x.shape[0], 128, 7, 7) # reshape 通道是 128，大小是 7x7
        	x = self.conv(x)
        return x
```



训练卷积生成对抗网络：

```python
def train_dc_gan(D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss, show_every=250, 
                noise_size=96, num_epochs=10):
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in train_data:
            bs = x.shape[0]
            # 判别网络
            real_data = Variable(x).cuda() # 真实数据
            logits_real = D_net(real_data) # 判别网络得分
            
            sample_noise = (torch.rand(bs, noise_size) - 0.5) / 0.5 # -1 ~ 1 的均匀分布
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据
            logits_fake = D_net(fake_images) # 判别网络得分

            d_total_error = discriminator_loss(logits_real, logits_fake) # 判别器的 loss
            D_optimizer.zero_grad()
            d_total_error.backward()
            D_optimizer.step() # 优化判别网络
            
            # 生成网络
            g_fake_seed = Variable(sample_noise).cuda()
            fake_images = G_net(g_fake_seed) # 生成的假的数据

            gen_logits_fake = D_net(fake_images)
            g_error = generator_loss(gen_logits_fake) # 生成网络的 loss
            G_optimizer.zero_grad()
            g_error.backward()
            G_optimizer.step() # 优化生成网络

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.data, g_error.data))
                imgs_numpy = deprocess_img(fake_images.data.cpu().numpy())
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1
    
```



得到结果：

``` python
D_DC = build_dc_classifier().cuda()
G_DC = build_dc_generator().cuda()

D_DC_optim = get_optimizer(D_DC)
G_DC_optim = get_optimizer(G_DC)

train_dc_gan(D_DC, G_DC, D_DC_optim, G_DC_optim, discriminator_loss, generator_loss, num_epochs=5)
```

![image-20210413101228168](C:\Users\L\AppData\Roaming\Typora\typora-user-images\image-20210413101228168.png)

如图，通过深度卷积生成对抗网络可以得到更好的效果