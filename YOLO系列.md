# YOLO系列

## yolo v1

- 2016 CVPR
- 45 FPS 448*448
- 63.4mAP

### 论文思想

1. 将图像分成S×S个网格，如果某个obj的中心落在这个网格中，则这个网格负责预测obj

2. 每个网格预测B个bbox和C个类别的分数，每个bbx包括4个预测位置信息和1个confidence

   以PASCAL VOC为例，B=2，C=20，则20+2×(4+1)=30

   - 每个bbox预测$x,y,w,h$，$(x,y)$表示bbox中心相对于cell的坐标，$w,h$表示相对于整幅图像的w,h
   - confidence表示预测box和gt box的IOU



## yolo v2(YOLO9000)

- 2017 CVPR
- 67 FPS 416*416
- 76.8mAP

### 论文思想

1. batch normalization
   - 添加bn层能帮助网络收敛，可以移除dropout操作
   - 提高2%mAP
2. higher resolution classifier
   - 采用448×448分辨率的分类器，10 epochs on ImageNet
   - 提供4%mAP
3. convolution with anchor boxes
   - yolo v1是直接预测bbox的坐标
   - yolo v2预测相对于anchor的offset
   - mAP下降0.3%，recall提高7%(81%,88%)
4. dimension clusters（anchor聚类）
   - yolo中anchor的两个问题
     - box dimension都是人工选择的，但如果能选择更好的priors anchors，则网络可以更容易学习
     - 利用k-means聚类在训练集的bboxe中自动选择那些good priors，这些priors可以得到更好的IOU score
5. direct location prediction
   - 5%mAP
6. fine-grained feature
   - 特征图融合passthrough layer
   - w,h,c -> w/2,h/2,c*4
   - 1%mAP
7. multi-scale training
   - 每10个batches，网络随机选择一个new image size
   - 因为网络的缩放因子是32（416/13），因此input size最小320×320，最大608×608

### BACKBONE：Darknet-19

![image-20220116115812409](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116115812409.png)



## yolo v3

- 2018 CVPR

### BackBone：Darknet-53

- 没有maxpooling层
- 通过卷积层实现下采样

### model structure

在3个特征层上进行预测，每个特征层有3种尺度的bbox，对每个bbox，预测4个offset量，1个obj prediction（confidence score），80个class prediction，即1个anchor对应85个参数

![img](https://img-blog.csdnimg.cn/2019040211084050.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM3NTQxMDk3,size_16,color_FFFFFF,t_70)



### 目标边界框的预测

![image-20220116142620482](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116142620482.png)



### 正负样本匹配

yolo v3对每一个bbox利用logistic回归预测一个objectness score

- 对一张图片的gt object分配一个正样本（bbx prior），原则：选择与gt重合程度最大的那个bbox
- 对那些重合程度超过给定阈值，但又不是最大重合的样本，直接丢弃
- 如果一个bbox没有分配给某一个gt object，那么这个bbox没有坐标loss和class loss，只有objectness loss，则视作负样本



## yolo v3 spp ultralytics

​       model                                size      coco-AP   coco-mAP0.5

- yolo-v3                               512         32.7            57.7
- yolo-v3-spp                       512         35.6            59.5
- yolo-v3-spp-ultralytics     512         42.6            62.4



### mosaic图像增强

将多张（一般是4张）图片拼接在一起训练

- 增加数据的多样性
- 增加目标个数
- BN能一次性统计多张图片的参数（变相增加了训练的batch size）



### SPP模块

实现了不同尺度的特征融合

![image-20220116154004963](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116154004963.png)

- 在maxpooling前需要进行相应的padding，以保证size不变
- w,h保持不变，C变为4*C

通常网络中只加入一个SPP模块，更多的SPP模块，网络的性能不会有显著的提升

![image-20220116154350344](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116154350344.png)



### Loss改进

- yolo v3损失为**平方损失**![image-20220116154511224](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116154511224.png)

  

#### 基于**IOU**的损失

#### IOU LOSS

- 更常用的IOU loss公式为：$IoU Loss=1-IoU$

<img src="C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116154655215.png" alt="image-20220116154655215" style="zoom: 80%;" /><img src="C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116154804516.png" alt="image-20220116154804516" style="zoom: 80%;" />

优点

- 更好地反映重合程度
- 具有尺度不变性

缺点：

- 当两个边界框不相交时，Loss=0

  

#### GIOU LOSS

- $GIoU=IoU-\frac{A^c-u}{A^c}$，其中$A^c$表示包含两个box的最小矩形的面积，$u$表示两个box并集的面积
- $GIoU Loss = 1-GIoU，GIoU Loss\in[0,2]$

<img src="C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116155213668.png" alt="image-20220116155213668" style="zoom: 50%;" />

- 当两个box水平或垂直时，GIoU会退化成IoU

  

#### IoU和GIoU的问题：

- 收敛非常慢
- 回归结果还不够准确

![image-20220116160242103](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116160242103.png)

![image-20220116160401055](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116160401055.png)



#### DIOU LOSS

![image-20220116161225116](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116161225116.png)

- $DIoU=IoU-\frac{p^2(b,b^{gt})}{c^2}=IoU-\frac{d^2}{c^2}$
  - $d$ 表示两个box中心点的距离distance
  - $c$表示两个box最小外接矩形的对角线长度
- $DIoULoss=1-DIoU，DIoULoss\in[0,2]$
  - DIoU Loss能够直接最小化两个boxes之间的距离，因此收敛更快

#### CIOU LOSS

一个优秀的回归定位损失应该考虑3种几何参数

- 重叠面积：IoU
- 中心点距离：$\frac{d^2}{c^2}$
- 长宽比

CIoU(Complete-IoU)

- $CIoU = IoU-(\frac{p^2(b,b^{gt})}{c^2}+\alpha v)$
  - 其中$v=\frac{4}{\pi^2}(arctan\frac{w^{gt}}{h^{gt}}-arctan\frac wh)^2,\alpha=\frac{v}{(1-IoU)+v}$
- $CIoULoss = 1-CIoU$



#### Focal Loss

focal loss主要针对one-stage检测模型，如ssd和yolo系列中正负样本不匹配的问题

- 通常一张图像能够匹配到目标的候选框（正样本）只有十几个到几十个，而没有匹配到的候选框（负样本）大概有1e4—1e5个，即**正负样本不匹配class imbalance**

对于一般的交叉熵损失，定义为

- $CE(p,y)=\begin{equation} \begin{cases} -log(p)&y=1\\ -log(1-p)& y=0 \end{cases} \end{equation}$

定义$p_t=\begin{equation} \begin{cases} p& y=1\\ 1-p& y=0 \end{cases} \end{equation}$，则$CE(p,y)=CE(p_t)=-log(p_t)$

 **1 balance cross entropy**：

- 定义一个factor：$\alpha$

- 当y=1时，为$\alpha$，当y=0时，为$1-\alpha$
- 此时：$CE(p_t)=-\alpha_tlog(p_t)$

<img src="C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116165019338.png" alt="image-20220116165019338" style="zoom:50%;" />

**2 focal loss definition**

- 因子$\alpha$不能区分easy example和hard example
- focal loss希望能够降低easy examples的权重，让model focus on hard negatives
- 因子：$(1-p_t)^{\gamma}$

**focal loss**：$FL(p_t)=-(1-p_t)^{\gamma}log(p_t)$

<img src="C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116165609165.png" alt="image-20220116165609165" style="zoom:67%;" />



更进一步的**focal loss**：$FL(p_t)=-\alpha_t (1-p_t)^{\gamma}log(p_t)$

- $FL(p)=-\alpha_t (1-p_t)^{\gamma}log(p_t)=\begin{equation} \begin{cases} -\alpha (1-p)^{\gamma}log(p)&y=1\\ -(1-\alpha)p^{\gamma}log(1-p)& y=0 \end{cases} \end{equation}$

<img src="C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116170009748.png" alt="image-20220116170009748" style="zoom:50%;" />

![image-20220116170430548](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220116170430548.png)

- 蓝色和红色是易分样本，FL可以大幅度降低易分样本的损失权重
- 对于绿色的hard分类样本，则损失权重的下降非常小
- 使得model可以专注于hard examples的学习

- focal loss易受噪音的干扰，对数据集的标注要求较高



### model architect

![yolov3spp](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/raw/master/pytorch_object_detection/yolov3_spp/yolov3spp.png)







# ViT（Vison Transformer）

![image-20220120205628002](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220120205628002.png)



![image-20220120205754985](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220120205754985.png)



## embedding层

对于标准的transformer模块，要求输入是token（向量）序列，即二维矩阵【num_token，token_dim】

对于图像，以224×224×3为例：

- 224，224，3

- 经过一个卷积核【kernel size=16×16，stride=16，num=768】
- 14，14，768
- flatten处理
- 196，768
- 拼接一个初始class token【1，768】
- 197，768
- 叠加position embedding
- 197，768
- 768就是token_dim，197看作num_token

## position embedding

![image-20220120210834904](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220120210834904.png)

- 如何编码空间信息对于结果不是很关键



## Encoder层

![image-20220120211123089](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220120211123089.png)



- 在transformer encoder前有个dropout层，后有一个LN层

## MLP Head层

- 训练ImageNet21k时是linear+tanh+linear
- 迁移学习时，只有linear



## result

![image-20220120212259990](C:\Users\zhaojialin\AppData\Roaming\Typora\typora-user-images\image-20220120212259990.png)

