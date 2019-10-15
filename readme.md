

## 思路
>基础攻击方法： 带动量的投影梯度下降(m-pgd)  

### 1）提高攻击成功率


#### trick 1  多模型集成提高转移能力
* 融合了3个模型，分别是IR50, IR152, IRSE50

#### trick 2  对生成的扰动进行高斯平滑
### 2）保证成功率的同时减少扰动


#### trick 1  用mask控制对抗样本扰动的添加
* 将 lfw 数据集经过替代模型 inference 得到到 512 emdedings特征与对应 labels 在本地持久化存储
* 初始化mask=1，默认未达到本地攻击成功标准，攻击的每次迭代过程中计算新样本的 embedding 与本地 embeddings 池的余弦距离，找出当前最大的余弦值，与此样本真实 label 对应的最大余弦值比较，设定全局变量margin（比如margin=0.2, 其中 0.0 <= margin <= 2.0），当且仅当当前最大余弦值与真实label对应的最大余弦值的差值大于等于 margin 的时候，更新此样本对应的 mask 值为0， 从而限制扰动的继续增加
* 使用此方法可以摆脱基于梯度的迭代攻击的弊端，即减少最大扰动会降低成功率， 为了增加成功率又会让已经足够被分类错误的样本增加多余扰动量，只需将最大扰动上限设定到全部攻击成功时的大小，仅关注每次迭代的扰动大小和通过线上分数来调节margin值即可

#### trick 2  给loss指定bias，让梯度朝着余弦值第二大的方向更新
* 原始loss的梯度方向是替代模型变化最快的方向，最快不代表路径是最短，计算干净样本 embeding和本地 embeddings 池的余弦值，找到非真实 label 的最大余弦值对应的embedding，让loss不仅是减小对抗样本与干净样本的余弦值，同时增大与biased embedding的余弦值

## 3.代码及模型
### 1） 代码说明
* embeds_pkl ：存放本地持久化embeddings池的文件夹
* images : 存放干净样本和对抗样本的文件夹
* nets ： 存放替代模型的文件夹
* bias_attack_with mask_multiple.py ： 主要攻击脚本
* utils.py ： 一些攻击需要的函数和类
* get_{}_embeds.py ： 生成持久化embedding池的脚本，712 代表只需攻击所用的712张， 712lfw 代表 lfw 中所有712个人的人脸图像， lfw 代表整个 lfw 数据集

### 2） 模型地址
* IR50_epoch120 : https://pan.baidu.com/s/1L8yOF1oZf6JHfeY9iN59Mg 
* IR152_epoch112 : https://pan.baidu.com/s/19c2_qdGeLo3CEiSSQztUeA , pw: y6ya
* IRSE50 : https://pan.baidu.com/s/12BUjjwy1uUTEF9HCx5qvoQ




