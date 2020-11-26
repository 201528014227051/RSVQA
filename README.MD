# Mutual Attention Inception Network for Remote Sensing Image Question Answering

## 服务器账户：

### 代码和数据   

 服务器6user2下路径为：/home/user2/VQA/DATA



#### 成数据集的代码：/home/user2/VQA/DATA/DOTA

用数据集dota举例：

dota2vqa.m:是用检测的数据集生成对应数量和存在与否问题答案的数据集



分类用sydney分类数据集举例：

class_vqa.m:根据分类标签生成

## 方法代码

首先激活环境：`source activate vqa`

或者安装`pip install -r requirements.txt`

训练+测试：

`python main_args_inception_vgg16.py `

这里测试和训练是一起的，也就是训练完毕之后直接测试，如果想要手动测试：

`python test_inception_vgg16.py`













