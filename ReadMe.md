Dataset and pre-trained model: https://huggingface.co/VIT-Learner
Github: https://github.com/Tt200411/Interpretative-model-VIT-based-on-Medical-Image


Welcome all members!
firstly we come, you should do this things:
1. Clone resposity
2. create yourself branch(do not use main branch to do any changes!!!!!)
3. Look my strcture, and use your small group own folder to add yours code

Upload rules:
1. do more commits as you can
2. upload everyday
3. after check and test do pull&request to main branch

Schedule:
refresh latter

All current tasks:
first week(9/10):
1. load all data(Lu response)
2. VIT implementation(Tang response)
3. Schedule refresh(Xiao response)

We decided final author order by work quality and quantify:
So work hard and try to upload your own code everyday

___________________________________________________________________________________________________________________________________
Enviorment:
python 3.8
torch 2.0.3
cv2  2017年后任意版本
sklearn2017年后任意版本
————————————————————————————
self-attention 中包括了CNN分类模型，self-attention机制，这部分后面的代码都是测试时使用不代表真实结果
其余大部分代码以及训练数据 都在Tumour classification
Test部分就是为了测试有没有读取数据和更改数据权限
Data preprocessing & segmention 这部分代码在同名文件夹，主要包括图像裁切和inveted
数据来自Kaggle开源数据集（https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset）
主要是一个多种肿瘤图片的3视图（未经裁剪）
___________________________________________________________________________________________________________________________________

主要分为几个部分先

Abstract

Introduction:
在这里我们简单描述基于KAN模型的 Vision Transformer 对于可解释性的重大进步和在精度上保持的优秀，以及其较大的宽容度，通过迁移学习和具体问题的本地再次训练，可以对于具体问题有更多specific的解法
而避免了基于多头注意力的模型硬伤，即要求极强的正则化和对于大数据大模型的强制要求，可以针对不同的问题提出不同的结论在医学背景有更大的进步

而对于KAN模型其的解释性优势和剪枝的表现，我们避免了使用MLP导致数据维度较高，难以解释的结构性问题，使用更简单的模型能模拟出更好的结果，配合解释性模块甚至可以输出具体的数学表达式子

最后基于迁移学习的模型，在不同疾病的小数据模型下有更好的表现，比较容易在不同疾病的不同地区状况下做出更specific的改动

VIT本身作为图像分类问题的SOTA，决定了整个模型的表现非常好（待改动），对比多头注意力机制加上mlp的组合，新模型不仅在解释性上的表现更加优秀，在小数据集上的最终表现也优于mlp

result and disscussion
重点放在这个部分
要体现我们在小模型，医学，和可解释性的优势
对比resnet160，VGG等头部





