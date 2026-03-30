# 从零开始：用新闻分类学会机器学习

> 一个面向完全零基础读者的交互式教程。以"给新闻文章自动打标签"为主线，从 Python 基础一路学到深度学习。

---

## 这个教程能学到什么？

完成全部章节后，你将能够：

- 用 **pandas** 加载和分析真实数据集
- 对文本数据做清洗和预处理
- 理解并实现 **TF-IDF** 文本表示
- 训练 4 种机器学习模型：**朴素贝叶斯、kNN、CNN、双向 LSTM**
- 读懂混淆矩阵、Accuracy、F1-Score 等评估指标
- 保存模型并写一个可直接调用的预测函数

---

## 使用方法

每章点击 **"Open in Colab"** 按钮，在浏览器里直接运行代码，无需安装任何软件。

**需要准备的文件：** `dataset.csv`（128,600 篇新闻文章，三列：Title / Description / Class）

---

## 章节目录

| 章节 | 主题 | 核心概念 | 打开 |
|------|------|----------|------|
| Chapter 0 | 工具准备 | Python、Colab、库（library） | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter00_setup.ipynb) |
| Chapter 1 | pandas 是什么，怎么用 | DataFrame、read_csv、value_counts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter01_pandas.ipynb) |
| Chapter 2 | 用图说话：数据可视化 | matplotlib、柱状图、直方图、类别平衡 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter02_visualization.ipynb) |
| Chapter 3 | 文本预处理 | 正则表达式、clean_text、train_test_split、分层划分 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter03_preprocessing.ipynb) |
| Chapter 4 | 模型是什么？训练是什么？ | 模型、训练、预测、过拟合、sklearn、TensorFlow | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter04_what_is_model.ipynb) |
| Chapter 5 | 把文字变成数字：TF-IDF | 词袋模型、TF、IDF、稀疏矩阵、ngram | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter05_tfidf.ipynb) |
| Chapter 6 | 第一个分类器：朴素贝叶斯 | 概率、贝叶斯定理、混淆矩阵、F1-Score | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter06_naive_bayes.ipynb) |
| Chapter 7 | 第二个分类器：kNN | 距离、维度灾难、SVD 降维、交叉验证 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter07_knn.ipynb) |
| Chapter 8 | 神经网络基础 | 神经元、反向传播、梯度下降、Keras | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter08_neural_networks.ipynb) |
| Chapter 9 | 词向量 Embedding | Word2Vec、Tokenizer、pad_sequences | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter09_embedding.ipynb) |
| Chapter 10 | 第三个分类器：CNN | 卷积、MaxPooling、n-gram 检测器、Dropout | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter10_cnn.ipynb) |
| Chapter 11 | 第四个分类器：双向 LSTM | RNN、梯度消失、LSTM 门控、双向 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter11_bilstm.ipynb) |
| Chapter 12 | 汇总、保存与部署 | 模型对比、joblib、predict 函数、下一步 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/chapter12_summary.ipynb) |

---

## 推荐学习路径

```
Chapter 0 → 1 → 2 → 3 → 4          ← 基础铺垫（约 3 小时）
     ↓
Chapter 5 → 6 → 7                   ← 传统机器学习（约 3 小时）
     ↓
Chapter 8 → 9 → 10 → 11 → 12       ← 深度学习（约 4 小时）
```

---

## 参考资料

- [pandas 官方文档](https://pandas.pydata.org/docs/)
- [scikit-learn 官网](https://scikit-learn.org/stable/)
- [TensorFlow / Keras 教程](https://www.tensorflow.org/tutorials)
- [3Blue1Brown 神经网络可视化](https://www.3blue1brown.com/topics/neural-networks)
- [Colah's LSTM 博文](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [Kim 2014 CNN 文本分类论文](https://arxiv.org/abs/1408.5882)

---

## 上传到 GitHub 后的操作

1. 把仓库设为 **Public**
2. 将所有链接里的 `chaosfrey-arch` 替换为你的 GitHub 用户名
3. 将仓库名 `news-classification-tutorial` 替换为实际仓库名（如果不同）
4. 每个章节的 Colab 按钮就会自动生效

---

*本教程基于 COMP42415 Text Mining and Language Analytics 课程作业，数据集来源为 AG News 衍生数据集。*
