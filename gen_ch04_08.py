# -*- coding: utf-8 -*-
import json, os, sys

FOLDER = os.path.dirname(os.path.abspath(__file__))

def nb(cells):
    return {'cells': cells,
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.10.0'}},
            'nbformat': 4, 'nbformat_minor': 5}

def md(src):  return {'cell_type': 'markdown', 'metadata': {}, 'source': src}
def code(src): return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src}

def badge(f): return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chaosfrey-arch/news-classification-tutorial/blob/main/{f})"

def save(filename, notebook):
    with open(os.path.join(FOLDER, filename), 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print("OK: " + filename)

# ── Chapter 4 ─────────────────────────────────────────────────────────────
save("chapter04_what_is_model.ipynb", nb([
    md(f"# Chapter 4 -- Model, Training, Prediction\n\n{badge('chapter04_what_is_model.ipynb')}\n\n**本章目标：** 从概念层面搞清楚什么是模型、什么是训练、为什么选这四个模型。\n\n**预计时间：** 30 分钟（本章以概念为主，代码较少）"),

    md("## 4.1 什么是模型（Model）？\n\n**模型** = 一套从输入到输出的映射规则。\n\n生活类比：\n- 医生看症状（输入）→ 判断是什么病（输出）：医生的「经验」就是模型\n- 银行看你的信用记录（输入）→ 决定是否贷款（输出）：审核规则就是模型\n- 我们的任务：看文章内容（输入）→ 判断是哪类新闻（输出）\n\n更准确地说：**模型是一个函数**，接收数据，输出预测。\n\n```\n输入（文章文本） → [模型] → 输出（World / Sports / Business / Science/Tech）\n```"),

    md("## 4.2 什么是训练（Training）？\n\n**训练** = 让机器从数据里自动总结规律。\n\n类比：学生刷题\n1. 看一道题和标准答案（输入 + 正确标签）\n2. 做出自己的答案\n3. 对比标准答案，发现自己哪错了\n4. 调整思路（更新参数）\n5. 反复练习，做对率越来越高\n\n机器也是这样：\n1. 把有标签的文章喂给模型\n2. 模型做出预测\n3. 和正确标签对比，计算误差\n4. 根据误差调整模型内部的参数\n5. 反复迭代，预测越来越准"),

    md("## 4.3 什么是预测（Prediction / Inference）？\n\n**预测** = 用训练好的模型处理新数据。\n\n类比：期末考试\n- 训练阶段：用练习题学习\n- 预测阶段：用学到的知识做考试题（考试题是没见过的新题）\n\n**关键：** 预测时用的数据，必须是模型在训练时从未见过的。"),

    md("## 4.4 什么是过拟合（Overfitting）？\n\n**过拟合** = 模型把训练集「背」下来了，但遇到新数据就不行了。\n\n类比：\n- 有个学生只背答案，不理解题目。模拟考试（训练集）全对，期末考试（测试集）完全不会做。\n- 这叫「背题」，不叫「学会」。\n\n**为什么要测试集？**\n- 测试集是模型从未见过的数据\n- 在测试集上评估，才能知道模型是真的学会了，还是只是「背题」\n- 测试集绝对不能参与训练！"),

    md("## 4.5 为什么选这四个模型？\n\n这个作业选了 4 个模型，从简单到复杂：\n\n| 模型 | 类型 | 为什么选它 |\n|------|------|----------|\n| **朴素贝叶斯（Naive Bayes）** | 传统机器学习 | 最简单，速度最快，作为「基准线」 |\n| **k 近邻（kNN）** | 传统机器学习 | 直觉最清晰，用「找相似」来分类 |\n| **卷积神经网络（CNN）** | 深度学习 | 擅长检测局部文字模式（短语）|\n| **双向 LSTM（BiLSTM）** | 深度学习 | 擅长理解长距离上下文和语序 |\n\n**四个模型覆盖了两大类方法：**\n- 传统机器学习：基于统计，速度快，可解释性强\n- 深度学习：基于神经网络，更强大，需要更多数据和计算资源"),

    md("## 4.6 sklearn 是什么？\n\n**scikit-learn**（简称 sklearn）是 Python 最流行的传统机器学习库。\n\n特点：\n- 已经实现了几十种算法（朴素贝叶斯、kNN、SVM、随机森林等）\n- **统一接口：** 所有模型用法相同：\n  ```python\n  model.fit(X_train, y_train)    # 训练\n  model.predict(X_test)          # 预测\n  model.score(X_test, y_test)    # 评估\n  ```\n- 不需要了解算法细节，直接用\n\n> 官方文档：https://scikit-learn.org/stable/"),

    md("## 4.7 TensorFlow / Keras 是什么？\n\n**TensorFlow** 是 Google 开发的深度学习框架，用于构建和训练神经网络。\n\n**Keras** 是 TensorFlow 的高层接口，像搭积木一样搭建神经网络：\n\n```python\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense\n\nmodel = Sequential([\n    Dense(128, activation='relu'),  # 第一层：128 个神经元\n    Dense(4, activation='softmax')  # 输出层：4 个类别\n])\nmodel.compile(optimizer='adam', loss='categorical_crossentropy')\nmodel.fit(X_train, y_train, epochs=10)\n```\n\n> 官方文档：https://www.tensorflow.org/tutorials"),

    code("# 用 sklearn 做一个简单演示：无需理解细节，感受一下流程\nfrom sklearn.datasets import make_classification\nfrom sklearn.naive_bayes import GaussianNB\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.metrics import accuracy_score\n\n# 生成一个假数据集（100 个样本，2 个特征，2 个类别）\nX, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)\n\n# 划分训练集和测试集\nX_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)\n\n# 三步走：创建模型 → 训练 → 预测\nmodel = GaussianNB()       # 1. 创建模型\nmodel.fit(X_tr, y_tr)      # 2. 训练（fit = 拟合）\ny_pred = model.predict(X_te)  # 3. 预测\n\nprint(f'测试集准确率：{accuracy_score(y_te, y_pred):.1%}')\nprint('这就是机器学习的完整流程！')"),

    md("## 总结\n\n| 概念 | 含义 |\n|------|------|\n| 模型（Model） | 从输入到输出的映射规则（函数）|\n| 训练（Training） | 让模型从有标签的数据里自动学习规律 |\n| 预测（Prediction） | 用训练好的模型处理新数据 |\n| 过拟合（Overfitting） | 模型背答案，遇到新数据失效 |\n| sklearn | 传统机器学习算法库，统一的 fit/predict 接口 |\n| TensorFlow/Keras | 深度学习框架，像搭积木一样搭神经网络 |\n\n**下一章 →** Chapter 5：把文字变成数字——TF-IDF")
]))

# ── Chapter 5 ─────────────────────────────────────────────────────────────
save("chapter05_tfidf.ipynb", nb([
    md(f"# Chapter 5 -- TF-IDF：把文字变成数字\n\n{badge('chapter05_tfidf.ipynb')}\n\n**本章目标：** 理解 TF-IDF 的原理，并用 sklearn 实现。\n\n**预计时间：** 45 分钟\n\n> 参考：Karen Sparck Jones 1972 年提出 IDF 概念，奠定了现代信息检索的基础。"),

    md("## 5.1 机器只懂数字\n\n你现在明白：模型需要输入数据、输出预测。\n\n但问题是：**文章是文字，模型只接受数字。**\n\n所以我们必须把文章先转换成数字。这个步骤叫**文本表示（Text Representation）**。"),

    md("## 5.2 最简单的方法：词袋模型（Bag of Words）\n\n**词袋模型**：不管词的顺序，只统计每个词出现了几次。\n\n类比：把文章里所有词「倒进一个袋子」，摇一摇，然后数每种词有多少个。\n\n**手算例子：**\n\n```\n文章 1：\"cat dog cat\"\n文章 2：\"dog fish\"\n文章 3：\"cat fish fish\"\n\n词汇表：[cat, dog, fish]\n\n          cat  dog  fish\n文章 1  [  2,   1,   0 ]\n文章 2  [  0,   1,   1 ]\n文章 3  [  1,   0,   2 ]\n```\n\n每篇文章变成了一行数字！这就是词袋矩阵。"),

    md("## 5.3 词袋模型的问题\n\n\"the\"、\"is\"、\"and\" 这类常见词在所有文章里都大量出现，数量高，但对判断类别没有帮助。\n\n我们需要一种方法，让「独特的词」权重高，让「到处都有的词」权重低。\n\n这就是 **TF-IDF**。"),

    md("## 5.4 TF：词频（Term Frequency）\n\n**TF(词, 文章)** = 这个词在这篇文章里出现了多少次（相对频率）\n\n```\nTF(\"quarterback\", 文章A) = 文章A中\"quarterback\"出现次数 / 文章A总词数\n```\n\n但问题还没解决：\"the\" 的 TF 也很高，因为它到处都出现。"),

    md("## 5.5 IDF：逆文档频率（Inverse Document Frequency）\n\n**IDF(词)** = 衡量这个词有多稀有\n\n```\nIDF(词) = log( 总文档数 / 包含该词的文档数 )\n```\n\n直觉解释：\n- \"the\" 在 128,000 篇文章里都出现 → IDF 接近 0（没价值）\n- \"quarterback\" 只在 500 篇体育文章里出现 → IDF 很高（很有价值）\n\n| 词 | 出现在多少文章里 | IDF（近似）|\n|---|---|---|\n| the | 128,000 | ≈ 0（没区分度）|\n| sports | 35,000 | 低 |\n| quarterback | 500 | 高 |\n| photosynthesis | 50 | 很高 |"),

    md("## 5.6 TF-IDF = TF × IDF\n\n$$\\text{TF-IDF}(词, 文章) = \\text{TF}(词, 文章) \\times \\text{IDF}(词)$$\n\n- 一个词在这篇文章里多 → TF 高\n- 这个词在所有文章里少见 → IDF 高\n- 两者乘积高的词，就是最能代表这篇文章的词\n\n**举例：** 一篇体育新闻里频繁出现 \"quarterback\"（TF 高），而 \"quarterback\" 只出现在体育文章里（IDF 高）→ TF-IDF 高 → 这个词非常能代表这篇文章是体育新闻。"),

    code("# 手动验证 TF-IDF 的直觉\nimport numpy as np\n\n# 3 篇迷你文章\ndocs = [\n    \"the cat sat on the mat\",\n    \"the dog played football at the stadium\",\n    \"the scientist discovered a new chemical compound\"\n]\n\n# 用 sklearn 计算 TF-IDF\nfrom sklearn.feature_extraction.text import TfidfVectorizer\n\nvectorizer = TfidfVectorizer()\nX = vectorizer.fit_transform(docs)\n\n# 特征名（词汇表）\nfeature_names = vectorizer.get_feature_names_out()\n\nprint('词汇表：', feature_names.tolist())\nprint('\\nTF-IDF 矩阵（每行是一篇文章，每列是一个词）：')\nprint(np.round(X.toarray(), 3))\n\n# 找每篇文章权重最高的词\nprint('\\n每篇文章权重最高的词：')\nfor i, doc in enumerate(docs):\n    tfidf_scores = X[i].toarray()[0]\n    top_idx = tfidf_scores.argsort()[-3:][::-1]\n    print(f'  文章{i+1}: {[feature_names[j] for j in top_idx]}')"),

    md("## 5.7 在 dataset.csv 上实现 TF-IDF"),

    code("import pandas as pd\nimport re\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\n\n# ── 数据准备（和 Chapter 3 相同）──────────────────────────\ndef clean_text(text):\n    text = str(text).lower()\n    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)\n    text = re.sub(r'[^a-z\\s]', ' ', text)\n    return re.sub(r'\\s+', ' ', text).strip()\n\ndf = pd.read_csv('dataset.csv').dropna(subset=['Class']).copy()\ndf['Description'] = df['Description'].fillna('')\ndf['text'] = (df['Title'] + ' ' + df['Description']).apply(clean_text)\nle = LabelEncoder()\ny = le.fit_transform(df['Class'])\nX_train, X_test, y_train, y_test = train_test_split(\n    df['text'].values, y, test_size=0.3, random_state=42, stratify=y)\nprint(f'训练集：{len(X_train):,}，测试集：{len(X_test):,}')"),

    code("# 创建 TF-IDF 向量化器\ntfidf = TfidfVectorizer(\n    max_features=20000,    # 只保留最重要的 20000 个词（词汇表大小）\n    ngram_range=(1, 2),    # 考虑单词和两词组合（bigram）\n                           # 例如 'stock market' 比 'stock' 和 'market' 单独更有价值\n    sublinear_tf=True,     # 用 1+log(tf) 代替原始 tf，防止高频词垄断\n    min_df=2               # 至少在 2 篇文章里出现，过滤极少见的词\n)\n\n# fit_transform：在训练集上学习词汇表，并转换训练集\n# （重要！只在训练集上 fit，测试集只 transform）\nX_train_tfidf = tfidf.fit_transform(X_train)\n\n# transform：只用学到的词汇表转换测试集（不重新学习）\nX_test_tfidf = tfidf.transform(X_test)\n\nprint(f'训练集 TF-IDF 矩阵形状：{X_train_tfidf.shape}')\nprint(f'  → {X_train_tfidf.shape[0]:,} 篇文章，每篇用 {X_train_tfidf.shape[1]:,} 个数字表示')\nprint(f'\\n稀疏度：{(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0]*X_train_tfidf.shape[1])):.1%}')\nprint('（绝大多数位置是 0，因为每篇文章只用到极少数词）')"),

    md("## 5.8 为什么只在训练集上 fit？\n\n**答案：防止数据泄露（Data Leakage）**\n\n如果在全部数据（包括测试集）上 fit TF-IDF：\n- 词汇表里会包含测试集独有的词\n- 这意味着模型「偷看」了测试集\n- 评估结果就不真实了\n\n正确做法：\n```python\ntfidf.fit_transform(X_train)  # 在训练集上学词汇表并转换\ntfidf.transform(X_test)       # 用同一个词汇表转换测试集\n```"),

    code("# 查看某篇文章权重最高的 10 个词\nimport numpy as np\n\n# 取第一篇训练文章\nfeature_names = tfidf.get_feature_names_out()\nsample = X_train_tfidf[0]\n\n# 找权重最高的词\ntop_indices = sample.toarray()[0].argsort()[-10:][::-1]\nprint('第一篇训练文章的 Top 10 TF-IDF 词：')\nfor idx in top_indices:\n    print(f'  {feature_names[idx]:30s} {sample.toarray()[0][idx]:.4f}')\n\nprint('\\n对应的文章类别：', le.classes_[y_train[0]])"),

    md("## 练习"),

    code("# 练习：找出「Sports」类文章中 TF-IDF 权重最高的 20 个词\n# 提示：先用 y_train==2（Sports 的编码）筛选出运动类文章，\n# 然后对这些文章的 TF-IDF 矩阵求平均\n\nimport numpy as np\n\n# 你的代码：\n# sports_mask = (y_train == ?)  # Sports 是哪个编码？用 le.transform(['Sports'])\n# sports_tfidf = X_train_tfidf[sports_mask]\n# avg_tfidf = np.array(sports_tfidf.mean(axis=0)).flatten()\n# top20 = avg_tfidf.argsort()[-20:][::-1]\n# for i in top20:\n#     print(feature_names[i], avg_tfidf[i])\n"),

    md("## 总结\n\n| 概念 | 含义 |\n|------|------|\n| 词袋模型（BoW）| 只数词频，不管顺序 |\n| TF（词频）| 词在本文中出现的频率 |\n| IDF（逆文档频率）| 词在所有文章中有多稀有 |\n| TF-IDF | TF × IDF，突出独特词 |\n| max_features | 词汇表大小上限 |\n| ngram_range=(1,2) | 同时考虑单词和双词组合 |\n| fit 只用训练集 | 防止数据泄露 |\n\n**下一章 →** Chapter 6：第一个分类器——朴素贝叶斯")
]))

# ── Chapter 6 ─────────────────────────────────────────────────────────────
save("chapter06_naive_bayes.ipynb", nb([
    md(f"# Chapter 6 -- 第一个分类器：朴素贝叶斯（Naive Bayes）\n\n{badge('chapter06_naive_bayes.ipynb')}\n\n**本章目标：** 训练第一个真实的分类器，并学会读懂评估结果。\n\n**预计时间：** 60 分钟"),

    md("## 6.1 从概率说起\n\n**概率** 是衡量某件事「可能发生」的程度，范围 0 到 1（或 0% 到 100%）。\n\n- P(明天下雨) = 0.7 → 70% 的可能性下雨\n- P(掷骰子得到 6) = 1/6 ≈ 0.17\n\n朴素贝叶斯用概率来分类：**看到这篇文章里的词，每个类别的概率是多少？**"),

    md("## 6.2 贝叶斯定理\n\n核心问题：看到一篇文章里有词 \"quarterback\"，它属于 Sports 类的概率是多少？\n\n$$P(Sports \\mid \\text{包含 quarterback}) = \\frac{P(\\text{包含 quarterback} \\mid Sports) \\times P(Sports)}{P(\\text{包含 quarterback})}$$\n\n用人话说：\n- **P(Sports)** = 先验概率：不看文章内容，本来有多少比例是体育新闻？（约 25%）\n- **P(包含 quarterback | Sports)** = 体育新闻里出现 \"quarterback\" 的频率高不高？（高！）\n- **结果** = 综合考虑后，这篇文章是体育的概率\n\n对所有 4 个类别都这样算，概率最高的类别就是预测结果。"),

    md("## 6.3 为什么叫「朴素」（Naive）？\n\n假设文章里所有词**相互独立**，互不影响。\n\n现实中这当然不对——\"artificial\" 和 \"intelligence\" 经常一起出现。但即使这个假设是错的，模型在实践中效果依然不错。\n\n这就是「朴素」的含义：用了一个简单但不完全正确的假设，换来了极快的计算速度。"),

    md("## 6.4 准备数据（复用 Chapter 5 的代码）"),

    code("import pandas as pd, numpy as np, re\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\n\ndef clean_text(t):\n    t = str(t).lower()\n    t = re.sub(r'http\\S+|www\\S+|https\\S+', '', t)\n    t = re.sub(r'[^a-z\\s]', ' ', t)\n    return re.sub(r'\\s+', ' ', t).strip()\n\ndf = pd.read_csv('dataset.csv').dropna(subset=['Class']).copy()\ndf['Description'] = df['Description'].fillna('')\ndf['text'] = (df['Title'] + ' ' + df['Description']).apply(clean_text)\nle = LabelEncoder()\ny = le.fit_transform(df['Class'])\nX_train, X_test, y_train, y_test = train_test_split(\n    df['text'].values, y, test_size=0.3, random_state=42, stratify=y)\n\ntfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2)\nX_train_tfidf = tfidf.fit_transform(X_train)\nX_test_tfidf  = tfidf.transform(X_test)\nprint('数据准备完毕！')"),

    md("## 6.5 训练朴素贝叶斯"),

    code("from sklearn.naive_bayes import MultinomialNB\n\n# alpha=0.1 是平滑参数（Laplace Smoothing）\n# 作用：防止「如果训练时没见过这个词，概率就是 0」的问题\n# alpha 越大，平滑越强；alpha=0.1 比默认值 1.0 更精准\nnb_model = MultinomialNB(alpha=0.1)\n\n# 训练（fit）\nnb_model.fit(X_train_tfidf, y_train)\n\nprint('朴素贝叶斯训练完成！')\nprint(f'模型学到了 {X_train_tfidf.shape[1]:,} 个特征（词）的统计规律')"),

    code("# 预测\ny_pred_nb = nb_model.predict(X_test_tfidf)\n\n# 看几个具体预测\nprint('前 5 个测试样本的预测结果：')\nfor i in range(5):\n    true_label = le.classes_[y_test[i]]\n    pred_label = le.classes_[y_pred_nb[i]]\n    correct = 'O' if true_label == pred_label else 'X'\n    print(f'  [{correct}] 真实：{true_label:15s} 预测：{pred_label}')"),

    md("## 6.6 评估指标详解\n\n### Accuracy（准确率）\n\n**定义：** 预测正确的比例\n\n$$\\text{Accuracy} = \\frac{\\text{预测正确的数量}}{\\text{总数量}}$$\n\n**局限：** 当类别不平衡时，Accuracy 会误导。（本数据集平衡，所以 Accuracy 可靠。）"),

    md("### Precision（精确率）和 Recall（召回率）\n\n以「Sports」类为例：\n\n**Precision（精确率）：**\n> 我预测为 Sports 的文章里，真的是 Sports 的有多少？\n\n$$\\text{Precision} = \\frac{\\text{真 Sports 且预测 Sports}}{\\text{所有预测为 Sports 的}}$$\n\n**Recall（召回率）：**\n> 所有真正的 Sports 文章里，我找到了多少？\n\n$$\\text{Recall} = \\frac{\\text{真 Sports 且预测 Sports}}{\\text{所有真正是 Sports 的}}$$\n\n**F1-Score：** Precision 和 Recall 的调和平均，两者都高 F1 才高。\n$$F1 = 2 \\times \\frac{\\text{Precision} \\times \\text{Recall}}{\\text{Precision} + \\text{Recall}}$$"),

    code("from sklearn.metrics import (\n    accuracy_score, f1_score, precision_score, recall_score,\n    classification_report, confusion_matrix, ConfusionMatrixDisplay\n)\nimport matplotlib.pyplot as plt\n\n# 基本指标\nacc_nb  = accuracy_score(y_test, y_pred_nb)\nf1_nb   = f1_score(y_test, y_pred_nb, average='weighted')\nprec_nb = precision_score(y_test, y_pred_nb, average='weighted')\nrec_nb  = recall_score(y_test, y_pred_nb, average='weighted')\n\nprint('=== Naive Bayes 测试集结果 ===')\nprint(f'  Accuracy          : {acc_nb:.4f} ({acc_nb:.1%})')\nprint(f'  Weighted F1-Score : {f1_nb:.4f}')\nprint(f'  Weighted Precision: {prec_nb:.4f}')\nprint(f'  Weighted Recall   : {rec_nb:.4f}')\nprint('\\n=== 逐类别详细报告 ===')\nprint(classification_report(y_test, y_pred_nb, target_names=le.classes_))"),

    code("# 混淆矩阵：行=真实类别，列=预测类别，对角线=预测正确\ncm = confusion_matrix(y_test, y_pred_nb)\n\nfig, ax = plt.subplots(figsize=(7, 5))\nConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap='Blues', colorbar=False)\nax.set_title('Naive Bayes -- Confusion Matrix')\nplt.tight_layout()\nplt.show()\n\n# 分析最大的错误\nprint('最多的误判：')\nnp.fill_diagonal(cm, 0)  # 把对角线（正确预测）清零\nmax_err = np.unravel_index(cm.argmax(), cm.shape)\nprint(f'  把 {le.classes_[max_err[0]]} 误判为 {le.classes_[max_err[1]]}: {cm[max_err]} 次')\nprint('原因：这两类在词汇上有一定重叠（科技公司新闻可能同时涉及商业和科技）')"),

    md("## 6.7 保存模型"),

    code("import joblib\n\n# joblib 专门用来保存 sklearn 模型\njoblib.dump(nb_model, 'naive_bayes_model.joblib')\njoblib.dump(tfidf, 'tfidf_vectorizer.joblib')\njoblib.dump(le, 'label_encoder.joblib')\nprint('模型已保存！')"),

    md("## 练习"),

    code("# 练习：用训练好的模型预测下面这段文字属于哪个类别\ntest_article = \"The Olympic champion won three gold medals in swimming at the World Championships\"\n\n# 步骤：\n# 1. clean_text(test_article)\n# 2. tfidf.transform([cleaned])\n# 3. nb_model.predict(transformed)\n# 4. le.inverse_transform(prediction)\n\n# 你的代码：\n"),

    md("## 总结\n\n**Naive Bayes 结果：Accuracy ≈ 90%**\n\n这对一个最简单的统计模型来说已经相当不错！\n\n| 概念 | 含义 |\n|------|------|\n| 贝叶斯定理 | 用先验概率和条件概率推断后验概率 |\n| 朴素假设 | 词与词之间相互独立（简化计算）|\n| Accuracy | 总体正确率 |\n| Precision | 预测为某类时，真的是那类的比例 |\n| Recall | 某类文章里，被正确找出的比例 |\n| F1-Score | Precision 和 Recall 的调和平均 |\n| 混淆矩阵 | 展示每种误判情况的数量 |\n\n**下一章 →** Chapter 7：第二个分类器——kNN")
]))

# ── Chapter 7 ─────────────────────────────────────────────────────────────
save("chapter07_knn.ipynb", nb([
    md(f"# Chapter 7 -- 第二个分类器：k 近邻（kNN）\n\n{badge('chapter07_knn.ipynb')}\n\n**本章目标：** 理解 kNN 的原理，解决高维问题，用交叉验证选最优 k。\n\n**预计时间：** 60 分钟\n\n> 延伸阅读：[维度灾难（Wikipedia）](https://en.wikipedia.org/wiki/Curse_of_dimensionality)"),

    md("## 7.1 kNN 的直觉：物以类聚\n\n**核心思想：** 要判断一个新样本的类别，就看它的 k 个最近邻居是什么类别。\n\n生活类比：\n- 你搬到一个新城市，想知道这个小区是什么风格\n- 看看附近 5 家餐馆（k=5）：3 家川菜、1 家粤菜、1 家火锅\n- 多数是川菜 → 判断这是个川菜风格的街区\n\n在机器学习里：\n- 每篇文章是空间里的一个**点**（用数字向量表示）\n- 相似的文章距离近\n- 新文章的类别 = 距它最近的 k 篇已知文章中，出现最多的类别"),

    md("## 7.2 维度灾难（Curse of Dimensionality）\n\n我们的 TF-IDF 矩阵有 20,000 维（每篇文章是 20,000 维空间里的一个点）。\n\n问题：**在高维空间里，「距离」的概念几乎失效了。**\n\n直觉解释：\n- 在 1D（一条线）上：随机放 100 个点，它们分布比较均匀，总能找到真正近的邻居\n- 在 2D（一个平面）上：需要更多的点才能覆盖空间\n- 在 20000D 上：所有点之间的距离变得几乎一样远！\n  就像在宇宙里找邻居，每颗星离你都差不多远\n\n**结果：** kNN 在原始 TF-IDF 空间上效果会很差，必须先降维。"),

    md("## 7.3 解决方法：SVD 降维（Latent Semantic Analysis）\n\n**Truncated SVD（截断奇异值分解）** = 从高维数据里找出最重要的「方向」，投影到低维空间。\n\n类比：\n- 你有一张 4K 分辨率的图（高维）\n- 把它压缩成 720P（低维）\n- 大部分视觉信息还在，但数据量少多了\n\n我们把 20,000 维压缩到 **200 维**：\n- 保留了约 15.6% 的方差信息\n- 去掉了噪声\n- 降维后的空间里，相似文章的距离更有意义\n\n**LSA（Latent Semantic Analysis）** = 用 SVD 对文本 TF-IDF 矩阵降维，这是信息检索领域的经典方法。"),

    code("import pandas as pd, numpy as np, re\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\n\ndef clean_text(t):\n    t = str(t).lower()\n    t = re.sub(r'http\\S+|www\\S+|https\\S+', '', t)\n    t = re.sub(r'[^a-z\\s]', ' ', t)\n    return re.sub(r'\\s+', ' ', t).strip()\n\ndf = pd.read_csv('dataset.csv').dropna(subset=['Class']).copy()\ndf['Description'] = df['Description'].fillna('')\ndf['text'] = (df['Title'] + ' ' + df['Description']).apply(clean_text)\nle = LabelEncoder()\ny = le.fit_transform(df['Class'])\nX_train, X_test, y_train, y_test = train_test_split(\n    df['text'].values, y, test_size=0.3, random_state=42, stratify=y)\n\ntfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2), sublinear_tf=True, min_df=2)\nX_train_tfidf = tfidf.fit_transform(X_train)\nX_test_tfidf  = tfidf.transform(X_test)\nprint('数据准备完毕！')"),

    code("from sklearn.decomposition import TruncatedSVD\nfrom sklearn.preprocessing import Normalizer\n\n# 步骤 1：SVD 降维到 200 维\nsvd = TruncatedSVD(n_components=200, random_state=42)\n\n# 步骤 2：L2 归一化（让每个向量长度为 1）\n# 好处：这样欧氏距离等价于余弦相似度，更适合文本\nnormalizer = Normalizer(copy=False)\n\n# 注意：只在训练集上 fit，测试集只 transform\nX_train_lsa = normalizer.fit_transform(svd.fit_transform(X_train_tfidf))\nX_test_lsa  = normalizer.transform(svd.transform(X_test_tfidf))\n\nprint(f'降维前：{X_train_tfidf.shape[1]:,} 维')\nprint(f'降维后：{X_train_lsa.shape[1]} 维')\nprint(f'保留方差：{svd.explained_variance_ratio_.sum():.1%}')\nprint(f'\\n降维后训练集形状：{X_train_lsa.shape}')\n\n# 保存降维器（后面 predict 函数需要用）\nimport joblib\njoblib.dump(svd, 'knn_svd.joblib')\njoblib.dump(normalizer, 'knn_normalizer.joblib')"),

    md("## 7.4 怎么选 k：交叉验证\n\n**问题：** k 越大越好还是越小越好？\n- k=1：只看最近的 1 个邻居，容易受噪声影响（过拟合）\n- k=100：看太多邻居，把离得很远的文章也纳入，判断变模糊（欠拟合）\n\n**交叉验证（Cross-Validation）：**\n把训练集再分成 5 份，轮流用 1 份验证、4 份训练，重复 5 次，取平均成绩。\n\n```\n训练集 → 分成5份\n第1轮：[验证|训练|训练|训练|训练] → 得到 F1_1\n第2轮：[训练|验证|训练|训练|训练] → 得到 F1_2\n...（共5轮）\n平均 F1 = (F1_1 + ... + F1_5) / 5\n```\n\n**为什么不直接用测试集选 k？** 那叫「数据泄露」——你根据考试结果调整了学习策略，考试就失去了意义。"),

    code("from sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.model_selection import StratifiedKFold, cross_val_score\nimport matplotlib.pyplot as plt\n\nk_values = [1, 3, 5, 7, 9, 11]\ncv_scores = []\n\n# 5 折分层交叉验证\nskf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n\nprint('正在进行交叉验证，请耐心等待（约需 10-20 分钟）...')\nfor k in k_values:\n    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)\n    scores = cross_val_score(knn, X_train_lsa, y_train, cv=skf,\n                             scoring='f1_weighted', n_jobs=-1)\n    cv_scores.append(scores.mean())\n    print(f'  k={k:2d} -> CV F1 = {scores.mean():.4f} (+-{scores.std():.4f})')\n\nbest_k = k_values[int(np.argmax(cv_scores))]\nprint(f'\\n最优 k = {best_k}（CV F1 = {max(cv_scores):.4f}）')"),

    code("# 画 k vs F1 图\nfig, ax = plt.subplots(figsize=(8, 5))\nax.plot(k_values, cv_scores, marker='o', linewidth=2, markersize=9, color='steelblue')\nax.axvline(best_k, color='red', linestyle='--', label=f'Best k={best_k}')\nax.set_title('kNN: Cross-Validation F1 vs. k', fontsize=14, fontweight='bold')\nax.set_xlabel('k (Number of Neighbours)', fontsize=12)\nax.set_ylabel('Weighted F1-Score (5-fold CV)', fontsize=12)\nax.set_xticks(k_values)\nax.legend(fontsize=11)\nax.grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()\nprint('观察：F1 随 k 增大而单调增加（k=11 最优），说明更多邻居更稳定。')"),

    code("from sklearn.metrics import (accuracy_score, f1_score, precision_score,\n                             recall_score, classification_report,\n                             confusion_matrix, ConfusionMatrixDisplay)\n\n# 用最优 k 训练最终模型\nknn_model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean', n_jobs=-1)\nknn_model.fit(X_train_lsa, y_train)\n\ny_pred_knn = knn_model.predict(X_test_lsa)\n\nprint(f'=== kNN (k={best_k}) 测试集结果 ===')\nprint(f'  Accuracy  : {accuracy_score(y_test, y_pred_knn):.4f}')\nprint(f'  F1-Score  : {f1_score(y_test, y_pred_knn, average=\"weighted\"):.4f}')\nprint(f'  Precision : {precision_score(y_test, y_pred_knn, average=\"weighted\"):.4f}')\nprint(f'  Recall    : {recall_score(y_test, y_pred_knn, average=\"weighted\"):.4f}')\nprint()\nprint(classification_report(y_test, y_pred_knn, target_names=le.classes_))\n\ncm = confusion_matrix(y_test, y_pred_knn)\nfig, ax = plt.subplots(figsize=(7,5))\nConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap='Greens', colorbar=False)\nax.set_title(f'kNN (k={best_k}) -- Confusion Matrix')\nplt.tight_layout()\nplt.show()\n\njoblib.dump(knn_model, 'knn_model.joblib')\nprint('kNN 模型已保存！')"),

    md("## 总结\n\n**kNN 结果：Accuracy ≈ 84%（低于 Naive Bayes）**\n\n原因：LSA 只保留了 15.6% 的方差，丢失了大量信息。\n\n| 概念 | 含义 |\n|------|------|\n| kNN | 找 k 个最近邻居，用多数类别投票 |\n| 维度灾难 | 高维空间里距离失效 |\n| SVD/LSA | 降维，保留主要信息 |\n| L2 归一化 | 让欧氏距离等价于余弦相似度 |\n| 交叉验证 | 在训练集内评估，避免数据泄露 |\n\n**下一章 →** Chapter 8：神经网络基础")
]))

# ── Chapter 8 ─────────────────────────────────────────────────────────────
save("chapter08_neural_networks.ipynb", nb([
    md(f"# Chapter 8 -- 神经网络基础\n\n{badge('chapter08_neural_networks.ipynb')}\n\n**本章目标：** 理解神经网络是什么，训练的本质是什么。\n\n**预计时间：** 60 分钟\n\n> 强烈推荐：[3Blue1Brown 神经网络可视化系列](https://www.3blue1brown.com/topics/neural-networks)（即使不懂英文，动画也能帮你理解）"),

    md("## 8.1 为什么需要神经网络？\n\nTF-IDF + 朴素贝叶斯能做到约 90% 的准确率，已经不错了。\n\n但它们的根本局限：\n- TF-IDF 不理解语义：\"football\" 和 \"soccer\" 是同一件事，但被当成完全不同的词\n- 它不理解语序：\"dog bites man\" 和 \"man bites dog\" 意思截然不同，但词袋看起来一样\n\n**神经网络** 能学习更复杂的模式，突破这些限制。"),

    md("## 8.2 神经元（Neuron）是什么？\n\n人工神经元模仿生物神经元：\n\n```\n输入 x1, x2, x3\n     ↓  ↓  ↓\n   × w1 × w2 × w3    ← 权重（weights），是模型「学习」的东西\n     ↓\n  求和 + 偏置 b      ← z = w1*x1 + w2*x2 + w3*x3 + b\n     ↓\n  激活函数 f(z)      ← 引入非线性\n     ↓\n   输出 y\n```\n\n**权重（Weights）** 是模型里唯一需要学习的参数。训练的本质就是调整权重，让预测越来越准。"),

    code("# 用纯 Python 手写一个神经元，感受一下\nimport numpy as np\n\n# 一个有 3 个输入的神经元\ndef neuron(x, w, b):\n    \"\"\"\n    x: 输入向量 [x1, x2, x3]\n    w: 权重向量 [w1, w2, w3]\n    b: 偏置（bias）\n    \"\"\"\n    z = np.dot(x, w) + b  # 加权求和 + 偏置\n    output = max(0, z)     # ReLU 激活函数：负数变 0，正数保留\n    return output\n\n# 测试\nx = np.array([0.5, 0.3, 0.8])  # 假设的输入（3 个 TF-IDF 特征）\nw = np.array([0.2, -0.4, 0.7]) # 权重（随机初始化）\nb = 0.1                          # 偏置\n\nresult = neuron(x, w, b)\nprint(f'输入：{x}')\nprint(f'权重：{w}')\nprint(f'加权求和 + 偏置：{np.dot(x,w)+b:.4f}')\nprint(f'ReLU 激活后输出：{result:.4f}')"),

    md("## 8.3 激活函数（Activation Function）\n\n**为什么需要激活函数？**\n如果没有激活函数，多层神经元叠在一起等于一层（因为线性函数的组合还是线性函数）。激活函数引入非线性，让网络能学习复杂模式。\n\n**常用激活函数：**\n\n- **ReLU（Rectified Linear Unit）：** `f(x) = max(0, x)`\n  - 负数变 0，正数保留\n  - 计算快，效果好，深度学习最常用\n\n- **Softmax：** 把数字变成概率分布（所有类别概率加起来 = 1）\n  - 用于最后一层输出层\n  - 例：输出 [2.1, 0.3, 5.2, 1.1] → Softmax → [0.07, 0.01, 0.89, 0.03]（概率）"),

    code("import matplotlib.pyplot as plt\nimport numpy as np\n\nx = np.linspace(-4, 4, 200)\n\nfig, axes = plt.subplots(1, 2, figsize=(11, 4))\n\n# ReLU\nrelu = np.maximum(0, x)\naxes[0].plot(x, relu, 'steelblue', linewidth=2.5)\naxes[0].axhline(0, color='gray', linewidth=0.8)\naxes[0].axvline(0, color='gray', linewidth=0.8)\naxes[0].set_title('ReLU: f(x) = max(0, x)', fontsize=13)\naxes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')\naxes[0].grid(True, alpha=0.3)\naxes[0].annotate('负数变0', xy=(-2, 0), xytext=(-3.5, 1),\n                 arrowprops=dict(arrowstyle='->', color='red'), color='red', fontsize=10)\n\n# Softmax 示例\nlogits = np.array([2.1, 0.3, 5.2, 1.1])\nexp_logits = np.exp(logits)\nsoftmax = exp_logits / exp_logits.sum()\naxes[1].bar(['World', 'Business', 'Sports', 'Science/Tech'],\n            softmax, color=['steelblue','coral','mediumseagreen','mediumpurple'])\naxes[1].set_title('Softmax Output (sum=1.00)', fontsize=13)\naxes[1].set_ylabel('Probability')\nfor i, v in enumerate(softmax):\n    axes[1].text(i, v+0.01, f'{v:.2f}', ha='center', fontsize=10)\n\nplt.tight_layout()\nplt.show()"),

    md("## 8.4 前向传播（Forward Propagation）\n\n数据从输入层流向输出层的过程：\n\n```\n输入层（文章特征）\n    ↓\n隐藏层 1（Dense 128 neurons）: 每个神经元 = 加权求和 + ReLU\n    ↓\n隐藏层 2（Dense 64 neurons）\n    ↓\n输出层（Dense 4 neurons）: Softmax → 4 个类别的概率\n```\n\n最后概率最大的类别就是预测结果。"),

    md("## 8.5 损失函数（Loss Function）\n\n**损失函数** 衡量预测有多错。\n\n分类问题常用 **Cross-Entropy Loss（交叉熵损失）**：\n- 如果真实类别是 Sports，模型预测 Sports 概率 = 0.9 → Loss 小（好）\n- 如果真实类别是 Sports，模型预测 Sports 概率 = 0.1 → Loss 大（差）\n\n训练的目标：**最小化 Loss。**"),

    md("## 8.6 反向传播（Backpropagation）+ 梯度下降\n\n**梯度下降（Gradient Descent）直觉：**\n想象你在山上，目标是到达山谷（Loss 最小点）：\n- 站在当前位置，看哪个方向向下最陡\n- 往那个方向走一小步\n- 重复，直到到达谷底\n\n**反向传播：** 自动计算「每个权重对 Loss 的影响」，告诉梯度下降该往哪个方向调整权重。\n\n你不需要手写这些计算，Keras 自动完成。\n\n> 深入学习：[Stanford CS231n 反向传播讲义](https://cs231n.github.io/optimization-2/)"),

    md("## 8.7 关键超参数\n\n| 参数 | 含义 | 类比 |\n|------|------|------|\n| **Epoch** | 把整个训练集看一遍 = 1 个 epoch | 把练习题做一遍 |\n| **Batch Size** | 每次更新权重用多少条数据 | 每做 256 道题对一次答案 |\n| **Learning Rate** | 每步走多大 | 步子大小 |\n| **Validation Split** | 从训练集留出一部分做验证 | 用部分练习题自测 |"),

    code("import tensorflow as tf\nfrom tensorflow.keras.models import Sequential\nfrom tensorflow.keras.layers import Dense, Dropout\nfrom tensorflow.keras.callbacks import EarlyStopping\nfrom tensorflow.keras.utils import to_categorical\nimport pandas as pd, numpy as np, re\nfrom sklearn.feature_extraction.text import TfidfVectorizer\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import LabelEncoder\n\ndef clean_text(t):\n    t = str(t).lower()\n    t = re.sub(r'http\\S+|www\\S+|https\\S+', '', t)\n    t = re.sub(r'[^a-z\\s]', ' ', t)\n    return re.sub(r'\\s+', ' ', t).strip()\n\ndf = pd.read_csv('dataset.csv').dropna(subset=['Class']).copy()\ndf['Description'] = df['Description'].fillna('')\ndf['text'] = (df['Title'] + ' ' + df['Description']).apply(clean_text)\nle = LabelEncoder()\ny = le.fit_transform(df['Class'])\nX_train, X_test, y_train, y_test = train_test_split(\n    df['text'].values, y, test_size=0.3, random_state=42, stratify=y)\n\ntfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True, min_df=2)\nX_tr = tfidf.fit_transform(X_train).toarray()  # Dense 层需要稠密矩阵\nX_te = tfidf.transform(X_test).toarray()\ny_tr_cat = to_categorical(y_train, 4)\ny_te_cat = to_categorical(y_test, 4)\nprint('数据准备完毕！')"),

    code("# 搭建一个简单的全连接神经网络，感受 Keras 的使用方式\nmodel = Sequential([\n    Dense(256, activation='relu', input_shape=(5000,)),  # 隐藏层 1\n    Dropout(0.3),                                         # Dropout：防止过拟合\n    Dense(128, activation='relu'),                        # 隐藏层 2\n    Dropout(0.3),\n    Dense(4, activation='softmax')                        # 输出层：4 个类别\n], name='SimpleNN')\n\n# compile：指定优化器、损失函数、评估指标\nmodel.compile(\n    optimizer='adam',              # Adam 是最常用的优化器，自动调整学习率\n    loss='categorical_crossentropy',  # 多分类问题的标准损失函数\n    metrics=['accuracy']\n)\n\nmodel.summary()  # 打印网络结构"),

    code("# EarlyStopping：当验证集 loss 连续 3 轮不下降时，自动停止训练\nearly_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)\n\nhistory = model.fit(\n    X_tr, y_tr_cat,\n    epochs=20,               # 最多训练 20 轮\n    batch_size=256,          # 每批 256 条数据\n    validation_split=0.1,   # 训练集的 10% 用于验证\n    callbacks=[early_stop],  # 早停\n    verbose=1\n)\nprint(f'实际训练了 {len(history.history[\"loss\"])} 轮（早停生效）')"),

    code("# 画训练曲线\nfig, axes = plt.subplots(1, 2, figsize=(12, 4))\nfor i, (metric, title) in enumerate([('accuracy','Accuracy'),('loss','Loss')]):\n    axes[i].plot(history.history[metric], label='Train')\n    axes[i].plot(history.history[f'val_{metric}'], label='Validation')\n    axes[i].set_title(f'Simple NN -- {title}', fontsize=13)\n    axes[i].set_xlabel('Epoch')\n    axes[i].set_ylabel(title)\n    axes[i].legend()\n    axes[i].grid(True, alpha=0.3)\nplt.tight_layout()\nplt.show()\nprint('观察：训练曲线和验证曲线接近 = 没有过拟合。若验证 loss 上升而训练 loss 继续下降 = 过拟合！')"),

    md("## 总结\n\n| 概念 | 含义 |\n|------|------|\n| 神经元 | 加权求和 + 激活函数 |\n| 权重（Weights）| 模型学习的参数 |\n| 前向传播 | 数据从输入到输出的计算过程 |\n| 损失函数 | 衡量预测有多错（越小越好）|\n| 反向传播 | 计算每个权重对 Loss 的影响 |\n| 梯度下降 | 沿 Loss 减小方向更新权重 |\n| EarlyStopping | 验证 loss 不降时自动停止 |\n| Dropout | 随机关闭神经元，防止过拟合 |\n\n**下一章 →** Chapter 9：词向量 Embedding")
]))

print("Chapter 4-8 done")
