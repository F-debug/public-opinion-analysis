# public-opinion-analysis
项目是一个NLP中的一个情感分析的业务，属于二分类任务。数据是舆情系统中从某电商平台上爬取下来的评论数据。人工对数据进行标记，分为两个类：分别为正面和负面。在很多模型进行比较后，决定用卷积网络，取得了很好的效果

# 文本预处理：data_preprocess.py
电商数据为csv格式，由evalution和label两个字段组成，风别为用户评论和正负面标签。对原始的文本进行分词，转编码等预处理

模型训练：net.py和text_classification.py
net.py:CNN模型和模型的参数
text_classification.py：训练模型

模型预测：demo.py
保存模型，输出score为0.9334
