import pymysql
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from scipy.special import softmax
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("D:\postgraduate program\CN_TextSummarization\django_news\ClassifierModel\Model")
model = AutoModelForSequenceClassification.from_pretrained("D:\postgraduate program\CN_TextSummarization\django_news\ClassifierModel\Model").to(device)
# 初始化数据库连接
db = pymysql.connect(host="localhost", user="root", password="1234", database="textsummarizationdatabase")
cursor = db.cursor()

# 初始化 T5 模型和 tokenizer

# 查询数据库并进行文本摘要和情感分析
query1 = "SELECT id, content FROM shanghaicompany"
query2 = "SELECT id, content FROM shanghaieconomynewspaper"
query3 = "SELECT id, content FROM sinaeconomynew"
query4 = "SELECT id, content FROM news WHERE scores IS NULL AND sentiment IS NULL AND content IS NOT NULL;"
cursor.execute(query4)
results = cursor.fetchall()

for row in results:
    id, content = row
    if content is None:
        continue
    inputs = tokenizer.encode(content, return_tensors='pt', max_length=512, truncation=True).to(device)
    classify = model(inputs)
    scores = classify[0][0].cpu().detach().numpy()
    scores = np.exp(scores) / np.exp(scores).sum()
    sentiment = np.argmax(scores)

    # 更新数据库中的记录
    update_query1 = "UPDATE shanghaicompany SET scores=%s, sentiment=%s WHERE id=%s"
    update_query2 = "UPDATE shanghaieconomynewspaper SET scores=%s, sentiment=%s WHERE id=%s"
    update_query3 = "UPDATE sinaeconomynew SET scores=%s, sentiment=%s WHERE id=%s"
    update_query4 = "UPDATE news SET scores=%s, sentiment=%s WHERE id=%s"
    cursor.execute(update_query4, (str(scores), str(sentiment), id))
    db.commit()

# 关闭数据库连接
db.close()
