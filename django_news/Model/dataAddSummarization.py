import pymysql
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MT5ForConditionalGeneration.from_pretrained("Model").to("cuda:0")
# model_heack =torch.load("Model/pytorch_model.bin")
tokenizer = T5Tokenizer.from_pretrained("heack/HeackMT5-ZhSum100k")
# 初始化数据库连接
db = pymysql.connect(host="localhost", user="root", password="1234", database="textsummarizationdatabase")
cursor = db.cursor()

# 初始化 T5 模型和 tokenizer

# 查询数据库并进行文本摘要和情感分析
query1 = "SELECT id, content FROM shanghaicompany"
query2 = "SELECT id, content FROM shanghaieconomynewspaper"
query3 = "SELECT id, content FROM sinaeconomynew"
query4 = "SELECT id, content FROM news WHERE summary IS NULL AND content IS NOT NULL;"
cursor.execute(query4)
results = cursor.fetchall()

for row in results:
    id, content = row
    if content is None:
        continue
    inputs = tokenizer.encode("summarize: " + content, return_tensors='pt', max_length=1024, truncation=True).to("cuda:0")
    summary_ids = model.generate(inputs, max_length=300, num_beams=10, length_penalty=0.1, no_repeat_ngram_size=10).to(
        "cuda:0")
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


    # 更新数据库中的记录
    update_query1 = "UPDATE shanghaicompany SET summary=%s WHERE id=%s"
    update_query2 = "UPDATE shanghaieconomynewspaper SET summary=%s WHERE id=%s"
    update_query3 = "UPDATE sinaeconomynew SET summary=%s WHERE id=%s"
    update_query4 = "UPDATE news SET summary=%s WHERE id=%s"
    cursor.execute(update_query4, (str(summary), id))
    db.commit()

# 关闭数据库连接
db.close()
