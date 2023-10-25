# Load model directly
import torch
from scipy.special import softmax
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoConfig
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("D:\postgraduate program\CN_TextSummarization\django_news\ClassifierModel\Model")
model = AutoModelForSequenceClassification.from_pretrained("D:\postgraduate program\CN_TextSummarization\django_news\ClassifierModel\Model").to(device)
# config = AutoConfig.from_pretrained(model)

def classify(text):
    inputs=tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
    classify=model(inputs)
    scores = classify[0][0].cpu().detach().numpy()
    scores = softmax(scores)
    #output=tokenizer.decode(classify)
    return scores

if __name__ == '__main__':
    #普通
    chunk1 = "上证报中国证券网讯（记者 韩宋辉）记者10月9日从信美相互人寿获悉，该公司经过对T（旅居养老）H（居家养老）H（住院养老）康养生态战略一年的探索与实践，形成了“信·美好”康养体系，包括“信·美好旅居”、“信·舒适居家”、“信·心安医养”。目前，信美相互人寿首批旅居养老基地已覆盖海南、浙江、云南、江苏、广西等地，共发团45场，服务客户超过1500人次。,据了解，信美相互人寿于2022年7月在三亚发起旅居养老首发共创团，通过与会员共同探讨、共同创造的形式，了解会员的旅居康养需求，打造美好的养老生活方式。,信美合伙人、会员服务部总经理郑璐介绍，历经一年时间打磨，信美旅居养老产品围绕“医食住行乐学”六个方面进行模块化设计，在舒心的居住环境、便利的配套设施、安全的医疗保障等多维度的准入标准下，保证旅居基地的服务体感。,记者获悉，目前，信美相互人寿旅居养老基地未来三年将努力实现超过30个旅居养老基地合作。"
    #积极
    chunk="沪指收报3233.67点，涨0.15%，成交额3772亿元"
    #消极
    chunk2 = "沪指收报3233.67点，下降0.15%，成交额下降3772亿元"
    inputs=tokenizer.encode(chunk2, return_tensors='pt', max_length=512, truncation=True).to(device)
    classify=model(inputs)
    scores = classify[0][0].cpu().detach().numpy()
    scores = softmax(scores)
    #output=tokenizer.decode(classify)
    #0消极
    #1普通
    #2积极
    print(scores)
    # ranking = np.argsort(scores)
    # ranking = ranking[::-1]
    # for i in range(scores.shape[0]):
    #     l = config.id2label[ranking[i]]
    #     s = scores[ranking[i]]
    #     print(f"{i + 1}) {l} {np.round(float(s), 4)}")

