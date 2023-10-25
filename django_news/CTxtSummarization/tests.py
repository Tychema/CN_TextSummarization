from django.test import TestCase
import json

from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from datetime import datetime
import sys
sys.path.append('..')
from django_news.Model.TextSummarization import model_summary
from django_news.ClassifierModel.financialNewSentimentClassifier import classify
import numpy as np
res =  '{\'status\': {},\'date\':{},\'txt\' :{},\'SummarizeText\':{},\'msg\':{}}'
# Create your tests here.
if __name__ == '__main__':
    # txt = "2021年1月，CCF决定启动新一轮中国计算机学会推荐国际学术会议和期刊目录（以下简称《目录》）调整工作并委托CCF学术工作委员会组织实施。经过前期的充分讨论和论证后，于2021年9月开始正式向各专委会征集调整建议。期间由于疫情反复，最终目录调整的评审会议于2022年10月在北京召开，会议以线上线下相结合的方式，顺利完成本次目录的修订工作。本次目录调整的总体原则是：在既有基础上进行微调，保持宁缺毋滥的原则；领域划分方式保持不变，期刊和会议的推荐类别体系保持不变，在同等情况下增加对国内期刊的支持。本次目录调整工作分为三个阶段完成：提议受理阶段，领域责任专家审议和初审推荐阶段，以及终审核准阶段。根据CCF的授权和工作安排，整个调整工作由CCF学术工作委员会主持并组织CCF相关领域的专家完成。同时，CCF学术工作委员会还负责为初审推荐阶段收集、整理和提供所需要的期刊会议相关数据以及国际上同行的观点与看法并制作了在线查询系统，以及制作了终审会议的详细材料。中国计算机学会推荐国际学术会议和期刊目录"
    # SummarizationTxt = model_summary(txt)
    # res1=res.format('200',str(datetime.now()),txt,SummarizationTxt,'Success')
    # print(res1)
    # print(type(res1))
    # res1=json.loads(res1)
    # print(res1)
    # print(type(res1))
    chunk1 = "上证报中国证券网讯（记者 韩宋辉）记者10月9日从信美相互人寿获悉，该公司经过对T（旅居养老）H（居家养老）H（住院养老）康养生态战略一年的探索与实践，形成了“信·美好”康养体系，包括“信·美好旅居”、“信·舒适居家”、“信·心安医养”。目前，信美相互人寿首批旅居养老基地已覆盖海南、浙江、云南、江苏、广西等地，共发团45场，服务客户超过1500人次。,据了解，信美相互人寿于2022年7月在三亚发起旅居养老首发共创团，通过与会员共同探讨、共同创造的形式，了解会员的旅居康养需求，打造美好的养老生活方式。,信美合伙人、会员服务部总经理郑璐介绍，历经一年时间打磨，信美旅居养老产品围绕“医食住行乐学”六个方面进行模块化设计，在舒心的居住环境、便利的配套设施、安全的医疗保障等多维度的准入标准下，保证旅居基地的服务体感。,记者获悉，目前，信美相互人寿旅居养老基地未来三年将努力实现超过30个旅居养老基地合作。"
    chunk="沪指收报3233.67点，涨0.15%，成交额3772亿元"
    scores=classify(chunk)
    sentiment = np.argmax(scores);
    #output=tokenizer.decode(classify)
    print(scores)
    print(sentiment)
