from django.test import TestCase
import json

from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from datetime import datetime
import sys
sys.path.append('..')
from Model.TextSummarization import model_summary
res =  '{\'status\': {},\'date\':{},\'txt\' :{},\'SummarizeText\':{},\'msg\':{}}'
# Create your tests here.
if __name__ == '__main__':
    txt = "2021年1月，CCF决定启动新一轮中国计算机学会推荐国际学术会议和期刊目录（以下简称《目录》）调整工作并委托CCF学术工作委员会组织实施。经过前期的充分讨论和论证后，于2021年9月开始正式向各专委会征集调整建议。期间由于疫情反复，最终目录调整的评审会议于2022年10月在北京召开，会议以线上线下相结合的方式，顺利完成本次目录的修订工作。本次目录调整的总体原则是：在既有基础上进行微调，保持宁缺毋滥的原则；领域划分方式保持不变，期刊和会议的推荐类别体系保持不变，在同等情况下增加对国内期刊的支持。本次目录调整工作分为三个阶段完成：提议受理阶段，领域责任专家审议和初审推荐阶段，以及终审核准阶段。根据CCF的授权和工作安排，整个调整工作由CCF学术工作委员会主持并组织CCF相关领域的专家完成。同时，CCF学术工作委员会还负责为初审推荐阶段收集、整理和提供所需要的期刊会议相关数据以及国际上同行的观点与看法并制作了在线查询系统，以及制作了终审会议的详细材料。中国计算机学会推荐国际学术会议和期刊目录"
    SummarizationTxt = model_summary(txt)
    res1=res.format('200',str(datetime.now()),txt,SummarizationTxt,'Success')
    print(res1)
    print(type(res1))
    res1=json.loads(res1)
    print(res1)
    print(type(res1))
