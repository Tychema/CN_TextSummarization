import json

from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from datetime import datetime
import sys
sys.path.append("..")
import numpy as np
from Model.TextSummarization import model_summary
res =  '{\'status\': {},\'date\':{},\'txt\' :{},\'SummarizeText\':{},\'msg\':{}}'

def TextSummarization(request):

    # logger = logging.getLogger(__name__)
    # if request.method == 'POST':
    #     txt = request.POST.get('txt', '')
    #     textSummarization = model_summary(txt)
    # else:
    #     return JsonResponse({'code': 100103, 'msg': '请求方法错误'})
    # return JsonResponse({'code': 200, 'request':request,'txt':txt,'SummarizeText': textSummarization})
    if request.method == 'POST':
        body = eval(request.body)
        txt = body.get('txt', '')
        if txt:
            SummarizationTxt = model_summary(txt)
            response_data = {
                'status': 200,
                'date': str(datetime.now()),
                'txt': txt,
                'SummarizeText': SummarizationTxt,
                'msg': 'Success',
            }
            return JsonResponse(response_data)
        else:
            return JsonResponse({'status': 400, 'date': str(datetime.now()), 'msg': 'Please check your body!'})

    return JsonResponse({'status': 400, 'date': str(datetime.now()), 'msg': 'Invalid request method!'})

def TextClassify(request):
    import sys
    sys.path.append("..")
    from ClassifierModel.financialNewSentimentClassifier import classify
    body = eval(request.body)
    txt = body.get('txt', '')
    if txt:
        classifyScore = classify(txt)
        sentiment=np.argmax(classifyScore);
        response_data = {
            'status': 200,
            'date': str(datetime.now()),
            'txt': txt,
            'ClassifyScores': str(classifyScore),
            'sentiment': str(sentiment),
            'msg': 'Success',
        }
        return JsonResponse(response_data)
    else:
        return JsonResponse({'status': 400, 'date': str(datetime.now()), 'msg': 'Please check your body!'})


    return JsonResponse({'status': 400, 'date': str(datetime.now()), 'msg': 'Invalid request method!'})