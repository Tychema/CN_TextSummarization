import json

from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from datetime import datetime
import sys
sys.path.append('..')
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

    # if request.method == 'POST':
    #     ##判断POST请求body是否为空
    #     if request.body.decode() == '':
    #         return JsonResponse({
    #             'status': 'Error',
    #             'date':datetime.now(),
    #             'txt' :'',
    #             'SummarizeText':'',
    #             'msg':'body is Null!'
    #             })
    #     ##不为空就将body转换成字典
    #     else:
    #         body = eval(request.body)
    #     ##确保字段不为空
    #     if body['txt'] == '':
    #         return JsonResponse({
    #             'status': 'Error',
    #             'date': datetime.now(),
    #             'txt': '',
    #             'SummarizeText': '',
    #             'msg': 'please check body!'
    #         })
    #     else:
    #         txt=body['txt']
    #         SummarizationTxt = model_summary(txt)
    #         return JsonResponse({
    #             'status': 'Success',
    #             'date': datetime.now(),
    #             'txt': txt,
    #             'SummarizeText': SummarizationTxt,
    #             'msg': please check body!'
    #         })
    # else:
    #     return JsonResponse({
    #         'status': 'Error',
    #         'date': datetime.now(),
    #         'txt': '',
    #         'SummarizeText': '',
    #         'msg': 'request method not is POST!'
    #     })
