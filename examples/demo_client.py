"""Run against an already-started server: python examples/demo_client.py"""
import json
import urllib.request

BASE='http://127.0.0.1:8000'


def call(path,body=None,token=None):
    headers={'Content-Type':'application/json'}
    if token: headers['Authorization']='Bearer '+token
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body is not None else None,headers=headers)
    with urllib.request.urlopen(req,timeout=20) as response: return json.load(response)


user=call('/v1/students',{'selected_topics':['loops','functions'],'preferred_format':'exercise','target_level':'beginner'})
token=user['token']  # Kept in this process; not printed or saved.
call('/v1/quiz/attempts',{'submission_id':'demo-quiz','answers':[
    {'question_id':'loops-1','selected_option':'b'},
    {'question_id':'loops-2','selected_option':'a'},
    {'question_id':'functions-1','selected_option':'c'}]},token)
result=call('/v1/recommendations',{'topics':['loops'],'max_minutes':15,'k':5},token)
print(json.dumps(result,indent=2))
if result['recommendations']:
    rid=result['recommendations'][0]['resource']['resource_id']
    print(json.dumps(call('/v1/feedback',{'resource_id':rid,'helpful':True,'completed':True},token),indent=2))
