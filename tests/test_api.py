"""Functional tests use temporary data. Synthetic profiles here are test fixtures only."""
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.catalog import Catalog
from app.clustering import ClusterModel, train
from app.main import Settings, create_app
from scripts.evaluate import metrics
from scripts.import_courses import convert

DATA = Path(__file__).resolve().parents[1] / 'data'


@pytest.fixture
def settings(tmp_path):
    return Settings(DATA, tmp_path/'test.sqlite3')


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def register(client, **kwargs):
    result=client.post('/v1/students',json={'selected_topics':['loops','functions'],**kwargs})
    assert result.status_code==201, result.text
    return result.json(), {'Authorization':'Bearer '+result.json()['token']}


def test_seed_and_openapi(client):
    assert client.get('/health').json()['resources']==30
    assert client.get('/health').json()['questions']==12
    assert client.get('/openapi.json').status_code==200
    assert client.get('/docs').status_code==200
    assert len(client.get('/v1/topics').json())==7


def test_questions_hide_answers(client):
    rows=client.get('/v1/quiz/questions?topic=loops').json()
    assert len(rows)==2
    assert all('correct_option' not in q and 'explanation' not in q for q in rows)
    assert client.get('/v1/quiz/questions?topic=not_real').status_code==422


def test_auth_and_isolation(client):
    _, a=register(client)
    _, b=register(client)
    assert client.get('/v1/me').status_code==401
    assert client.get('/v1/me',headers={'Authorization':'Bearer wrong'}).status_code==401
    client.post('/v1/quiz/attempts',headers=a,json={'submission_id':'one','answers':[{'question_id':'loops-1','selected_option':'a'}]})
    assert client.get('/v1/me',headers=a).json()['topic_scores']['loops']['attempted']==1
    assert client.get('/v1/me',headers=b).json()['topic_scores']=={}
    assert 'token_hash' not in client.get('/v1/me',headers=a).text


def test_quiz_idempotency_and_latest_question_counts(client):
    _, headers=register(client)
    body={'submission_id':'run1','answers':[{'question_id':'loops-1','selected_option':'b'}]}
    first=client.post('/v1/quiz/attempts',headers=headers,json=body)
    assert first.status_code==200 and first.json()['correct']==0
    assert client.post('/v1/quiz/attempts',headers=headers,json=body).json()==first.json()
    body['answers'][0]['selected_option']='a'
    assert client.post('/v1/quiz/attempts',headers=headers,json=body).status_code==409
    body['submission_id']='run2'
    assert client.post('/v1/quiz/attempts',headers=headers,json=body).json()['correct']==1
    profile=client.get('/v1/me',headers=headers).json()['topic_scores']['loops']
    assert profile['attempted']==1 and profile['correct']==1
    assert profile['performance_proxy']==pytest.approx(2/3)


def test_invalid_submission_has_no_partial_writes(client):
    _, headers=register(client)
    payload={'submission_id':'bad','answers':[{'question_id':'loops-1','selected_option':'a'},
                                            {'question_id':'missing','selected_option':'a'}]}
    assert client.post('/v1/quiz/attempts',headers=headers,json=payload).status_code==422
    assert client.get('/v1/me',headers=headers).json()['topic_scores']=={}
    payload['answers'][1]=payload['answers'][0]
    assert client.post('/v1/quiz/attempts',headers=headers,json=payload).status_code==422


def test_cold_start_filters_weights_and_explanations(client):
    _, headers=register(client,preferred_format='exercise',target_level='beginner')
    result=client.post('/v1/recommendations',headers=headers,json={'topics':['loops'],'max_minutes':10}).json()
    assert result['effective_algorithm']=='performance'
    assert result['candidate_count']==1
    item=result['recommendations'][0]
    assert item['resource']['resource_id']=='exercise-loop-sum'
    assert sum(item['weights'].values())==pytest.approx(1)
    assert item['score']==pytest.approx(sum(item['components'][k]*v for k,v in item['weights'].items()))
    assert item['diagnostic_evidence']=={}
    assert any('neutral' in r for r in item['reasons'])
    empty=client.post('/v1/recommendations',headers=headers,json={'topics':['loops'],'max_minutes':1}).json()
    assert empty['recommendations']==[]


def test_performance_changes_need_not_content(client):
    _, headers=register(client)
    args={'topics':['loops'],'algorithm':'performance'}
    before=client.post('/v1/recommendations',headers=headers,json=args).json()['recommendations']
    client.post('/v1/quiz/attempts',headers=headers,json={'submission_id':'wrong','answers':[
        {'question_id':'loops-1','selected_option':'b'},{'question_id':'loops-2','selected_option':'a'}]})
    after=client.post('/v1/recommendations',headers=headers,json=args).json()['recommendations']
    by_id={r['resource']['resource_id']:r for r in before}
    for r in after:
        old=by_id[r['resource']['resource_id']]
        assert r['components']['content']==old['components']['content']
        assert r['components']['topic_need']>.5
        assert r['score']>old['score']


def test_content_baseline_ignores_quiz_scores(client):
    _, headers=register(client)
    args={'algorithm':'content'}
    a=client.post('/v1/recommendations',headers=headers,json=args).json()['recommendations']
    client.post('/v1/quiz/attempts',headers=headers,json={'submission_id':'quiz','answers':[{'question_id':'loops-1','selected_option':'a'}]})
    b=client.post('/v1/recommendations',headers=headers,json=args).json()['recommendations']
    assert [(r['resource']['resource_id'],r['score']) for r in a]==[(r['resource']['resource_id'],r['score']) for r in b]
    assert all(r['weights']=={'content':1.} for r in b)


def test_feedback_persistence_and_partial_updates(client):
    _, headers=register(client)
    rid='exercise-loop-sum'
    assert client.post('/v1/feedback',headers=headers,json={'resource_id':rid,'helpful':False}).status_code==200
    update=client.post('/v1/feedback',headers=headers,json={'resource_id':rid,'completed':True}).json()
    assert update['helpful'] is False and update['completed'] is True
    args={'topics':['loops']}
    result=client.post('/v1/recommendations',headers=headers,json=args).json()
    assert rid not in [r['resource']['resource_id'] for r in result['recommendations']]
    args.update(include_completed=True,include_dismissed=True)
    result=client.post('/v1/recommendations',headers=headers,json=args).json()
    assert rid in [r['resource']['resource_id'] for r in result['recommendations']]
    assert client.post('/v1/feedback',headers=headers,json={'resource_id':'missing','helpful':True}).status_code==404


def test_data_survives_restart(settings):
    with TestClient(create_app(settings)) as c:
        user,headers=register(c)
        c.post('/v1/feedback',headers=headers,json={'resource_id':'exercise-loop-sum','completed':True})
    with TestClient(create_app(settings)) as c:
        assert c.get('/v1/me',headers=headers).json()['student_id']==user['student_id']
        result=c.post('/v1/recommendations',headers=headers,json={'topics':['loops']}).json()
        assert 'exercise-loop-sum' not in [x['resource']['resource_id'] for x in result['recommendations']]


@pytest.mark.parametrize('body',[{'k':0},{'k':100},{'max_minutes':-1},{'topics':['missing']},{'topics':[]},{'topics':['loops','loops']},{'student_id':'another'}])
def test_invalid_recommendation_requests(client,body):
    _,headers=register(client)
    assert client.post('/v1/recommendations',headers=headers,json=body).status_code==422


def test_registration_and_update_validation(client):
    assert client.post('/v1/students',json={'selected_topics':['unknown']}).status_code==422
    assert client.post('/v1/students',json={'selected_topics':['loops','loops']}).status_code==422
    _,headers=register(client)
    assert client.put('/v1/me',headers=headers,json={'selected_topics':['lists'],'target_level':'intermediate'}).status_code==200
    assert client.get('/v1/me',headers=headers).json()['selected_topics']==['lists']


def synthetic_profiles():
    # 24 artificial records for algorithm tests ONLY; never shipped as learner observations.
    profiles=[]
    for i in range(24):
        profiles.append({t:{'performance_proxy':value} for t,value in
            zip(['loops','functions'],[.2,.8] if i<12 else [.8,.2])})
    return profiles


def test_cluster_training_serving_and_refusal(tmp_path):
    profiles=synthetic_profiles()
    artifact=train(profiles,['loops','functions'])
    model=ClusterModel(artifact)
    assert model.predict(profiles[0])[0]!=model.predict(profiles[-1])[0]
    assert model.predict({'loops':{'performance_proxy':.2}}) is None
    with pytest.raises(ValueError,match='20'): train(profiles[:4],['loops','functions'])
    with pytest.raises(ValueError,match='non-degenerate'): train([profiles[0]]*24,['loops','functions'])
    path=tmp_path/'clusters.json';path.write_text(json.dumps(artifact))
    assert ClusterModel.load(path).predict(profiles[0])==model.predict(profiles[0])


def test_cluster_api_component(tmp_path):
    artifact=train(synthetic_profiles(),['loops','functions'])
    path=tmp_path/'model.json';path.write_text(json.dumps(artifact))
    with TestClient(create_app(Settings(DATA,tmp_path/'db.sqlite',path))) as c:
        _,headers=register(c)
        c.post('/v1/quiz/attempts',headers=headers,json={'submission_id':'two-topics','answers':[
            {'question_id':'loops-1','selected_option':'b'}, {'question_id':'functions-1','selected_option':'c'}]})
        r=c.post('/v1/recommendations',headers=headers,json={}).json()
        assert r['effective_algorithm']=='cluster'
        assert all('cluster' in item['components'] for item in r['recommendations'])


def test_importer_schema_dedup_and_unknown_level(tmp_path):
    import csv
    source=tmp_path/'courses.csv'
    with source.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['Course Name','Course URL','Course Description','Difficulty Level','Skills'])
        w.writerow(['Python loops','https://example.org/course','Learn Python iteration','Conversant','Python'])
        w.writerow(['Python loops copy','https://example.org/course/','Python loops','Beginner','Python'])
        w.writerow(['French','https://example.org/french','French literature','Beginner','French'])
    audit=convert(source,tmp_path/'out','coursera')
    assert audit['imported']==1 and audit['duplicate_urls']==1 and audit['unknown_difficulties']==1
    catalog=Catalog(tmp_path/'out');r=next(iter(catalog.resources.values()))
    assert r.difficulty is None and r.duration_minutes is None and 'loops' in r.topics
    with pytest.raises(ValueError,match='already exists'): convert(source,tmp_path/'out','coursera')
    source.write_text('wrong,columns\n1,2\n')
    with pytest.raises(ValueError,match='Missing'): convert(source,tmp_path/'bad','coursera')


def test_independent_ranking_metrics():
    labels={'s':{'a':2,'b':1,'c':0}}
    perfect=metrics(labels,{'s':['a','b','c']},2)['macro_average']
    assert perfect=={'precision':1.,'recall':1.,'ndcg':1.}
    worse=metrics(labels,{'s':['c','a']},2)['macro_average']
    assert worse['precision']==.5 and worse['ndcg']<1
    with pytest.raises(ValueError): metrics(labels,{'s':['unjudged']})
    with pytest.raises(ValueError): metrics(labels,{'s':['a','a']})
    assert metrics({'s':{'c':0}},{'s':[]})['macro_average']['ndcg'] is None
