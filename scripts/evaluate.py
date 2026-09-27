"""Evaluate saved rankings against a fully judged, fixed candidate set per student."""
import argparse
import csv
import json
import math
from pathlib import Path


def metrics(judgements, rankings, k=5):
    if k<1: raise ValueError('k must be positive')
    if not judgements or set(judgements)!=set(rankings):
        raise ValueError('Judgements and rankings must contain the same non-empty set of students')
    per_student=[]
    for sid,labels in judgements.items():
        order=rankings[sid]
        if not labels or any(grade not in (0,1,2) for grade in labels.values()):
            raise ValueError('Each student needs a non-empty candidate set with grades 0, 1 or 2')
        if len(set(order))!=len(order) or any(r not in labels for r in order):
            raise ValueError('Ranked resources must be unique and independently judged')
        cutoff=min(k,len(labels))
        top=order[:cutoff]
        gains=[labels[r] for r in top]
        relevant=sum(v>0 for v in labels.values())
        hits=sum(v>0 for v in gains)
        def dcg(vals): return sum((2**v-1)/math.log2(i+2) for i,v in enumerate(vals))
        ideal=dcg(sorted(labels.values(),reverse=True)[:cutoff])
        per_student.append({'student_id':sid,'cutoff':cutoff,'returned':len(top),
            'precision':hits/cutoff,'recall':hits/relevant if relevant else None,
            'ndcg':dcg(gains)/ideal if ideal else None})
    means={name:sum(values)/len(values) if values else None
           for name in ('precision','recall','ndcg')
           for values in [[r[name] for r in per_student if r[name] is not None]]}
    return {'k':k,'students':len(per_student),'macro_average':means,'per_student':per_student,
            'note':'Relevant means grade > 0. No-relevant cases excluded from recall/NDCG averages. Missing recommendations count as misses. Metrics apply only to the declared judged candidate sets.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judgements',type=Path,required=True,help='CSV: student_id,resource_id,relevance')
    parser.add_argument('--rankings',type=Path,required=True,help='JSON: student ID -> ordered resource ID list')
    parser.add_argument('--k',type=int,default=5)
    args=parser.parse_args()
    try:
        judges={}
        with args.judgements.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                labels=judges.setdefault(row['student_id'],{})
                if row['resource_id'] in labels: raise ValueError('Duplicate relevance judgement')
                labels[row['resource_id']]=int(row['relevance'])
        print(json.dumps(metrics(judges,json.loads(args.rankings.read_text()),args.k),indent=2,allow_nan=False))
    except (ValueError,KeyError) as exc: parser.error(str(exc))


if __name__=='__main__': main()
