"""Train only on an explicitly prepared training split, never on test students."""
import argparse
import csv
import json
from pathlib import Path

from app.clustering import train


def load_profiles(path, topics):
    profiles={}
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={'student_id','topic_id','correct','attempted'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV needs {sorted(required)}')
        for row in reader:
            sid,topic=row['student_id'].strip(),row['topic_id'].strip()
            correct,attempted=int(row['correct']),int(row['attempted'])
            if not sid or topic not in topics or attempted<1 or not 0<=correct<=attempted:
                raise ValueError('Invalid student, topic or counts')
            profile=profiles.setdefault(sid,{})
            if topic in profile:
                raise ValueError('Expected one aggregated row per student and topic')
            profile[topic]={'correct':correct,'attempted':attempted,'performance_proxy':(correct+1)/(attempted+2)}
    return profiles


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--topics',type=Path,default=Path('data/topics.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    try:
        topics=[x['topic_id'] for x in json.loads(args.topics.read_text())]
        profiles=load_profiles(args.input,topics)
        profiles={sid:profile for sid,profile in profiles.items()
                  if len(set(profile) & set(topics)) >= 2}
        artifact=train(list(profiles.values()),topics,args.seed)
        artifact['training_student_ids']=sorted(profiles)  # Supports split audits; model file stays offline.
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
        print(json.dumps({key:artifact[key] for key in ['training_students','selected_k','silhouette','candidates']},indent=2))
    except ValueError as exc: parser.error(str(exc))


if __name__=='__main__': main()
