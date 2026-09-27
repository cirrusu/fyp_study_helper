"""Convert a downloaded Coursera/edX CSV into the API's canonical catalogue."""
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from app.catalog import Catalog
from app.schemas import Resource

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['resource_id','title','description','topics','format','difficulty','duration_minutes',
          'duration_is_estimate','url','content','source']
RULES = {
    'python_basics': r'\bbasics?\b|\bfundamentals?\b|\bbeginners?\b',
    'loops': r'\bloops?\b|\biteration\b|\biterating\b',
    'functions': r'\bfunctions?\b',
    'lists': r'\blists?\b|\bcomprehensions?\b',
    'dictionaries': r'\bdictionaries\b|\bdictionary\b',
    'exceptions': r'\bexceptions?\b|\berror handling\b',
}


def convert(input_path, output_dir, source, include_starter=False):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError('Output directory already exists; choose a new directory to preserve your data')
    mapping = ({'title':'Course Name','url':'Course URL','description':'Course Description',
                'level':'Difficulty Level','extra':'Skills'} if source == 'coursera'
               else {'title':'Name','url':'Link','description':'Course Description',
                     'level':'Difficulty Level','extra':'About'})
    audit = {'source':source,'rows_read':0,'python_matches':0,'duplicate_urls':0,
             'invalid_rows':0,'unknown_difficulties':0,'imported':0,
             'input_sha256':hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
             'note':'Topic tags are keyword heuristics and need manual review. No learner interactions are inferred.'}
    records, seen = [], set()
    with Path(input_path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        required={mapping[x] for x in ('title','url','description','level')}
        missing=required-set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Missing expected columns: {sorted(missing)}; actual: {reader.fieldnames}')
        for row in reader:
            audit['rows_read']+=1
            title=(row[mapping['title']] or '').strip()
            description=(row[mapping['description']] or '').strip()
            extra=(row.get(mapping['extra']) or '').strip()
            text=' '.join([title,description,extra])
            if not re.search(r'\bpython\b',text,re.I):
                continue
            audit['python_matches']+=1
            url=(row[mapping['url']] or '').strip()
            try:
                parts=urlsplit(url)
                url=urlunsplit((parts.scheme.lower(),parts.netloc.lower(),parts.path.rstrip('/'),parts.query,''))
                if url in seen:
                    audit['duplicate_urls']+=1
                    continue
                raw_level=(row[mapping['level']] or '').strip().lower()
                level=raw_level if raw_level in {'beginner','intermediate','advanced'} else None
                topics=['python_general']+[t for t,pattern in RULES.items() if re.search(pattern,text,re.I)]
                resource=Resource(resource_id='course-'+hashlib.sha256(url.encode()).hexdigest()[:16],
                    title=title,description='\n'.join(x for x in [description,extra] if x),topics=topics,
                    format='course',difficulty=level,url=url,
                    source=f'{source} catalogue imported from local CSV; topic tags need review')
                record=resource.model_dump(mode='json')
                record['topics']='|'.join(record['topics'])
                records.append(record);seen.add(url)
                audit['unknown_difficulties']+=int(level is None)
            except ValueError:
                audit['invalid_rows']+=1
    if not records:
        raise ValueError('No valid Python course records found; output was not written')
    audit['imported']=len(records)
    if include_starter:
        with (ROOT/'data/resources.csv').open(encoding='utf-8',newline='') as f:
            records=list(csv.DictReader(f))+records
    output_dir.mkdir(parents=True)
    for name in ('topics.json','questions.json'):
        shutil.copyfile(ROOT/'data'/name,output_dir/name)
    with (output_dir/'resources.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(records)
    Catalog(output_dir)  # Validation applies to the generated canonical data as well.
    (output_dir/'import_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--source',choices=['coursera','edx'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--include-starter',action='store_true')
    args=parser.parse_args()
    try: print(json.dumps(convert(args.input,args.output_dir,args.source,args.include_starter),indent=2))
    except ValueError as exc: parser.error(str(exc))


if __name__=='__main__': main()
