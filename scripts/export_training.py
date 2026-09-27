"""Export topic counts for selected training students from the pilot database."""
import argparse
import csv
import sqlite3
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=Path,required=True)
    parser.add_argument('--student-ids',type=Path,required=True,help='Text file: one training student UUID per line')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.db.is_file(): parser.error('Database does not exist')
    ids=set(args.student_ids.read_text().split())
    if not ids: parser.error('Training ID list is empty')
    rows=[]
    with sqlite3.connect(f'{args.db.resolve().as_uri()}?mode=ro',uri=True) as db:
        for sid in sorted(ids):
            if not db.execute('SELECT 1 FROM students WHERE student_id=?',(sid,)).fetchone():
                parser.error(f'Unknown student ID: {sid}')
            result=db.execute('''SELECT a.topic_id,SUM(a.correct),COUNT(*) FROM answers a
                JOIN (SELECT MAX(answer_id) id FROM answers WHERE student_id=? GROUP BY question_id) z
                ON a.answer_id=z.id GROUP BY a.topic_id''',(sid,)).fetchall()
            rows.extend((sid,*row) for row in result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['student_id','topic_id','correct','attempted']);w.writerows(rows)
    print(f'Exported {len(rows)} topic rows for {len(ids)} selected training students')


if __name__=='__main__': main()
