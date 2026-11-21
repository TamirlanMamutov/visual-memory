"""Export the unchanged baseline for a paired hint/refinement evaluation."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.candidate_tools.candidate_common import validate_query,track_prediction

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    bundle=json.loads(a.bundle.read_text());output=[]
    for q in bundle['queries']:validate_query(q);output.append(track_prediction(q,q['baseline_id']))
    result=dict(method='unchanged_baseline',queries=output,cohort=bundle.get('cohort','unspecified'))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.out)

if __name__=='__main__':main()
