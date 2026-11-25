"""Compare portable CV choices/metrics with the original recorded studies."""
from pathlib import Path
import argparse,json
import numpy as np
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--expected',type=Path,required=True);p.add_argument('--group',choices=['hand','v10','v11'],required=True);p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    expected=json.loads(a.expected.read_text())[a.group];checks=[]
    for name,old in expected.items():
        marker=a.results/name/'COMPLETE.json'
        if not marker.exists():continue
        got=json.loads(marker.read_text());z=dict(np.load(marker.parent/'choices.npz'))
        assert z['keys'].tolist()==old['keys'];assert z['choices'].tolist()==old['choices'],name+' candidate selections differ'
        for cohort,original in [('all','all60'),('new','new40')]:
            for field in ['stAP25','tAP25','Recovery','Success','mean_track_IoU']:
                assert abs(got['metrics'][cohort][field]-old['metrics'][original][field])<1e-10,(name,cohort,field)
        checks.append(name)
    if not checks:raise ValueError('No completed recipes to compare')
    d=dict(passed=True,matched_recipes=checks,all_candidate_choices_exact=True,metrics_tolerance=1e-10,group=a.group)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
if __name__=='__main__':main()
