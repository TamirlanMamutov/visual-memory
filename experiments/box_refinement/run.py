"""Refine boxes without additional hints or any change to the baseline window."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from experiments.box_refinement.refinement import refine_boxes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['image','name','description'],default='image');p.add_argument('--margin',type=float,default=.025)
    a=p.parse_args();bundle=json.loads(a.bundle.read_text());predictions=[refine_boxes(q,a.mode,a.margin) for q in bundle['queries']]
    result=dict(method='window_preserving_box_refinement',queries=predictions,cohort=bundle.get('cohort','unspecified'),parameters=dict(mode=a.mode,margin=a.margin),
                frame_ids_preserved=True,AP_confidence='unchanged baseline confidence',additional_hint_used=False)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.out)

if __name__=='__main__':main()
