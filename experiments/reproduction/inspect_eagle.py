"""Inventory exact-replay artifacts for the teammate's zr EAGLE runs; load no models."""
from pathlib import Path
import argparse,json
RUNS=['eagle-20261111-221053','eagle-20261114-002928','eagle-20261114-204835']
def inventory(project,repo):
    project=Path(project);repo=Path(repo);rows=[]
    for run in RUNS:
        evaluations=repo/'output'/run/'eval_parts';checkpoint=project/'cs230-cache/eagle/checkpoints'/run/'last.pt'
        rows.append(dict(run=run,evaluation_folder=str(evaluations),evaluation_present=evaluations.is_dir() and any(evaluations.glob('*.json')),checkpoint=str(checkpoint),checkpoint_present=checkpoint.is_file()))
    annotation=project/'Ego4D/v2/annotations/vq_val.json';clips=project/'Ego4D/v2/clips'
    return dict(runs=rows,annotations_present=annotation.is_file(),clips_folder_present=clips.is_dir(),
        saved_trace_replay_ready=all(r['evaluation_present'] for r in rows) and annotation.exists(),
        checkpoint_replay_ready=all(r['checkpoint_present'] for r in rows) and annotation.exists() and clips.is_dir(),
        note='Saved evaluations reproduce the ensemble on CPU. SAM2 replay additionally needs licensed clips and SAM2 weights; saved SAM2 per-clip files allow rescoring without propagation. New training is not bit-identical checkpoint reproduction.')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',type=Path,required=True);p.add_argument('--repo-root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--out',type=Path)
    a=p.parse_args();d=inventory(a.project_root,a.repo_root)
    if a.out:a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(d,indent=2)+'\n')
    print(json.dumps(d,indent=2))
if __name__=='__main__':main()
