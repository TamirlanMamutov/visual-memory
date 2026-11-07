"""Select ten local RGB takes and build a small, unlabelled query review page."""
import csv
import json
import os
from pathlib import Path
from experiment_paths import HERE, ROOT as PROJECT_ROOT

ROOT = PROJECT_ROOT / 'Ego-Exo4D'
OUT = HERE / 'output' / 'egoexo-query-review'

def main():
    takes = json.loads((ROOT / 'takes.json').read_text(encoding='utf-8'))
    rows, captures = [], set()
    for category in ['Cooking', 'Bike Repair']:
        candidates = sorted((t for t in takes if t['parent_task_name'] == category
                             and 30 <= t['duration_sec'] <= 180),
                            key=lambda t: (t['duration_sec'], t['take_uid']))
        count = 0
        for t in candidates:
            if t['capture_uid'] in captures:
                continue
            hand = next((ROOT / 'annotations' / 'ego_pose' / split / 'hand' / 'annotation' / (t['take_uid']+'.json')
                         for split in ['train', 'val']
                         if (ROOT / 'annotations' / 'ego_pose' / split / 'hand' / 'annotation' / (t['take_uid']+'.json')).exists()), None)
            if hand is None:
                continue
            streams = [v for cam in t['frame_aligned_videos'].values() for v in cam.values()
                       if v.get('readable_stream_id') == 'rgb']
            paths = [ROOT / t['root_dir'] / 'frame_aligned_videos' / 'downscaled' / '448' / Path(v['relative_path']).name for v in streams]
            video = next((p for p in paths if p.exists()), None)
            if video is None:
                continue
            rows.append(dict(query_id=f'q{len(rows)+1:02}', take_uid=t['take_uid'],
                             capture_uid=t['capture_uid'], take_name=t['take_name'],
                             task=t['task_name'], duration_sec=t['duration_sec'],
                             source_split=hand.parents[2].name, video_path=video.relative_to(PROJECT_ROOT).as_posix(),
                             video_url=Path(os.path.relpath(video, OUT)).as_posix(),
                             hand_annotation=hand.relative_to(PROJECT_ROOT).as_posix(),
                             query_object='', query_time_sec=None, query_box=None,
                             response_start_sec=None, response_end_sec=None,
                             response_track=[], annotation_status='unlabelled'))
            captures.add(t['capture_uid'])
            count += 1
            if count == 5:
                break
    assert len(rows) == 10, f'Only {len(rows)} suitable takes found'
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'queries.json').exists():
        existing = json.loads((OUT / 'queries.json').read_text(encoding='utf-8'))
        if any(r.get('annotation_status') != 'unlabelled' or r.get('query_object') or r.get('query_box') or r.get('response_track') for r in existing):
            raise RuntimeError('Existing query labels found. Preserve them before regenerating the review page.')
    (OUT / 'queries.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    with (OUT / 'takes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        keys = ['query_id','take_name','task','duration_sec','source_split','capture_uid','video_path','hand_annotation']
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    template = Path(__file__).with_name('egoexo_review_template.html').read_text(encoding='utf-8')
    (OUT / 'review.html').write_text(template.replace('__DATA__', json.dumps(rows).replace('</', '<\\/')), encoding='utf-8')
    print(f'Prepared {len(rows)} unique-capture takes: {OUT}')

if __name__ == '__main__':
    main()
