"""Check local Ego-Exo4D assets; no downloads or model inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
from experiment_paths import HERE, ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-json', type=int, default=10, help='Number of annotation files to parse (default: 10)')
    args = parser.parse_args()
    root = ROOT / 'Ego-Exo4D'
    print(f'Dataset: {root}', flush=True)
    takes = json.loads((root / 'takes.json').read_text(encoding='utf-8'))
    videos = sorted((root / 'takes').rglob('*.mp4'))
    if not videos:
        raise RuntimeError('No local MP4 videos found. Make the dataset available locally first.')
    print(f'Found {len(videos):,} MP4 streams; checking file sizes...', flush=True)
    empty = [str(p.relative_to(root)) for p in videos if p.stat().st_size == 0]
    if empty:
        raise RuntimeError(f'Empty video files: {empty[:10]}')
    local_names = {p.relative_to(root / 'takes').parts[0] for p in videos}
    selected = [t for t in takes if t['take_name'] in local_names]
    paths, missing = [], []
    annotation_index = {}
    for p in sorted((root / 'annotations' / 'ego_pose').rglob('*.json')):
        annotation_index.setdefault(p.stem, []).append(p)
    coverage = Counter()
    for t in selected:
        uid_paths = annotation_index.get(t['take_uid'], [])
        if not uid_paths:
            missing.append(t['take_uid'])
        for p in uid_paths:
            rel = p.relative_to(root / 'annotations' / 'ego_pose')
            kind = 'camera' if 'camera_pose' in rel.parts else ('hand' if 'hand' in rel.parts else 'body')
            coverage[f'{rel.parts[0]}/{kind}'] += 1
        paths.extend(uid_paths)
    checked = 0
    for p in paths[:max(0, args.sample_json)]:
        json.loads(p.read_text(encoding='utf-8'))
        checked += 1
    result = dict(status='passed' if not missing else 'incomplete',
                  takes=len(selected), mp4_streams=len(videos),
                  video_bytes=sum(p.stat().st_size for p in videos),
                  takes_missing_pose_annotations=missing,
                  annotation_coverage=dict(sorted(coverage.items())),
                  annotation_json_files_parsed=checked,
                  scope='File availability, nonzero size and sampled JSON syntax; not video decoding or integrity verification.',
                  localization='Not evaluated. Object query boxes/tracks and model inference are still required.')
    out = HERE / 'output' / ('egoexo-check-' + time.strftime('%Y%m%d-%H%M%S'))
    out.mkdir(parents=True)
    (out / 'dataset-check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    print(f'Results: {out}', flush=True)
    if missing:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
