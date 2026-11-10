"""Score frozen trim and expansion settings on cached, source-video-disjoint Ego4D predictions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ego4d", type=Path)
    args = parser.parse_args()
    evaluator = args.ego4d.resolve() / "full_validation"
    batches = evaluator / "batches"
    sys.path.insert(0, str(evaluator))
    np.float, np.int = float, int
    from evaluate_vq import evaluate

    prior = [args.ego4d / name for name in ("vq_val_pilot_51.json", "vq_val_holdout.json")]
    prior += [batches / f"{i:03}" / "annotations.json" for i in range(18)]
    excluded = {v["video_uid"] for path in prior for v in read(path)["videos"]}
    gt_groups, pred_groups = {}, {}
    included = []
    for batch in sorted(batches.iterdir()):
        predictions = batch / "patch/predictions.json"
        if int(batch.name) < 18 or not predictions.is_file():
            continue
        gt, pred = read(batch / "annotations.json"), read(predictions)
        for video, result in zip(gt["videos"], pred["results"]["videos"], strict=True):
            uid = video["video_uid"]
            assert uid == result["video_uid"]
            for clip, output in zip(video["clips"], result["clips"], strict=True):
                assert clip["clip_uid"] == output["clip_uid"]
            if uid not in excluded:
                gt_groups.setdefault(uid, dict(video, clips=[]))["clips"].extend(video["clips"])
                pred_groups.setdefault(uid, dict(result, clips=[]))["clips"].extend(result["clips"])
                if batch.name not in included:
                    included.append(batch.name)
    if not included:
        raise SystemExit("No eligible cached predictions")
    gt["videos"] = list(gt_groups.values())
    pred["results"]["videos"] = list(pred_groups.values())
    print(f"Batches: {', '.join(included)}. Source videos: {len(gt_groups)}.", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        annotations, baseline = directory / "annotations.json", directory / "baseline.json"
        annotations.write_text(json.dumps(gt))
        baseline.write_text(json.dumps(pred))
        print("Raw baseline", flush=True)
        evaluate(str(annotations), str(baseline))
        for scale in (1.15, 1.25):
            output = directory / f"scale-{scale}.json"
            subprocess.run([sys.executable, str(Path(__file__).with_name("postprocess_tracks.py")),
                            str(annotations), str(baseline), str(output),
                            "--max-frames", "20", "--scale", str(scale)], check=True)
            print(f"20 frames, scale {scale}", flush=True)
            evaluate(str(annotations), str(output))
            subprocess.run([sys.executable, str(evaluator / "bootstrap.py"),
                            str(annotations), str(baseline), str(output)], check=True)


if __name__ == "__main__":
    main()
