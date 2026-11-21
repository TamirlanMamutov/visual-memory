# Additional sighting-time hints

This experiment changes **which candidate track is returned** using extra query
information. It may select a different time interval. It does not redraw boxes.

- Real hints: independently supplied approximate time or early/middle/late clue.
- Simulated hints: explicitly GT-derived feasibility inputs, not real-user results.
- [Recorded hint results](RESULTS.md) and [aggregate numbers](results/ren_hint_simulation.json).

First prepare a candidate bundle with the [shared tools](../candidate_tools/README.md).
No model, video or GT file is loaded by the real-hint inference command.

## No-data format check

```text
python experiments/additional_hints/run.py --bundle experiments/candidate_tools/examples/toy_bundle.json --hints experiments/additional_hints/examples/toy_hints.json --out output/hint-demo/predictions.json
```

This invented example selects `later` for `toy_query_a`. It is not a data result.

## Real timing hints

Create a template without reading ground truth:

```text
python experiments/additional_hints/run.py --template --bundle output/eagle-context-pilot/bundle.json --out output/eagle-context-pilot/hints.json
```

Fill every query with either:

```json
{"key":"actual_query_key","kind":"time","seconds":20.0,"sigma_seconds":5.0}
```

or:

```json
{"key":"actual_query_key","kind":"third","value":"late"}
```

Times are **seconds into the benchmark clip**, not a camera clock or frame number.
The 5-second value is a Gaussian standard deviation, not a guaranteed +/-5-second
bound. For `third`, use `early`, `middle`, or `late` relative to searchable history.
Record top-level `provenance` accurately. Real clues should be written independently
of target boxes/timestamps. These are sighting-time clues, not action recognition.

```text
python experiments/additional_hints/run.py --bundle output/eagle-context-pilot/bundle.json --hints output/eagle-context-pilot/hints.json --out output/eagle-context-pilot/hinted.json
```

Hint ranking uses image cosine + `0.05 * temporal_log_density`; AP confidence stays
unchanged. It returns the winning track and a ranked candidate-ID list. A robust
mixture is optional through `--prior-json`, which must contain externally fitted
training-only candidate log densities; never use evaluation GT to construct a
real-user prior. No action parser or speech recognizer is included. A speech
transcript of the same words is an interface test, not extra semantic evidence.

## Oracle-derived timing simulation — keep separate from real-user results

```text
python experiments/additional_hints/simulate.py --bundle output/eagle-context-pilot/bundle.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/simulation
```

This **deliberately derives hints from GT** to test the potential value of timing
information. It is not a real-user evaluation or a benchmark improvement. It tests:

- 2-, 5- and 10-second Gaussian noise;
- 20% wrong hints;
- hints biased 5 seconds earlier;
- synthetic early/middle/late categories;
- 200 independently shuffled source-disjoint controls, random controls, latest-center
  and last-third baselines, and an out-of-fold timing prior.

Five seeds x 40 hint draws = 200 simulations, **not 200 independent videos**.
The marginal timing prior uses only other source folds, but these are still
development analyses. On a validation bundle, fitting a prior on other validation
sources does not make the result an untouched validation score. Use separate
training sources for any deployable prior. `old` tags merely define a reporting
subset; they do not establish independence from pretrained model training.

The robust mixture is fixed at 80% hint density + 20% prior density. No parameter
sweep is performed. Full official metric calls check the cached-statistic fast
path on 12 fixed scenario/cohort combinations. To re-score frozen simulation
rankings without regenerating hints, repeat the command with `--evaluate-only`.


## Score real-hint predictions yourself

```text
python experiments/candidate_tools/evaluate.py --bundle output/eagle-context-pilot/bundle.json --predictions output/eagle-context-pilot/hinted.json --truth output/eagle-context-pilot/ground_truth.json --evaluator-root ../VQLoC --out output/eagle-context-pilot/hint_metrics.json
```

Compare with the exact same bundle's unchanged baseline. Keep simulated-hint
results separate from ordinary image-query and real-user tables. After a code
upgrade, use a fresh simulation output folder; existing frozen hashes are retained.
