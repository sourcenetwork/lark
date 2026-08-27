# Perf report

Renders one self-contained HTML page from a schema v1 run file, or from
two of them as a before/after comparison. CI publishes it as an artifact
keyed by the `main` commit it was built from.

```
python3 tools/perf-report/render_html.py \
    --current runs/<sha>.json [--baseline runs/<sha>.json] \
    --out perf-<sha>.html
```

The page is one file: no CDN, no external stylesheet, no fonts fetched
at view time beyond the one Google Fonts link, so an artifact downloaded
from CI months later still renders.

## What it will not do

A metric family the run file marks `absent` renders as an explicit gap,
never as a zero. A run recorded on a contaminated host renders with the
contamination stated on the page rather than dropped. Both rules exist
because a performance page is read as evidence, and a plausible-looking
zero is worse than a blank.
