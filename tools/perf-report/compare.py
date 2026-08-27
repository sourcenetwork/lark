#!/usr/bin/env python3
"""Compare two schema v1 run files and decide what actually regressed.

The hard part is not computing a percentage, it is refusing to report
one that means nothing. A run recorded while the host was busy can shift
a figure 25% with no code change, so a comparison that treats every
delta as signal produces alerts nobody trusts, and an alert nobody
trusts is worse than none.

Three rules keep a verdict honest:

* **Both sides must be trusted.** If either run marked a family
  ``contaminated`` or ``absent``, the comparison is reported as
  ``unverified`` and never as a pass or a regression.
* **Ranges must separate.** Each point carries the ``min`` and ``max``
  seen across repetitions. A delta counts only when the two ranges do
  not overlap: if the new ``max`` still reaches the old ``min``, the
  runs did not measure a difference, whatever their medians say.
* **A floor on top of that.** Non-overlapping ranges can still be a
  fraction of a percent apart. A change under ``--threshold`` is
  reported as noise regardless.
"""

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass, field

# Higher is better for these; lower is better for the rest.
HIGHER_IS_BETTER = {"ops", "ops_per_s", "thrpt"}

VERDICT_PASS = "pass"
VERDICT_REGRESSED = "regressed"
VERDICT_IMPROVED = "improved"
VERDICT_NOISE = "noise"
VERDICT_UNVERIFIED = "unverified"


@dataclass
class Delta:
    """One metric compared across two runs."""

    family: str
    name: str
    unit: str
    before: float | None
    after: float | None
    verdict: str
    pct: float | None = None
    reason: str = ""
    higher_is_better: bool = True
    # The measured ranges, so a reader can see why a verdict was reached.
    before_range: tuple | None = None
    after_range: tuple | None = None

    @property
    def significant(self):
        return self.verdict in (VERDICT_REGRESSED, VERDICT_IMPROVED)


@dataclass
class Report:
    deltas: list = field(default_factory=list)
    regressions: list = field(default_factory=list)
    improvements: list = field(default_factory=list)
    unverified: list = field(default_factory=list)

    @property
    def verdict(self):
        if self.regressions:
            return VERDICT_REGRESSED
        if not self.deltas or all(d.verdict == VERDICT_UNVERIFIED for d in self.deltas):
            return VERDICT_UNVERIFIED
        return VERDICT_PASS


def _trusted(trust):
    return trust in (None, "clean")


def _classify(before, after, brange, arange, higher_is_better, threshold, reason):
    """Verdict for one pair of measurements."""
    if reason:
        return VERDICT_UNVERIFIED, None, reason
    if before in (None, 0) or after is None:
        return VERDICT_UNVERIFIED, None, "no comparable measurement on both sides"

    pct = (after - before) / before * 100.0
    if not higher_is_better:
        pct = -pct

    if abs(pct) < threshold:
        return VERDICT_NOISE, pct, f"within the {threshold:g}% threshold"

    # Ranges must separate before a delta is called real.
    if brange and arange:
        blo, bhi = brange
        alo, ahi = arange
        if not (ahi < blo or alo > bhi):
            return (VERDICT_NOISE, pct,
                    "the two runs' measured ranges overlap, so they did not "
                    "measure a difference")

    return (VERDICT_REGRESSED if pct < 0 else VERDICT_IMPROVED), pct, ""


def compare(base, cur, threshold=3.0):
    rep = Report()

    bt = base.get("metrics", {}).get("throughput", {})
    ct = cur.get("metrics", {}).get("throughput", {})
    unit = ct.get("unit", "ops/s")
    for name, cw in ct.get("workloads", {}).items():
        bw = bt.get("workloads", {}).get(name, {})
        cpts = {p["threads"]: p for p in cw.get("points", [])}
        bpts = {p["threads"]: p for p in bw.get("points", [])}
        for threads in sorted(set(cpts) | set(bpts)):
            b, c = bpts.get(threads), cpts.get(threads)
            reason = ""
            if not _trusted(bw.get("trust")):
                reason = f"baseline {name} is {bw.get('trust')}"
            elif not _trusted(cw.get("trust")):
                reason = f"this run's {name} is {cw.get('trust')}"
            elif b is None or c is None:
                reason = "only one run measured this thread count"
            verdict, pct, why = _classify(
                b.get("ops") if b else None,
                c.get("ops") if c else None,
                (b.get("min"), b.get("max")) if b and b.get("min") is not None else None,
                (c.get("min"), c.get("max")) if c and c.get("min") is not None else None,
                True, threshold, reason)
            rep.deltas.append(Delta(
                family="throughput", name=f"{name} @ {threads}t", unit=unit,
                before=b.get("ops") if b else None,
                after=c.get("ops") if c else None,
                verdict=verdict, pct=pct, reason=why or reason,
                before_range=(b.get("min"), b.get("max")) if b else None,
                after_range=(c.get("min"), c.get("max")) if c else None))

    bs = base.get("metrics", {}).get("binary_size", {})
    cs = cur.get("metrics", {}).get("binary_size", {})
    bmap = {r.get("artifact"): r.get("lark_cost_kib") for r in bs.get("rows", [])}
    for r in cs.get("rows", []):
        art = r.get("artifact")
        reason = "" if _trusted(cs.get("trust")) and _trusted(bs.get("trust")) else "size not trusted in one run"
        # Binary size is deterministic, so no range test applies and any
        # movement past the threshold is real.
        verdict, pct, why = _classify(
            bmap.get(art), r.get("lark_cost_kib"), None, None, False, threshold, reason)
        rep.deltas.append(Delta(
            family="binary_size", name=f"{art} binary", unit="KiB",
            before=bmap.get(art), after=r.get("lark_cost_kib"),
            verdict=verdict, pct=pct, reason=why or reason, higher_is_better=False))

    br = base.get("metrics", {}).get("rss", {})
    cr = cur.get("metrics", {}).get("rss", {})
    bmap = {s.get("variant"): s.get("rss_peak_mib") for s in br.get("soaks", [])}
    for s in cr.get("soaks", []):
        reason = "" if _trusted(cr.get("trust")) and _trusted(br.get("trust")) else "rss not trusted in one run"
        verdict, pct, why = _classify(
            bmap.get(s.get("variant")), s.get("rss_peak_mib"),
            None, None, False, threshold, reason)
        rep.deltas.append(Delta(
            family="rss", name=f'peak RSS, {s.get("label", s.get("variant", ""))}',
            unit="MiB", before=bmap.get(s.get("variant")), after=s.get("rss_peak_mib"),
            verdict=verdict, pct=pct, reason=why or reason, higher_is_better=False))

    rep.regressions = [d for d in rep.deltas if d.verdict == VERDICT_REGRESSED]
    rep.improvements = [d for d in rep.deltas if d.verdict == VERDICT_IMPROVED]
    rep.unverified = [d for d in rep.deltas if d.verdict == VERDICT_UNVERIFIED]
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--current", required=True)
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="percent change below which a delta is noise (default 3)")
    ap.add_argument("--fail-on-regression", action="store_true")
    args = ap.parse_args()

    base = json.loads(pathlib.Path(args.baseline).read_text())
    cur = json.loads(pathlib.Path(args.current).read_text())
    rep = compare(base, cur, args.threshold)

    for d in rep.regressions:
        print(f"REGRESSED  {d.name:32} {d.before:>12,.0f} -> {d.after:>12,.0f} "
              f"{d.unit}  {d.pct:+.1f}%")
    for d in rep.improvements:
        print(f"improved   {d.name:32} {d.before:>12,.0f} -> {d.after:>12,.0f} "
              f"{d.unit}  {d.pct:+.1f}%")
    if rep.unverified:
        print(f"unverified {len(rep.unverified)} metric(s): not compared, see the report")
    print(f"verdict: {rep.verdict}")

    if args.fail_on_regression and rep.regressions:
        sys.exit(1)


if __name__ == "__main__":
    main()
