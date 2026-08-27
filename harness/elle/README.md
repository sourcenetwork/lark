# Elle consistency harness

Drives concurrent transactions against lark, records a Jepsen-format
history, and hands it to [Elle](https://github.com/jepsen-io/elle) for a
transactional-safety verdict.

This crate lives outside the `lark-kv` workspace, exactly like `fuzz/`.
It declares its own `[workspace]` table, so `cargo check --workspace` at
the repository root never builds it and the MSRV job is unaffected.

```sh
cd harness/elle
cargo build --release
./target/release/elle-gen --model list-append --out history.json --dir db
java -jar elle-cli.jar --model list-append history.json
```

## Getting elle-cli

`elle-cli` is a standalone jar from
[github.com/ligurio/elle-cli](https://github.com/ligurio/elle-cli). Build
it from source with [Leiningen](https://leiningen.org/):

```sh
git clone https://github.com/ligurio/elle-cli
cd elle-cli
lein deps
lein uberjar
cp target/elle-cli-*-standalone.jar /path/to/lark/harness/elle/elle-cli.jar
```

Or take the prebuilt jar from a release, which needs no Clojure
toolchain:

```sh
curl -sSL -o elle-cli-bin.zip \
  https://github.com/ligurio/elle-cli/releases/download/0.1.9/elle-cli-bin-0.1.9.zip
unzip -j elle-cli-bin.zip 'target/*.jar'
mv elle-cli-0.1.9-standalone.jar elle-cli.jar
```

The jar and every generated history are gitignored.

## Checking a history

```sh
java -jar elle-cli.jar --model list-append history.json
java -jar elle-cli.jar --model rw-register history.json
```

`true` means the history was valid, `false` means Elle found an anomaly.
Add `--verbose` for the anomaly list and the cycles behind it, and
`--consistency-models <model>` to check against a specific level rather
than the default `strict-serializable`:

```sh
java -jar elle-cli.jar --model list-append \
  --consistency-models read-committed --verbose history.json
```

### History format

The generator writes two files. `history.json` is a JSON array with one
operation object per line; `history.jsonl` is the same records as bare
newline-delimited JSON. Both hold the same operations in the same order.
Use `history.json` with elle-cli: version 0.1.9 parses a single JSON
array or an EDN stream, and rejects bare newline-delimited JSON with
`IllegalArgumentException: Key must be integer`.

Each record carries the Jepsen fields:

```json
{"type":"invoke","f":"txn","process":3,"time":62135091,"index":1,"value":[["r",0,null],["append",0,1]]}
{"type":"ok","f":"txn","process":3,"time":62135127,"index":2,"value":[["r",0,null],["append",0,1]]}
```

`type` is `invoke`, `ok`, `fail` or `info`. The fourth type is not
optional: an operation that was in flight when its process was killed is
genuinely indeterminate, and recording it as `fail` would tell the
checker a write definitely did not happen when it may well have.
`info` is also used for operations whose write-ahead-log region the
harness is about to damage on purpose.

## Models

`--model list-append` builds each key as a list and performs appends as
read-modify-write through `Transaction::get_for_update`, which is the
API a caller reaches for precisely to make an increment safe. Append
values are globally unique, as Elle requires to reconstruct version
order.

`--model rw-register` performs blind writes of globally unique values
and reads, `[["w",k,v],["r",k,v]]`.

List-append is the sharper of the two: it reconstructs the version order
of every key from the observed lists, so a clobbered update is visible
directly rather than only through a dependency cycle.

## Isolation levels

`--isolation read-committed|repeatable-read|serializable` selects the
level to exercise. lark today exposes snapshot isolation only, through
two transaction flavors, so not every level is reachable:

| Requested | Runs against | Reachable today? |
| --- | --- | --- |
| `read-committed` | `TransactionDb` (pessimistic locks) | Yes, as a sound over-approximation. Snapshot isolation is strictly stronger than read-committed, so every anomaly the checker reports at `--consistency-models read-committed` is a genuine violation. |
| `repeatable-read` | `OptimisticTransactionDb` | No. Snapshot isolation is incomparable with repeatable-read: it permits write skew (`G2-item`), which repeatable-read forbids. A `G2-item` verdict here is legal behavior, not a lark bug. |
| `serializable` | `OptimisticTransactionDb` | No. Snapshot isolation is strictly weaker than serializable, for the same reason. |

The two unreachable levels still run, so the flag is exercisable and
will keep working once the engine grows the levels, but the generator
prints an unmissable warning naming the gap. To get a verdict that means
something about lark today, check the optimistic history against the
level lark actually claims:

```sh
java -jar elle-cli.jar --model list-append \
  --consistency-models snapshot-isolation history.json
```

## Fault injection

`--faults kill,torn-write,truncate-wal` (or `--faults all`) puts recovery
into the history rather than only steady-state concurrency. Faults are
applied in sequence to one database, because wiping it mid-run would
reset every key and inject anomalies that are artifacts of the harness.

- `kill` re-executes the generator as a child process, waits for it to
  get transactions in flight, and SIGKILLs it. Its completed operations
  stay `ok`: child processes run with `DurabilityMode::Immediate`, so a
  commit that returned is already fsynced. The one operation it had in
  flight becomes `info`.
- `torn-write` runs a child that exits without closing the database, so
  its writes live only in the write-ahead log, then overwrites the final
  bytes of that log with a pattern no checksum accepts.
- `truncate-wal` does the same, then truncates the log inside the same
  region, leaving a partial record at the tail.

Both log faults record a high-water mark immediately after the child
opens the database and damage only bytes written above it. Everything
the history reports as committed lives below the mark, so a checker
failure after one of these faults is a real durability bug and never an
artifact. If the log rotated or did not grow, the fault is skipped and
said so rather than damaging committed data.

## Built-in checker

Every run ends with a built-in witness detector, so the harness produces
evidence on a machine with no JVM and so a failure points at a concrete
pair of operations rather than a cycle diagram. It reports lost appends
for `list-append` and stale reads for `rw-register`, both stated against
real time and therefore sound under strict serializability. Run it alone
on an existing history with `--verify history.json`. elle-cli remains
the authority; the built-in check is a witness, not a substitute.

`elle-gen` exits non-zero when it finds witnesses.

## Status on this tree

Measured on the full stack with `just elle-matrix`. Every level lark
claims is green:

| Workload | Engine | Checked against | Verdict |
| --- | --- | --- | --- |
| `list-append`, `--isolation repeatable-read` | `OptimisticTransactionDb` | `snapshot-isolation` | `true` |
| `rw-register`, `--isolation repeatable-read` | `OptimisticTransactionDb` | `snapshot-isolation` | `true` |
| `list-append`, `--isolation read-committed` | `TransactionDb` | `read-committed` | `true` |
| `list-append`, `--keys 1` (one hot key) | `TransactionDb` | `read-committed` | `true` |

The last two rows were `false` when PR 1 landed: `get_for_update` took
the key lock and then read at the sequence the transaction began with,
so two transactions serialized on the lock and both still read the
pre-lock value, and the second one's read-modify-write clobbered the
first. Elle called it `incompatible-order` and `G0`. The pessimistic
commit path now validates what it read before it applies, and both rows
pass.

### What a green matrix does and does not say

lark provides **snapshot isolation**, not serializability. Checking an
SI history against `strict-serializable` returns `false`, and that is
correct rather than a defect: SI permits write skew (G2 anti-dependency
cycles) by definition. That check is kept as the matrix's calibration -
if it ever returned `true`, the checker would be accepting everything
and a green matrix would mean nothing:

```sh
java -jar elle-cli.jar --model list-append \
  --consistency-models strict-serializable /tmp/elle-optimistic-si.json
# => false, with G2-item / G-nonadjacent-item witnesses
```

A history with no committed transactions is also refused by the
generator rather than reported as passing, for the same reason: an empty
dependency graph has no cycles, so it checks out while proving nothing.


## Options

```text
--model <list-append|rw-register>   Workload model (default: list-append)
--isolation <read-committed|repeatable-read|serializable>
--faults <kill,torn-write,truncate-wal|all>
--dir <path>                        Database directory (default: db)
--out <path>                        History output (default: history.json)
--threads <n>                       Concurrent processes (default: 8)
--txns <n>                          Transactions per process (default: 50)
--keys <n>                          Distinct keys (default: 4)
--seed <n>                          Deterministic workload seed
--verify <path>                     Analyze an existing history and exit
```
