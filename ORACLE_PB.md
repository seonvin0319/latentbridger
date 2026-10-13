# Oracle PB

Pinned backbone: official `PathBridger` `main` @ `e592665` (SChoish/PathBridger).

**Oracle PB** = original PathBridger with oracle task projection `φ` on the
transitive scalar value only (not a learned latent encoder).

| Component | PB (`pathbridger_mode=pb`) | Oracle PB (`pathbridger_mode=oracle_pb`) |
| --- | --- | --- |
| Value | `V(s, g)` | `V_φ(φ(s), φ(g))` |
| Candidate score | `V(s,u)·V(u,g)` | `V_φ(φ(s),φ(u))·V_φ(φ(u),φ(g))` |
| Proposer weighting | uses `V` | uses `V_φ` (same wiring) |
| Endpoint proposer | full-state `u ∼ p(u\|s, φ(g))` | unchanged |
| Bridge / IDM / actor | full-state | unchanged |
| Action/chunk critics | full-state | unchanged |

`φ` is the environment oracle goal projection in
`pathbridger_core/utils/goal_representation.py` (buttons / cube poses / maze xy).

## Prior experiments (renamed)

Earlier working names for the same method: LatentBridger / GS-TPB
(`gs_trl_weighted` + `GoalSpaceScalarValue` under `latentbridger-goalspace-ablation`).
This port keeps that definition on the latest PB stack.

## How to run

```bash
# PB (default)
python main.py --agent=configs/pbf/puzzle_3x3.py --seed=0 ...

# Oracle PB
python main.py --agent=configs/oracle_pb/puzzle_3x3.py --seed=0 ...
```

Or set `pathbridger_mode='oracle_pb'` on any `best_config(...)` preset.

## Tests

```bash
python tests/test_oracle_pb.py
python tests/test_oracle_pb_gradients.py
```
