# Old online goal-abstraction pilot

The earlier `gcbc`, `actnce_local`, `actnce_multihorizon`, and
`latent_rf_sparse` experiments were **online SGCRL trained from scratch**. They
were not offline learned-phi pretraining and were not learned-goalspace
GSDTRL/PathBridger transfer.

Their preserved success results are, respectively:

- `gcbc`: 0
- `actnce_local`: 0
- `actnce_multihorizon`: 0.02
- `latent_rf_sparse`: 0.02

These results remain historical evidence about that online pilot. They are not
a failure result for the offline learned-goalspace method implemented here.
