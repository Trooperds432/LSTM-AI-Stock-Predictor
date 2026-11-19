# LSEG recovery and membership reconciliation

The four failed batch requests covered 20 RICs. Individual history requests
were started in the `indicators_data/lseg/recovery` directory. The original
snapshot and its manifest remain the record of the initial requests.

The nine reference batches returned an ISIN for all 857 requested instruments.
The response did not include either requested PermID field. ISIN is useful for
checking candidate aliases, but does not by itself prove historical identity
continuity through reorganizations.

Replay of daily joiners and leavers between the dated snapshots found:

| Checkpoint | Extra RICs | Missing RICs | Identity evidence |
| --- | --- | --- | --- |
| 2019-12-31 | PX.N^J18 | LIN.OQ | Different ISINs. Requires corporate-action and membership evidence. |
| 2023-12-29 | BKR.N, HST.N | BKR.OQ | BKR aliases share an ISIN. HST remains unexplained. |
| 2026-11-18 | CPB.N, KMB.N, KVUE.N^K26 | KMB.OQ | KMB aliases share an ISIN. The feed records CPB.OQ as a leaver. The CPB identity link and KVUE removal date require verification. |

Reconciliation starts from the previous dated snapshot for each interval.
The first 2011 request returned no constituent RICs. No manual membership
changes have been inferred from these endpoint differences.
