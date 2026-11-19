"""Generate the provisional daily membership panel from the audited LSEG snapshot."""
import json
from pathlib import Path

import exchange_calendars as xc
import pandas as pd


def main():
    root = Path(__file__).parent / 'indicators_data/lseg'
    snapshot = root / '20261119T132813190380Z'
    reference = pd.concat([pd.read_csv(f) for f in snapshot.glob('identifiers_*.csv')])
    identities = reference.set_index('Instrument').ISIN.to_dict()
    bemis = 'US0814371052'
    amcor = identities['AMCR.N']
    historical_names = {
        ('DD.N', 'DOW CHEMICAL ORD'): 'DOW.N^I17',
        ('DD.N', 'EI D P DE NEMOUR ORD'): 'DD.N^I17',
        ('FTI.N', 'FMC TECHNOLOGIES ORD'): 'FTI.N^A17',
        ('CB.N', 'CHUBB ORD'): 'CB.N^A16',
        ('JCI.N', 'JOHNSON CONTROLS ORD'): 'JCI.N^I16',
    }
    checkpoints = {}
    for path in sorted(snapshot.glob('members_*.csv')):
        rows = pd.read_csv(path).dropna(subset=['Constituent RIC'])
        if rows.empty:
            continue
        # Before the 2020 exchange transfer, two snapshots list both HST aliases.
        if path.stem[8:] < '2020-11-02' and {'HST.N', 'HST.OQ'} <= set(rows['Constituent RIC']):
            assert identities['HST.N'] == identities['HST.OQ']
            rows = rows.loc[rows['Constituent RIC'].ne('HST.OQ')]
        members = [identities[historical_names.get((ric, name), ric)]
                   for ric, name in zip(rows['Constituent RIC'], rows['Constituent Name'])]
        if len(set(members)) != len(members):
            raise ValueError(f'Unresolved identity collision in {path.name}')
        checkpoints[path.stem[8:]] = set(members)

    changes = pd.read_csv(snapshot / 'membership_changes.csv')
    changes['ISIN'] = changes['Constituent RIC'].map(identities)
    changes.loc[changes['Constituent RIC'].eq('AMCR.N')
                & changes.Date.isin(['2014-12-05', '2019-06-07']), 'ISIN'] = bemis
    corrections = json.loads((snapshot / 'membership_isin_reconciliation.json').read_text())
    kenvue, linde = corrections['correction'], corrections['praxair_linde_correction']
    extra = pd.DataFrame([
        (kenvue['effective'], kenvue['isin'], 'Leaver'),
        (linde['effective'], linde['remove_isin'], 'Leaver'),
        (linde['effective'], linde['add_isin'], 'Joiner'),
        ('2019-06-11', bemis, 'Leaver'),
        ('2019-06-11', amcor, 'Joiner'),
    ], columns=['Date', 'ISIN', 'Change'])
    changes = pd.concat([changes[['Date', 'ISIN', 'Change']], extra])
    assert changes.ISIN.notna().all()
    events = {}
    for date, group in changes.groupby('Date'):
        joins = set(group.loc[group.Change.eq('Joiner'), 'ISIN'])
        leaves = set(group.loc[group.Change.eq('Leaver'), 'ISIN'])
        events[date] = joins - leaves, leaves - joins

    anchor = min(checkpoints)
    members = checkpoints[anchor].copy()
    reverse_conflicts = []
    for date, (joins, leaves) in reversed(list(events.items())):
        if date <= anchor:
            if joins - members or leaves & members:
                reverse_conflicts.append(date)
            members = (members - joins) | leaves
    opening_count = len(members)
    sessions = xc.get_calendar('XNYS', start='2011-01-03', end='2026-11-18').sessions
    dates = sessions.strftime('%Y-%m-%d')
    assert set(events) <= set(dates)
    rows, conflicts, residuals = [], [], {}
    for date in dates:
        joins, leaves = events.get(date, (set(), set()))
        if leaves - members or joins & members:
            conflicts.append({'date': date, 'missing_leavers': sorted(leaves - members),
                              'existing_joiners': sorted(joins & members)})
        members = (members - leaves) | joins
        if date in checkpoints:
            observed = checkpoints[date]
            residuals[date] = {'extra': sorted(members - observed), 'missing': sorted(observed - members)}
        rows.extend((date, isin) for isin in sorted(members))
    pd.DataFrame(rows, columns=['session', 'isin']).to_csv(
        snapshot / 'membership_dated_candidate.csv.gz', index=False)
    audit = dict(opening_count=opening_count, opening_independently_verified=False,
                 rows=len(rows), checkpoints=residuals, forward_conflicts=conflicts,
                 reverse_conflicts=reverse_conflicts,
                 bemis_transition_note='June 11 uses issuer-confirmed first trade; LSEG snapshot still says BMS.')
    (snapshot / 'membership_dated_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
