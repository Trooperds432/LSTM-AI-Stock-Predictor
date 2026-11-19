"""Audit raw bar availability for the dated membership replay."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def main():
    root = Path(__file__).parent / 'indicators_data/lseg'
    snapshot = root / '20261119T132813190380Z'
    micro = root / 'micro_focus/20261119T173917164993Z'
    bemis = root / 'recovery/20261119T152440659949Z'
    references = [*snapshot.glob('identifiers_*.csv'), micro / 'reference_snapshot.csv',
                  bemis / 'reference_snapshot.csv']
    ref = pd.concat([pd.read_csv(path) for path in references])
    if ref.Instrument.duplicated().any():
        raise ValueError('Duplicate reference instruments')
    identities = ref.set_index('Instrument').ISIN.to_dict()
    transfers = pd.read_csv(Path(__file__).with_name('lseg_listing_transfers.csv'))
    listing_bounds = {}
    for row in transfers.itertuples(index=False):
        if identities[row.prior_ric] != row.isin or identities[row.successor_ric] != row.isin:
            raise ValueError(f'Listing transfer has inconsistent ISIN: {row}')
        listing_bounds[row.prior_ric] = ('', row.first_successor_session)
        listing_bounds[row.successor_ric] = (row.first_successor_session, '9999')
    manifests = [snapshot / 'manifest.json', *(root / 'recovery').glob('*/manifest.json'),
                 micro / 'manifest.json']
    valid = defaultdict(set)
    files = 0
    fields = ['OPEN_PRC', 'HIGH_1', 'LOW_1', 'TRDPRC_1', 'ACVOL_UNS']
    for manifest in manifests:
        for record in json.loads(manifest.read_text())['datasets']:
            if record['kind'] != 'daily_prices' or record['status'] != 'retrieved':
                continue
            file = manifest.parent / record['file']
            if hashlib.sha256(file.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError(f'Checksum mismatch: {file}')
            files += 1
            universe = record['universe']
            batch = isinstance(universe, list)
            bars = pd.read_csv(file, header=[0, 1] if batch else 0, index_col=0)
            for ric in universe if batch else [universe]:
                if ric not in identities or (batch and ric not in bars):
                    continue
                values = (bars[ric] if batch else bars).reindex(columns=fields)
                o, h, l, c, v = (pd.to_numeric(values[col], errors='coerce') for col in fields)
                ok = (l > 0) & (h >= o) & (h >= c) & (l <= o) & (l <= c) & (v > 0)
                for date in values.index[ok]:
                    date = date[:10]
                    first, stop = listing_bounds.get(ric, ('', '9999'))
                    if not first <= date < stop:
                        continue
                    isin = identities[ric]
                    if ric == 'JCI.N':
                        if date < '2014-11-17':
                            isin = 'CH0100383485'
                        elif date < '2016-09-06':
                            isin = 'IE00BQRQXQ92'
                    valid[isin].add(date)
    members = pd.read_csv(snapshot / 'membership_dated_candidate.csv.gz')
    if members.duplicated(['session', 'isin']).any():
        raise ValueError('Duplicate membership keys')
    missing = members.loc[[session not in valid[isin]
                           for session, isin in zip(members.session, members['isin'])]]
    ledger = pd.read_csv(snapshot / 'member_session_exception_ledger.csv')
    keys = ['session', 'isin']
    current = set(missing[keys].itertuples(index=False, name=None))
    previous = set(ledger[keys].itertuples(index=False, name=None))
    audit = dict(member_sessions=len(members), checksum_verified_price_files=files,
                 missing_member_sessions=len(current), absent_from_exception_ledger=sorted(current - previous),
                 no_longer_missing=sorted(previous - current),
                 dated_listing_transfers=len(transfers),
                 quote_basis='Dated listing transfers and Tyco mapping; remaining RICs use downloaded reference ISINs.')
    (snapshot / 'member_price_replay_verification.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({
        'member_sessions': len(members),
        'missing_member_sessions': len(current),
        'absent_from_exception_ledger': len(current - previous),
        'no_longer_missing': len(previous - current),
    }))


if __name__ == '__main__':
    main()
