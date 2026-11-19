"""Compare LSEG split records with raw same-session and next-session opens."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('actions', type=Path)
    parser.add_argument('price_manifests', nargs='+', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.actions / 'manifest.json').read_text())
    records = [d for d in manifest['datasets'] if d['kind'] == 'capital_change_events'
               and d['status'] == 'retrieved']
    events = []
    for record in records:
        path = args.actions / record['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
        events.append(pd.read_csv(path).assign(source_file=record['file']))
    capital = pd.concat(events, ignore_index=True)
    splits = capital[capital['Corporate Change Event Type'].str.contains(
        'split', case=False, na=False)].copy()
    requested = set(splits.Instrument)
    prices = {}
    for path in args.price_manifests:
        for record in json.loads(path.read_text())['datasets']:
            if record['kind'] != 'daily_prices' or record['status'] != 'retrieved':
                continue
            universe = record['universe']
            batch = isinstance(universe, list)
            names = requested.intersection(universe if batch else [universe])
            if not names:
                continue
            file = path.parent / record['file']
            assert hashlib.sha256(file.read_bytes()).hexdigest() == record['sha256'], file
            frame = pd.read_csv(file, header=[0, 1] if batch else 0, index_col=0)
            for ric in names:
                if batch and ric not in frame:
                    continue
                series = frame[ric] if batch else frame
                series = series[['OPEN_PRC', 'TRDPRC_1']].apply(pd.to_numeric, errors='coerce')
                series.index = pd.to_datetime(series.index, errors='coerce')
                series = series.loc[series.index.notna()].sort_index()
                prices[ric] = series.where(series.gt(0)).dropna()
    rows = []
    for event in splits.to_dict('records'):
        series = prices.get(event['Instrument'])
        date = pd.to_datetime(event['Capital Change Effective Date'], errors='coerce')
        factor = pd.to_numeric(event['Adjustment Factor'], errors='coerce')
        out = dict(event, same_session_ratio=None, next_session_ratio=None,
                   next_price_session=None, diagnostic='insufficient_evidence')
        if series is not None and pd.notna(date) and pd.notna(factor) and factor > 0 and factor != 1:
            ratios = series.OPEN_PRC / series.TRDPRC_1.shift()
            same = ratios.get(date)
            later = ratios.loc[(ratios.index > date) & (ratios.index <= date + pd.Timedelta(days=4))]
            nxt = later.iloc[0] if not later.empty else None
            out.update(same_session_ratio=same, next_session_ratio=nxt,
                       next_price_session=str(later.index[0].date()) if not later.empty else None)
            matches = [label for label, ratio in [('same', same), ('next', nxt)]
                       if pd.notna(ratio) and abs(ratio / factor - 1) <= .10]
            if len(matches) == 1:
                out['diagnostic'] = matches[0]
            elif matches:
                out['diagnostic'] = 'ambiguous'
        rows.append(out)
    audit = pd.DataFrame(rows)
    audit.to_csv(args.actions / 'split_timing_audit.csv', index=False)
    print(audit.diagnostic.value_counts().to_dict())


if __name__ == '__main__':
    main()
