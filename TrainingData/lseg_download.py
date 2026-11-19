"""Download an isolated LSEG research snapshot; never overwrite training inputs.

Uses LSEG_APP_KEY, or the Windows user-level value if the inherited value is stale.
Requires an authenticated local Workspace and the optional lseg-data package.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', default='2011-01-01')
    parser.add_argument('--end', default='2026-11-18')
    parser.add_argument('--rics', nargs='*', default=['AAPL.O', 'MSFT.O', 'SPY.P'])
    parser.add_argument('--macro-rics', nargs='*', default=['.VIX', '.SPX', '.DXY', 'US2YT=RR', 'US10YT=RR'])
    parser.add_argument('--skip-supplemental', action='store_true')
    parser.add_argument('--sp500-universe', action='store_true',
                        help='Acquire historical membership, reference data and batched equity history.')
    parser.add_argument('--actions-from', type=Path,
                        help='Read RICs from a CSV and acquire corporate actions only.')
    parser.add_argument('--output', type=Path, default=Path('TrainingData/indicators_data/lseg'))
    args = parser.parse_args()
    import lseg.data as ld
    logging.disable(logging.CRITICAL)  # SDK diagnostics can contain authentication context.
    keys = []
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as registry:
                keys.append(winreg.QueryValueEx(registry, 'LSEG_APP_KEY')[0])
        except OSError:
            pass
    keys.append(os.environ.get('LSEG_APP_KEY'))
    keys = list(dict.fromkeys(key.strip() for key in keys if key and key.strip()))
    if not keys:
        raise SystemExit('LSEG_APP_KEY is not configured.')
    session = None
    for key in keys:
        candidate = ld.session.desktop.Definition(app_key=key).get_session()
        if candidate.open().name == 'Opened':
            session = candidate
            break
        candidate.close()
    if session is None:
        raise SystemExit('Workspace authentication failed; check your app key and login.')
    ld.session.set_default(session)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = args.output / stamp
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {
        'retrieved_at_utc': stamp, 'provider': 'LSEG Workspace desktop',
        'sdk_version': ld.__version__, 'start': args.start, 'end': args.end,
        'purpose': 'Data acquisition and coverage only; no strategy confirmation results.',
        'datasets': [],
        'limitations': [
            'Current reference data is not historical point-in-time reference data.',
            'Restated fundamentals are not proven as-reported or point-in-time.',
            'Price adjustment and total-return conventions require reconciliation.',
            'No verified delisting returns, historical spread model, disclosure timestamps, or news archive yet.',
            'Pilot RIC list is not a survivorship-free research universe.',
        ],
    }

    def request(name, kind, universe, fields, parameters=None, adjustments=None):
        record = dict(name=name, kind=kind, universe=universe, fields=fields,
                      parameters=parameters, adjustments=adjustments)
        try:
            if kind == 'daily_prices':
                frame = ld.get_history(universe=universe, fields=fields, interval='1D',
                                       start=args.start, end=args.end, adjustments=adjustments)
            else:
                frame = ld.get_data(universe=universe, fields=fields, parameters=parameters)
            if frame is None or frame.empty:
                record.update(status='empty', rows=0)
            else:
                path = destination / (re.sub(r'[^A-Za-z0-9_-]', '_', name) + '.csv')
                frame.to_csv(path, index=kind == 'daily_prices')
                record.update(status='retrieved', rows=len(frame), file=path.name,
                              columns=[str(c) for c in frame.columns],
                              missing_by_column={str(c): int(frame[c].isna().sum()) for c in frame.columns},
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if kind == 'daily_prices':
                    record.update(first_date=str(frame.index.min()), last_date=str(frame.index.max()),
                                  duplicate_dates=int(frame.index.duplicated().sum()))
        except Exception as exc:
            message = str(exc)
            for secret in keys:
                message = message.replace(secret, '[REDACTED]')
            record.update(status='failed', error_type=type(exc).__name__, error=message[:1500])
        manifest['datasets'].append(record)
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(name, record['status'], record.get('rows', ''), flush=True)
        return frame if record['status'] == 'retrieved' else None

    try:
        if args.actions_from:
            import pandas as pd

            rics = sorted(set(pd.read_csv(args.actions_from)['RIC'].dropna()) | {'SPY.P', 'BMS.N^F19'})
            dates = {'SDate': args.start, 'EDate': args.end}
            for offset in range(0, len(rics), 25):
                batch = rics[offset:offset + 25]
                request(f'dividends_{offset:04d}', 'cash_dividend_events', batch,
                        ['TR.DivExDate', 'TR.DivPayDate', 'TR.DivAnnouncementDate',
                         'TR.DivUnadjustedGross', 'TR.DivType', 'TR.DivCurrency'],
                        {**dates, 'DateType': 'ED'})
                request(f'capital_changes_{offset:04d}', 'capital_change_events', batch,
                        ['TR.CACorpActEventType', 'TR.CAEffectiveDate', 'TR.CAAdjustmentFactor',
                         'TR.CAAdjustmentType', 'TR.CATermsOldShares', 'TR.CATermsNewShares'], dates)
            return
        if args.sp500_universe:
            import pandas as pd
            import exchange_calendars as xc

            calendar = xc.get_calendar('XNYS', start=args.start, end=args.end)
            first_session = calendar.sessions[0].strftime('%Y-%m-%d')
            last_session = calendar.sessions[-1].strftime('%Y-%m-%d')
            members = []
            for date in dict.fromkeys([first_session, '2015-12-31', '2019-12-31', '2023-12-29', last_session]):
                if not args.start <= date <= args.end:
                    continue
                frame = request('members_' + date, 'historical_membership',
                                '.SPX', ['TR.IndexConstituentRIC', 'TR.IndexConstituentName'],
                                {'SDate': date, 'EDate': date})
                if frame is not None:
                    members.extend(frame['Constituent RIC'].dropna())
            changes = request('membership_changes', 'historical_membership_changes', '.SPX',
                              ['TR.IndexJLConstituentChangeDate', 'TR.IndexJLConstituentRIC',
                               'TR.IndexJLConstituentituentChange'],
                              {'SDate': args.start, 'EDate': args.end, 'IC': 'B'})
            if changes is not None:
                members.extend(changes['Constituent RIC'].dropna())
            rics = sorted({ric for ric in members if isinstance(ric, str) and ric.strip()})
            pd.DataFrame({'RIC': rics}).to_csv(destination / 'requested_equity_rics.csv', index=False)
            for offset in range(0, len(rics), 5):
                batch = rics[offset:offset + 5]
                request(f'equities_{offset:04d}', 'daily_prices', batch,
                        ['OPEN_PRC', 'HIGH_1', 'LOW_1', 'TRDPRC_1', 'ACVOL_UNS'],
                        adjustments='unadjusted')
            for offset in range(0, len(rics), 100):
                batch = rics[offset:offset + 100]
                request(f'identifiers_{offset:04d}', 'current_reference_only', batch,
                        ['TR.RIC', 'TR.ISIN', 'TR.QuotePermID', 'TR.InstrumentPermID',
                         'TR.GICSSector'])
        for ric in args.rics:
            request(ric + '_prices', 'daily_prices', ric,
                    ['OPEN_PRC', 'HIGH_1', 'LOW_1', 'TRDPRC_1', 'ACVOL_UNS'], adjustments='unadjusted')
        for ric in args.macro_rics:
            request(ric + '_macro', 'daily_prices', ric,
                    ['BID'] if ric.endswith('YT=RR') else ['TRDPRC_1'])
        if args.skip_supplemental:
            return
        request('reference_snapshot', 'current_reference_only', args.rics,
                ['TR.RIC', 'TR.CommonName', 'TR.ISIN', 'TR.PermID', 'TR.GICSSector', 'TR.CompanyMarketCap'])
        for field in ['TR.Revenue', 'TR.NetIncome', 'TR.TotalAssets']:
            request(field + '_history', 'historical_fundamentals_not_pit_verified', args.rics[:2],
                    [field + '.date', field],
                    {'SDate': args.start, 'EDate': args.end, 'Frq': 'FQ'})
        request('sp500_members_at_start', 'historical_index_membership_probe',
                f'0#.SPX({args.start})', ['TR.RIC', 'TR.CommonName', 'TR.ISIN'])
    finally:
        session.close()
    print('Snapshot:', destination.resolve(), flush=True)


if __name__ == '__main__':
    main()
