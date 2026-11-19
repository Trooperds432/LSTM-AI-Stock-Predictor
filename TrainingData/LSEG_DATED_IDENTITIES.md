# Dated identity corrections

The December 2015 constituent response contains 505 rows but only 502
distinct RICs. Preserve the constituent name when resolving these rows.
Current reference lookup alone assigns the old constituent to its successor.

| 2015 constituent name | Returned RIC | Historical price RIC | ISIN |
| --- | --- | --- | --- |
| DOW CHEMICAL ORD | DD.N | DOW.N^I17 | US2605431038 |
| EI D P DE NEMOUR ORD | DD.N | DD.N^I17 | US2635341090 |
| FMC TECHNOLOGIES ORD | FTI.N | FTI.N^A17 | US30249U1016 |
| CHUBB ORD | CB.N | CB.N^A16 | US1712321017 |
| JOHNSON CONTROLS ORD | JCI.N | JCI.N^I16 | US4783661071 |

The historical RIC assignments agree with the expired instruments in LSEG
reference data and the corresponding leaver records. These mappings apply to
the named historical rows. They do not stitch successor prices or returns.

The reconstructed 2011 roster carries the Irish Tyco/Johnson Controls
identifier backward. Vanguard's year-end 2010 holdings distinguish Johnson
Controls Inc. from Tyco International Ltd. Tyco's SEC filing confirms it
changed domicile from Switzerland to Ireland on 17 November 2014 and
exchanged shares one for one. The opening roster therefore still needs an
effective-dated Tyco share-class mapping before it can be checked by ISIN.
The fund filing is an independent holdings check, with issuer names rather
than permanent identifiers. Direct programmatic SEC retrieval returned 403.

- [Vanguard year-end 2010 report](https://www.sec.gov/Archives/edgar/data/36405/000093247111001799/indexfunds_final.htm)
- [Tyco closing filing](https://www.sec.gov/Archives/edgar/data/833444/000119312514414675/d823374d8k.htm)

LSEG's 5 December 2014 leaver and 7 June 2019 joiner labelled AMCR.N both
refer to Bemis. S&P's notices identify Bemis as the company removed after
the close on 4 December 2014 and added before the open on 7 June 2019.
Bemis was recovered as BMS.N^F19 (US0814371052). Its last daily price is
10 June 2019. The replay applies the issuer-confirmed first Amcor trading
date of 11 June even though the LSEG snapshot still names Bemis that day.

The recomputed audit verified 190 price-file checksums and found 137
member-sessions without valid OHLCV out of 2,009,434. The coverage check
accepts any downloaded RIC with the same ISIN and therefore does not yet
establish that each accepted quote belongs to the effective listing.

The two largest gaps have documented trading constraints. Millrose's 11
missing sessions precede its when-issued trading and begin on the spin-off
record date. Its regular-way trading began on 7 February 2025. Noble
Energy's five missing sessions follow the Chevron acquisition and precede
S&P's stated 12 October 2020 index deletion. Preserve these dates as
non-executable pending reconciliation of any distribution or merger payout.

- [Millrose spin-off timetable](https://www.sec.gov/Archives/edgar/data/920760/000119312525004622/d917261dex991.htm)
- [Noble Energy index deletion](https://press.spglobal.com/2020-10-05-Vontier-Set-to-Join-S-P-500)

- [S&P 2014 Bemis deletion](https://press.spglobal.com/2014-12-01-Royal-Caribbean-Cruises-Set-to-Join-the-S-P-500-Bemis-Tyler-Technologies-to-Join-the-S-P-MidCap-400-Montpelier-Re-to-Join-the-S-P-SmallCap-600)
- [S&P 2019 Bemis addition](https://press.spglobal.com/2019-06-03-Bemis-Set-to-Join-S-P-500-Mattel-to-Join-S-P-MidCap-400)
- [TechnipFMC trading start and predecessor](https://www.sec.gov/files/litigation/admin/2019/34-87055.pdf)
- [DowDuPont merger](https://www.sec.gov/Archives/edgar/data/1666700/000119312517274834/d426905dex991.htm)
