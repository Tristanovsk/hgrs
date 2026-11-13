'''
Matchup inventory: EnMAP L1C scenes covering an AERONET-OC station with measurements within
``MAX_LAG`` of the overpass. Writes ``OUT_DIR/matchups.csv`` (one row per scene and station, with the
nearest level 2.0 record, or 1.5 if there is no level 2.0 record).

Usage: python inventory.py [max_lag_hours]
'''
import sys
import warnings

import numpy as np
import pandas as pd

from common import OUT_DIR, enmap_scenes, aeronet_sites, inside, read_aeronet_oc, cams_file

warnings.filterwarnings('ignore')
MAX_LAG = pd.Timedelta(hours=float(sys.argv[1]) if len(sys.argv) > 1 else 3)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = enmap_scenes()
    sites = aeronet_sites()
    print(f'{len(scenes)} EnMAP products, {len(sites)} AERONET-OC sites')
    cache = {}
    rows = []
    for s in scenes.itertuples():
        for st in sites.itertuples():
            if not np.isfinite(st.lon) or not inside(st.lon, st.lat, s.footprint):
                continue
            row = dict(product=s.product, path=s.path, version=s.version, time=s.time, site=st.site,
                       lon=st.lon, lat=st.lat, cams=cams_file(s.time))
            for level in ('lev20', 'lev15'):
                f = getattr(st, level)
                if not f:
                    continue
                if f not in cache:
                    try:
                        cache[f] = read_aeronet_oc(f).index
                    except Exception as e:  # unreadable file
                        print('cannot read', f, e)
                        cache[f] = pd.DatetimeIndex([])
                times = cache[f]
                if len(times) == 0:
                    continue
                lag = np.abs(times - s.time)
                i = int(np.argmin(lag))
                if lag[i] <= MAX_LAG:
                    row.update(level=level, aeronet_time=times[i], lag_min=lag[i].total_seconds() / 60,
                               n_records=int((lag <= MAX_LAG).sum()))
                    break
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'scenes_over_stations.csv', index=False)
    m = df.dropna(subset=['aeronet_time']) if 'aeronet_time' in df else df.iloc[:0]
    m.to_csv(OUT_DIR / 'matchups.csv', index=False)
    print(f'{len(df)} scene-station pairs, {len(m)} with AERONET-OC data within {MAX_LAG}')
    if len(m):
        print(m.groupby(['site', 'level']).size().unstack(fill_value=0))
        print(m.groupby('version').size())


if __name__ == '__main__':
    main()
