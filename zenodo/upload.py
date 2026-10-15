'''
Create a Zenodo DRAFT with the GRS / hGRS LUTs and their metadata. The draft is NOT published: review
it on the web page printed at the end, then press "Publish" there.

Usage:
  ZENODO_TOKEN=... python upload.py [--sandbox]

  --sandbox   use https://sandbox.zenodo.org (test site, token created on the sandbox site)

The token is read from the ZENODO_TOKEN environment variable (scopes: deposit:write, deposit:actions).
'''

import hashlib
import json
import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).parent
FILES = {
    # name on Zenodo: (local file, expected SHA256)
    'toa_lut_opac_wind_v3.nc': (
        '/data/grs/grsdata/toa_lut_opac_wind_v3.nc',
        '1ad772695523e48cb6fb9a9230b166737cab34854d8d7b567fe7d8bd037dddec'),
    'transmittance_lut_opac_wind_v3.nc': (
        '/data/grs/grsdata/transmittance_lut_opac_wind_v3.nc',
        '475f825f7d4b112b44266b5f4922b7425c80be7520b36db6cf4affdb688f28a0'),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    base = 'https://sandbox.zenodo.org' if '--sandbox' in sys.argv else 'https://zenodo.org'
    token = os.environ.get('ZENODO_TOKEN')
    if not token:
        sys.exit('set the ZENODO_TOKEN environment variable')

    metadata = json.loads((HERE / 'metadata.json').read_text())
    if 'TODO' in json.dumps(metadata):
        sys.exit('metadata.json still contains TODO: complete it first')
    for creator in metadata['creators']:  # drop the optional fields left empty
        for key in ('affiliation', 'orcid'):
            if not creator.get(key):
                creator.pop(key, None)

    print('checking the local files')
    for name, (path, expected) in FILES.items():
        if sha256(path) != expected:
            sys.exit(f'SHA256 mismatch for {path}')
        print(f'  ok {name}')

    session = requests.Session()
    session.headers['Authorization'] = f'Bearer {token}'

    response = session.post(f'{base}/api/deposit/depositions', json={})
    response.raise_for_status()
    deposition = response.json()
    bucket = deposition['links']['bucket']
    print(f'draft created: {deposition["id"]}')

    for name, (path, _) in FILES.items():
        size = os.path.getsize(path) / 1e6
        print(f'  uploading {name} ({size:.0f} MB)...', flush=True)
        with open(path, 'rb') as file:
            response = session.put(f'{bucket}/{name}', data=file)
        response.raise_for_status()
        print(f'  uploaded, checksum {response.json()["checksum"]}')

    response = session.put(f'{base}/api/deposit/depositions/{deposition["id"]}',
                           json={'metadata': metadata})
    response.raise_for_status()
    print(f'metadata set; reserved DOI: {response.json()["metadata"]["prereserve_doi"]["doi"]}')
    print(f'review and publish the draft: {deposition["links"]["html"]}')


if __name__ == '__main__':
    main()
