'''
LUT names and download from Zenodo (hgrs.config): published name of the TOA LUT, fallback to its former
name, `download_luts` against a fake Zenodo record (no network access).
'''
import hashlib
import io
import json

import pytest

from hgrs.config import app_config, loader

TOA = 'toa_lut_opac_wind_v3.nc'
TOA_FORMER = 'toa_lut_opac_wind_up_v3.nc'
TRANS = 'transmittance_lut_opac_wind_v3.nc'


@pytest.fixture
def lut_dir(tmp_path, monkeypatch):
    '''empty LUT folder set by HGRS_DATA_ROOT, empty user configuration'''
    user = tmp_path / 'config.yml'
    user.write_text('{}\n')
    monkeypatch.setenv('HGRS_CONFIG', str(user))
    monkeypatch.setenv('HGRS_DATA_ROOT', str(tmp_path / 'lut'))
    (tmp_path / 'lut').mkdir()
    loader._load_config.cache_clear()
    yield tmp_path / 'lut'
    loader._load_config.cache_clear()


def fake_zenodo(monkeypatch, contents, corrupt=()):
    '''serve a Zenodo record with the files `contents` {name: bytes}; md5 wrong for the names in `corrupt`'''
    record_url = loader.zenodo_record_url()
    files = [dict(key=name, size=len(data), links=dict(self=f'https://files.test/{name}'),
                  checksum='md5:' + ('0' * 32 if name in corrupt else hashlib.md5(data).hexdigest()))
             for name, data in contents.items()]
    record = json.dumps(dict(files=files, links=dict(self_html='https://zenodo.test/record'))).encode()
    requested = []

    def urlopen(url):
        requested.append(url)
        if url == record_url:
            return io.BytesIO(record)
        return io.BytesIO(contents[url.rsplit('/', 1)[1]])

    monkeypatch.setattr(loader.urllib.request, 'urlopen', urlopen)
    return requested


def test_default_toa_lut_is_the_published_one(lut_dir):
    config = app_config.get_app_config()
    assert config.lut_files['toa'] == TOA
    assert loader.zenodo_record_url() == 'https://zenodo.org/api/records/23394202'


def test_former_name_of_the_toa_lut_is_used_when_it_is_the_one_there(lut_dir):
    config = app_config.get_app_config()
    assert config.resolve_lut('toa') == lut_dir / TOA            # missing: reported under the configured name
    (lut_dir / TOA_FORMER).write_bytes(b'lut')
    assert config.resolve_lut('toa') == lut_dir / TOA_FORMER
    (lut_dir / TOA).write_bytes(b'lut')
    assert config.resolve_lut('toa') == lut_dir / TOA            # the configured name first


def test_download_writes_the_missing_luts_and_keeps_the_packaged_one(lut_dir, monkeypatch):
    contents = {TOA: b'toa lut', TRANS: b'transmittance lut'}
    fake_zenodo(monkeypatch, contents)
    paths = loader.download_luts(progress=False)
    assert (lut_dir / TOA).read_bytes() == contents[TOA]
    assert (lut_dir / TRANS).read_bytes() == contents[TRANS]
    assert not list(lut_dir.glob('*.part'))
    gas = app_config.get_app_config().resolve_lut('abs_gas')
    assert gas in paths and 'data/lut' in gas.as_posix()       # shipped with the package, not downloaded
    assert all(exists for _, _, exists in loader.lut_status())


def test_download_skips_the_luts_there_unless_forced(lut_dir, monkeypatch):
    (lut_dir / TOA_FORMER).write_bytes(b'former copy')
    (lut_dir / TRANS).write_bytes(b'old')
    requested = fake_zenodo(monkeypatch, {TOA: b'toa lut', TRANS: b'transmittance lut'})
    loader.download_luts(progress=False)
    assert requested == [loader.zenodo_record_url()]
    assert not (lut_dir / TOA).exists()
    loader.download_luts(force=True, progress=False)
    assert (lut_dir / TOA).read_bytes() == b'toa lut'           # downloaded under the published name
    assert (lut_dir / TRANS).read_bytes() == b'transmittance lut'
    assert app_config.get_app_config().resolve_lut('toa') == lut_dir / TOA


def test_download_checksum_mismatch_leaves_no_file(lut_dir, monkeypatch):
    fake_zenodo(monkeypatch, {TOA: b'toa lut', TRANS: b'transmittance lut'}, corrupt=(TOA,))
    with pytest.raises(OSError, match='checksum'):
        loader.download_luts(progress=False)
    assert not (lut_dir / TOA).exists()
    assert not list(lut_dir.glob('*.part'))


def test_missing_lut_not_in_the_record(lut_dir, monkeypatch):
    fake_zenodo(monkeypatch, {TRANS: b'transmittance lut'})
    with pytest.raises(FileNotFoundError, match=TOA):
        loader.download_luts(progress=False)


def test_former_name_is_downloaded_under_the_published_one():
    assert loader._published_name(TOA_FORMER, {TOA: {}}) == TOA
    assert loader._published_name(TOA, {TOA: {}}) == TOA
    assert loader._published_name('other.nc', {TOA: {}}) is None
