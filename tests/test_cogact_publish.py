import io
import json
from pathlib import Path
import runpy
import urllib.error


def publisher():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / 'tools/cogact_publish.py'))


def test_markers_follow_all_verified_files(tmp_path, monkeypatch):
    code = publisher()
    calls = []
    def upload(entry):
        calls.append(entry['path'])
        return 3
    manifest = {'files': [{'path': p, 'url': 'unused'} for p in
                         ('a/complete.json', 'a/weights.pt', 'a/rng.pt')],
                'receipt_path': str(tmp_path / 'receipt.json'), 'result_subdir': 'probe'}
    monkeypatch.setitem(code['main'].__globals__, 'upload', upload)
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(manifest)))
    assert code['main']() == 0
    assert set(calls[:2]) == {'a/weights.pt', 'a/rng.pt'}
    assert calls[-1] == 'a/complete.json'
    assert json.loads((tmp_path / 'receipt.json').read_text())['verified_files'] == 3


def test_failure_never_publishes_marker_or_signed_url(tmp_path, monkeypatch):
    code = publisher()
    calls = []
    def upload(entry):
        calls.append(entry['path'])
        raise RuntimeError('upload https://store/blob?sig=private-token failed')
    manifest = {'files': [{'path': p, 'url': 'unused'} for p in
                         ('a/weights.pt', 'a/complete.json')],
                'receipt_path': str(tmp_path / 'receipt.json'), 'result_subdir': 'probe'}
    monkeypatch.setitem(code['main'].__globals__, 'upload', upload)
    monkeypatch.setattr('sys.stdin', io.StringIO(json.dumps(manifest)))
    assert code['main']() == 1
    assert calls == ['a/weights.pt']
    receipt = (tmp_path / 'receipt.json').read_text()
    assert 'private-token' not in receipt
    assert json.loads(receipt)['exit_code'] == 1


def test_head_retries_a_connection_reset(monkeypatch):
    code = publisher()
    attempts = []
    class Response:
        headers = {'Content-Length': '12'}
        def __enter__(self): return self
        def __exit__(self, *args): pass
    def urlopen(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.URLError(ConnectionResetError('reset'))
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    monkeypatch.setattr('time.sleep', lambda _: None)
    assert code['remote_size']('https://example.test/blob') == 12
    assert len(attempts) == 2


def test_verified_existing_file_is_not_reuploaded(tmp_path, monkeypatch):
    code = publisher()
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'abc')
    monkeypatch.setitem(code['upload'].__globals__, 'remote_size', lambda _: 3)
    def unexpected(*args, **kwargs):
        raise AssertionError('An already verified file must not be retransmitted')
    monkeypatch.setattr('subprocess.run', unexpected)
    assert code['upload']({'path': str(path), 'url': 'unused', 'verify_existing': True}) == 3
