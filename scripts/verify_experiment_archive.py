"""Verify raw archives against manifest hashes and staged Git blob content."""
import csv,gzip,hashlib,subprocess
from pathlib import Path
repo=Path(__file__).resolve().parents[1]
root=repo/'reports/all-experiments-20261027'
records=list(csv.DictReader((root/'artifact_manifest.csv').open(encoding='utf-8-sig')))
staged={}
for entry in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=repo).split(b'\0'):
    if not entry:continue
    meta,name=entry.split(b'\t',1)
    staged[name.decode('utf-8')]=meta.split()[1].decode()
checked=set()
for r in records:
    relative=r['archived_file']
    if relative in checked:continue
    p=root/relative;stored=p.read_bytes()
    assert len(stored)<100*1024*1024
    data=gzip.decompress(stored) if r['compressed']=='True' else stored
    assert hashlib.sha256(data).hexdigest()==r['sha256'],relative
    key=p.relative_to(repo).as_posix()
    expected=hashlib.sha1(b'blob '+str(len(stored)).encode()+b'\0'+stored).hexdigest()
    assert staged.get(key)==expected,('missing/normalized/stale staged archive',key)
    checked.add(relative)
print(f'PASS: {len(records)} source mappings, {len(checked)} unique staged archives, original SHA256 and Git blob hashes match.')
