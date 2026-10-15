"""Reproduce A's fixed pipeline unchanged; audit train/dev before any B training.

The generated internal test is hashed as opaque bytes only. Its examples/log are
not read, and no test predictions or performance statistics are computed.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from common import read_rows, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--raw-file', required=True)
    p.add_argument('--split-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--reproduce', action='store_true')
    a = p.parse_args()
    repo = Path(__file__).resolve().parent.parent
    manifest = json.loads(Path(a.manifest).read_text(encoding='utf-8-sig'))
    splits = Path(a.split_dir).resolve()
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = dict(data_version=manifest['version'], status='PENDING',
                  a_manifest_sha256=sha256(a.manifest),
                  code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                  official_validation_read=False, internal_test_performance_evaluated=False,
                  internal_test_access='opaque byte count and SHA-256 only; generated log not inspected',
                  raw_file_verified=False, files={}, overlap={}, smoke_membership={})
    if sha256(a.raw_file) != manifest['source']['sha256'] or Path(a.raw_file).stat().st_size != manifest['source']['bytes']:
        raise ValueError('Raw source hash/size mismatch')
    report['raw_file_verified'] = True
    script = repo/manifest['generation']['script']
    report['a_preprocess_script_sha256'] = sha256(script)
    if a.reproduce:
        if splits.exists():
            raise FileExistsError('Reproduction requires a new split directory')
        command = [sys.executable,str(script),'--data',str(Path(a.raw_file).resolve()),
                   '--out-dir',str(splits),'--seed',str(manifest['generation']['seed']),
                   '--min-len',str(manifest['generation']['min_text_len'])]
        # A's unchanged script emits examples for all internal splits. Keep this private
        # log unread; never surface or parse the internal test examples.
        with (out/'private_preprocess.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    for name,expected in manifest['splits'].items():
        path = splits/expected['path']
        actual = dict(bytes=path.stat().st_size,sha256=sha256(path))
        actual['matches_a_manifest'] = actual['bytes']==expected['bytes'] and actual['sha256']==expected['sha256']
        report['files'][name] = actual
    write_json(out/'data_acceptance.json',report)
    reproduction_matches = all(item['matches_a_manifest'] for item in report['files'].values())
    # Hash failure remains a hard training gate. Continue read-only train/dev
    # diagnostics to give A actionable evidence without altering the frozen split.
    rows = {name:read_rows(splits/manifest['splits'][name]['path'],labeled=True) for name in ('train','dev')}
    for name,items in rows.items():
        counts = Counter(r['label'] for r in items)
        expected=manifest['splits'][name]
        if len(items)!=expected['samples'] or counts[0]!=expected['human'] or counts[1]!=expected['machine']:
            raise ValueError('Split row/class counts differ from A manifest')
        if not all(set(r)==set(manifest['fields']) and isinstance(r['sample_id'],str)
                   and isinstance(r['group_id'],str) and (r['label']!=0 or r['generator']=='human') for r in items):
            raise ValueError('Invalid six-field sample contract')
        report['files'][name].update(rows=len(items),human=counts[0],machine=counts[1],
                                   unique_groups=len({r['group_id'] for r in items}),six_fields_verified=True)
    evidence=[]
    transforms=dict(sample_id=lambda r:r['sample_id'],group_id=lambda r:r['group_id'],
                    whitespace_text=lambda r:' '.join(r['text'].split()),
                    casefold_whitespace_text=lambda r:' '.join(r['text'].split()).casefold())
    for kind,fn in transforms.items():
        maps={}
        for name,items in rows.items():
            mapping=defaultdict(list)
            for r in items:
                mapping[fn(r)].append(r)
            maps[name]=mapping
        shared=maps['train'].keys() & maps['dev'].keys()
        report['overlap'][kind]=dict(shared_keys=len(shared),
             affected_train_rows=sum(len(maps['train'][k]) for k in shared),
             affected_dev_rows=sum(len(maps['dev'][k]) for k in shared))
        # Fingerprints and IDs only; no text exports. Stable order aids A's reproduction.
        for key in sorted(shared)[:100]:
            left,right=maps['train'][key],maps['dev'][key]
            evidence.append(dict(check=kind,fingerprint_sha256=hashlib.sha256(key.encode()).hexdigest(),
                                 train_sample_ids=[r['sample_id'] for r in left],
                                 dev_sample_ids=[r['sample_id'] for r in right],
                                 train_group_ids=sorted({r['group_id'] for r in left}),
                                 dev_group_ids=sorted({r['group_id'] for r in right}),
                                 train_labels=sorted({r['label'] for r in left}),
                                 dev_labels=sorted({r['label'] for r in right})))
    for name,key in [('train','smoke_train'),('dev','smoke_dev')]:
        small=Path(a.manifest).resolve().parent/manifest['smoke']['files'][key]['path']
        smoke=read_rows(small,labeled=True)
        by_id={r['sample_id']:r for r in rows[name]}
        matched=sum(by_id.get(r['sample_id'])==r for r in smoke)
        report['smoke_membership'][name]=dict(smoke_rows=len(smoke),exact_matches=matched,passed=matched==len(smoke))
    write_json(out/'overlap_evidence.json',dict(contains_raw_text=False,per_check_key_limit=100,evidence=evidence))
    isolation_passed=all(v['shared_keys']==0 for v in report['overlap'].values())
    membership_passed=all(v['passed'] for v in report['smoke_membership'].values())
    passed=reproduction_matches and isolation_passed and membership_passed
    report['training_gate_passed']=passed
    report['failed_checks']=([name for name,ok in [('file_reproduction',reproduction_matches),
                             ('train_dev_isolation',isolation_passed),('smoke_membership',membership_passed)] if not ok])
    report['status']='PASS_TRAIN_DEV_ACCEPTANCE' if passed else 'FAIL_TRAIN_DEV_ACCEPTANCE'
    write_json(out/'data_acceptance.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if not passed:
        raise SystemExit(2)


if __name__=='__main__':
    main()
