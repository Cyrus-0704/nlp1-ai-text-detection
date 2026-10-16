"""Standalone CPU reference audit for an independent A prediction directory."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from common import read_rows
from fusion_inference import digest,save,verify_bundle


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=['sample_id','p_machine']:raise ValueError('Unexpected prediction columns')
        rows=list(reader)
    ids=[r['sample_id'] for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate IDs')
    p=np.array([float(r['p_machine']) for r in rows])
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Invalid probability')
    return ids,p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--predictions',required=True,type=Path,help='Independent prediction directory')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Never overwrite an acceptance report')
    meta=verify_bundle(args.bundle)
    ids=[r['sample_id'] for r in read_rows(args.input)]
    checks={}
    for component in ('C','B','predictions'):
        actual_ids,actual=read(args.predictions/(component+'.csv'))
        if actual_ids!=ids:raise ValueError('Input/prediction ID order mismatch')
        ref_ids,ref=read(args.bundle/'references'/(component+'.csv'))
        mapping=dict(zip(ref_ids,ref))
        if not set(ids)<=mapping.keys():raise ValueError('Reference does not cover input')
        expected=np.array([mapping[i] for i in ids])
        rt,at=(1e-6,1e-8) if component=='C' else (1e-5,1e-6)
        np.testing.assert_allclose(actual,expected,rtol=rt,atol=at)
        if not np.array_equal(actual>=.5,expected>=.5):raise ValueError('Threshold decisions differ')
        checks[component]=dict(max_abs_difference=float(np.max(np.abs(actual-expected))),rtol=rt,atol=at)
    save(args.output,dict(status='PASS_REFERENCE_CHECK',bundle_sha256=digest(args.bundle/'bundle.json'),
        input_sha256=digest(args.input),rows=len(ids),checks=checks,
        reviewer_receipt='This numeric check alone is not a signed A acceptance receipt'))


if __name__=='__main__':main()
