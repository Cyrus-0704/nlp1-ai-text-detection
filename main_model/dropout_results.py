"""Measured single-B and fixed C/B comparisons; no model loading or selection."""
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score,log_loss,brier_score_loss,roc_auc_score
from common import read_rows,write_predictions
from round_support import ROOT,digest,predictions,csv_out,dump,read_json


def generate(config,out,destination):
    from local_round import evaluation_runs
    destination.mkdir()
    rows=read_rows(ROOT/config['dev_file'],labeled=True)
    ids=[r['sample_id'] for r in rows];y=np.array([r['label'] for r in rows])
    short=np.array([len(r['text'].split())<100 for r in rows])
    cpath=ROOT/config['c_reference']
    if digest(cpath)!=config['c_reference_sha256']:raise ValueError('C reference changed')
    pc=np.array(predictions(cpath,ids))
    records=[]
    for name,folder in evaluation_runs(config,out).items():
        args=read_json(folder/'arguments.json');s=read_json(folder/'run_summary.json')
        if s['status']!='PASS' or s['input_files']['dev']['sha256']!=config['dev_sha256']:
            raise ValueError('Comparison dev/version/status mismatch')
        pb=np.array(predictions(folder/'dev_predictions.csv',ids))
        for mode,p in [('B',pb),('C07_B03',.7*pc+.3*pb)]:
            pred=p>=.5;cm=confusion_matrix(y,pred,labels=[0,1])
            all_f1=float(f1_score(y,pred,average='macro'));short_f1=float(f1_score(y[short],pred[short],average='macro'))
            records.append(dict(run=name,mode=mode,seed=args['seed'],steps=s['optimizer_updates'],
                classifier_dropout=args.get('classifier_dropout') or .1,
                macro_f1_all=all_f1,macro_f1_short_temporary=short_f1,S_temporary=.7*all_f1+.3*short_f1,
                human_recall=float(cm[0,0]/cm[0].sum()),machine_recall=float(cm[1,1]/cm[1].sum()),
                human_false_positive=int(cm[0,1]),machine_false_negative=int(cm[1,0]),
                human_confident_errors_p_ge_099=int(((y==0)&(p>=.99)).sum()),
                log_loss=float(log_loss(y,p,labels=[0,1])),brier=float(brier_score_loss(y,p)),auc=float(roc_auc_score(y,p)),
                attempts=s['attempted_updates'],fp16_skips=s['fp16_skipped_updates'],seen=s['processed_samples']))
            if mode=='C07_B03':write_predictions(destination/(name+'_fixed_fusion.csv'),rows,p)
    csv_out(destination/'single_and_fixed_fusion.csv',records)
    dump(destination/'protocol.json',dict(status='MEASURED_NODES_ONLY',threshold=.5,short='temporary whitespace words<100',
        weights={'C':.7,'B':.3},weight_search=False,checkpoint_selection=False,
        comparison='Report both seeds42/43 and all predetermined nodes, no seed cherry-picking',
        missing_nodes='Absent candidate rows are not measured; never interpolate or backfill',
        diagnostics='Train loss curve is per-update CE, not a train/dev generalization-gap estimate'))
