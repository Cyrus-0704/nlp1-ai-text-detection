"""Internal-dev diagnostics only: strict alignment, A metrics, grouped uncertainty, figures."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, log_loss,
                            brier_score_loss, roc_curve, precision_recall_curve)
from common import read_rows, validate_probabilities, sha256, write_json

def export_figure(fig, out, name, panel_auditor=None):
    """Export report figures, optionally checking rendered axes before export."""
    if panel_auditor:
        spec = importlib.util.spec_from_file_location('panel_auditor', panel_auditor)
        auditor = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = auditor
        spec.loader.exec_module(auditor)
        auditor.require_matplotlib_panel_alignment(
            fig, json_out=out/(name+'.alignment.json'), strict=True)
        # Keep measured axes fixed across PNG, editable SVG and PDF exports.
        fig.set_layout_engine('none')
    fig.savefig(out/(name+'.png'), dpi=300)
    fig.savefig(out/(name+'.svg'))
    fig.savefig(out/(name+'.pdf'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold-internal-dev',required=True)
    p.add_argument('--run',action='append',required=True,help='NAME=run_directory; all runs must use the same dev hash')
    p.add_argument('--output-dir',required=True)
    p.add_argument('--bootstrap-repetitions',type=int,default=1000)
    p.add_argument('--panel-auditor', help='Optional trusted Python panel alignment auditor module')
    p.add_argument('--prefix-run',action='append',default=[],
                   help='Run whose training curve is a prefix of another run; metrics still shown')
    args=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    rows=read_rows(args.gold_internal_dev,labeled=True)
    y=np.array([r['label'] for r in rows])
    short=np.array([len(r['text'].split())<100 for r in rows])
    spec=importlib.util.spec_from_file_location('a_eval',root/'data_tools/evaluate.py')
    a=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    runs={}
    for item in args.run:
        name,path=item.split('=',1)
        if name in runs:
            raise ValueError('Duplicate run name')
        folder=Path(path)
        summary=json.loads((folder/'run_summary.json').read_text(encoding='utf-8'))
        if summary['status']!='PASS' or summary['input_files']['dev']['sha256']!=sha256(args.gold_internal_dev):
            raise ValueError('Run failed or dev version mismatch')
        with (folder/'dev_predictions.csv').open(encoding='utf-8',newline='') as f:
            predictions=list(csv.DictReader(f))
        assert [r['sample_id'] for r in predictions]==[r['sample_id'] for r in rows]
        prob=np.array([float(r['p_machine']) for r in predictions])
        validate_probabilities(prob,len(rows))
        pred=(prob>=.5).astype(int)
        f1=a.macro_f1(y,pred)
        f1s=a.macro_f1(y[short],pred[short]) if short.any() else None
        cm=a.confusion_matrix(y,pred)
        metrics=dict(macro_f1_all=f1,macro_f1_short_temporary=f1s,
                     weighted_score_temporary=.7*f1+.3*f1s if f1s is not None else None,
                     roc_auc=float(roc_auc_score(y,prob)),average_precision=float(average_precision_score(y,prob)),
                     log_loss=float(log_loss(y,prob)),brier=float(brier_score_loss(y,prob)),
                     human_recall=float(cm[0,0]/sum(y==0)),machine_recall=float(cm[1,1]/sum(y==1)),
                     predicted_machine=int(pred.sum()),confusion_matrix=cm.tolist(),
                     probability_quantiles_by_label={str(c):np.quantile(prob[y==c],[0,.25,.5,.75,1]).tolist() for c in (0,1)})
        arguments_path=folder/'arguments.json'
        arguments=json.loads(arguments_path.read_text(encoding='utf-8')) if arguments_path.exists() else {}
        runs[name]=dict(folder=folder,summary=summary,prob=prob,pred=pred,metrics=metrics,
                       training_seed=arguments.get('seed'))
        # Only labels, probabilities, and summaries enter figures; never export raw text.
    if not set(args.prefix_run).issubset(runs):
        raise ValueError('Unknown prefix run')
    training_seeds=sorted({r['training_seed'] for r in runs.values() if r['training_seed'] is not None})
    seed_note='training seeds '+','.join(map(str,training_seeds)) if training_seeds else 'training seeds unspecified'
    groups={}
    for i,r in enumerate(rows):
        groups.setdefault(r['group_id'],[]).append(i)
    members=list(groups.values())
    rng=np.random.default_rng(42)
    samples={name:[] for name in runs}
    for _ in range(args.bootstrap_repetitions):
        indices=np.concatenate([members[j] for j in rng.integers(0,len(members),len(members))])
        for name,r in runs.items():
            samples[name].append(a.macro_f1(y[indices],r['pred'][indices]))
    for name,r in runs.items():
        r['metrics']['macro_f1_group_bootstrap_ci95']=np.quantile(samples[name],[.025,.975]).tolist()
    names=list(runs)
    differences={}
    for i in range(1,len(names)):
        previous=names[i-1]
        current=names[i]
        differences[current+' minus '+previous]=dict(
            macro_f1_difference=runs[current]['metrics']['macro_f1_all']-runs[previous]['metrics']['macro_f1_all'],
            paired_group_bootstrap_ci95=np.quantile(np.array(samples[current])-np.array(samples[previous]),[.025,.975]).tolist())
    report=dict(scope='Exploratory internal dev; not official validation or final test',
                gold_sha256=sha256(args.gold_internal_dev),rows=len(rows),groups=len(groups),short_rows=int(short.sum()),
                threshold=.5,short_rule='TEMPORARY len(text.split())<100',
                uncertainty='95% percentile bootstrap of dev groups; conditional on these fitted models; not seed/training uncertainty',
                bootstrap_repetitions=args.bootstrap_repetitions,seed=42,
                runs={name:r['metrics'] for name,r in runs.items()},paired_differences=differences,
                training_seeds=training_seeds,omitted_prefix_curves=args.prefix_run,
                official_score=None,threshold_selected=False)
    write_json(out/'diagnostics.json',report)
    records=[]
    for name,r in runs.items():
        for field in ('domain','generator'):
            values=sorted(set(row[field] for row in rows))
            for value in values:
                mask=np.array([row[field]==value for row in rows])
                n=int(mask.sum())
                cm=a.confusion_matrix(y[mask],r['pred'][mask])
                n_human,n_machine=int(sum(y[mask]==0)),int(sum(y[mask]==1))
                records.append(dict(run=name,field=field,value=value,samples=n,human=int(sum(y[mask]==0)),
                                    machine=int(sum(y[mask]==1)),errors=int(sum(y[mask]!=r['pred'][mask])),
                                    macro_f1=a.macro_f1(y[mask],r['pred'][mask]) if n_human and n_machine else None,
                                    human_recall=float(cm[0,0]/n_human) if n_human else None,
                                    machine_recall=float(cm[1,1]/n_machine) if n_machine else None,
                                    warning='small_n_or_one_class' if n<20 or len(set(y[mask]))<2 else ''))
    with (out/'slices.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    with (out/'metrics.csv').open('w',encoding='utf-8',newline='') as f:
        keys=['macro_f1_all','macro_f1_short_temporary','weighted_score_temporary','roc_auc','log_loss','brier','human_recall','machine_recall','predicted_machine']
        writer=csv.DictWriter(f,fieldnames=['run']+keys)
        writer.writeheader()
        for name,r in runs.items():
            writer.writerow(dict(run=name,**{key:r['metrics'][key] for key in keys}))
    # Optional project-local plotting dependencies do not change the training environment.
    local_deps=root/'.local/analysis_deps'
    if local_deps.exists():
        sys.path.insert(0,str(local_deps))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                         'font.size':10,'axes.spines.top':False,'svg.fonttype':'none',
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    fig,axes=plt.subplots(2,2,figsize=(11.6,8.2),layout='constrained')
    ax=axes[0,0]
    x=np.arange(len(names)); width=.24
    for j,(key,label) in enumerate([('macro_f1_all','All macro F1'),('macro_f1_short_temporary','Short macro F1*'),('weighted_score_temporary','Temporary S*')]):
        bars=ax.bar(x+(j-1)*width,[runs[n]['metrics'][key] for n in names],width,label=label,color=colors[j])
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.set(xticks=x,xticklabels=names,ylim=(0,1.15),title='A  Fixed-threshold performance',ylabel='Score')
    ax.legend(loc='upper left',fontsize=8,ncol=3)
    ax=axes[0,1]
    for i,(name,r) in enumerate(runs.items()):
        fpr,tpr,_=roc_curve(y,r['prob'])
        ax.plot(fpr,tpr,color=colors[i],label=f"{name}: {r['metrics']['roc_auc']:.3f}")
    ax.plot([0,1],[0,1],color='.65',ls='--',lw=1)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='B  Ranking performance',xlim=(0,1),ylim=(0,1.03))
    ax.legend(loc='lower right',fontsize=9)
    ax=axes[1,0]
    for i,(name,r) in enumerate(runs.items()):
        for c,offset in [(0,-.16),(1,.16)]:
            values=r['prob'][y==c]
            parts=ax.boxplot([values],positions=[i+offset],widths=.25,patch_artist=True,
                             showfliers=False,manage_ticks=False)
            parts['boxes'][0].set(facecolor=colors[c],alpha=.65)
    ax.axhline(.5,color='.35',ls='--',lw=1,label='Threshold 0.5')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors[0],label='True human'),Patch(color=colors[1],label='True machine')],fontsize=8,loc='upper left',ncol=2)
    ax.set(xticks=x,xticklabels=names,ylim=(0,1.15),ylabel='p_machine',title='C  Class-wise probability distributions')
    ax=axes[1,1]
    curves = []
    for i,(name,r) in enumerate(runs.items()):
        if name in args.prefix_run or r['summary'].get('inference_source','').startswith('50-step prefix'):
            continue
        path=r['folder']/'training_log.jsonl'
        if not path.exists():
            continue
        log=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        values=np.array([entry.get('unweighted_loss',entry['loss']) for entry in log if not entry['skipped']])
        if len(values)>=10:
            steps = np.arange(10,len(values)+1)
            smoothed = np.convolve(values,np.ones(10)/10,mode='valid')
            ax.plot(steps,smoothed,color=colors[i],label=name)
            curves.extend(dict(run=name,optimizer_step=int(step),unweighted_ce_mean10=float(value))
                          for step,value in zip(steps,smoothed))
    for prefix_name in args.prefix_run:
        endpoint=runs[prefix_name]['summary']['optimizer_updates']
        ax.axvline(endpoint,color=colors[names.index(prefix_name)],ls=':',lw=1,
                   label=f'{prefix_name} endpoint')
    ax.set(xlabel='Optimizer update',ylabel='Unweighted CE (10-update mean)',title='D  Training loss (10-update mean)')
    ax.legend(fontsize=8)
    fig.suptitle(f'Internal dev: {len(rows)} samples / {len(groups)} groups; threshold 0.5\n*Short = <100 whitespace words (temporary); {seed_note}; exploratory',fontsize=11)
    export_figure(fig,out,'performance',args.panel_auditor); plt.close(fig)
    if curves:
        with (out/'training_curves.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(curves[0]))
            writer.writeheader(); writer.writerows(curves)
    fig,axs=plt.subplots(1,len(names),figsize=(max(5,3.8*len(names)),4),layout='constrained',squeeze=False)
    for ax,(name,r) in zip(axs[0],runs.items()):
        cm=np.array(r['metrics']['confusion_matrix'])
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=max(sum(y==0),sum(y==1)))
        for i in range(2):
            for j in range(2):
                ax.text(j,i,f'{cm[i,j]}\n({cm[i,j]/cm[i].sum():.1%})',ha='center',va='center',fontsize=15,color='white' if cm[i,j]>.5*max(sum(y==0),sum(y==1)) else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['human','machine'],yticklabels=['human','machine'],
               xlabel='Predicted',ylabel='True',title=name)
    fig.suptitle('Internal dev confusion matrices; threshold 0.5\nCounts and row percentages; true labels in rows',fontsize=11)
    export_figure(fig,out,'confusion',args.panel_auditor); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.2), layout='constrained')
    ax = axes[0, 0]
    for j, (key, label) in enumerate([('human_recall', 'Human recall'), ('machine_recall', 'Machine recall')]):
        bars = ax.bar(x+(j-.5)*.32, [runs[n]['metrics'][key] for n in names], .32,
                      color=colors[j], label=label)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set(xticks=x, xticklabels=names, ylim=(0, 1.16), ylabel='Recall',
           title='A  Which class improves at threshold 0.5?')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax = axes[0, 1]
    for i, (name, r) in enumerate(runs.items()):
        precision, recall, _ = precision_recall_curve(y, r['prob'])
        ax.plot(recall, precision, color=colors[i], label=f"{name}: AP={r['metrics']['average_precision']:.3f}")
    ax.axhline(y.mean(), color='.65', ls='--', lw=1, label=f'Machine prevalence {y.mean():.3f}')
    ax.set(xlabel='Machine recall', ylabel='Machine precision', xlim=(0, 1),
           ylim=(0, 1.03), title='B  Precision-recall trade-off (no threshold tuning)')
    ax.legend(fontsize=8, loc='lower left')
    ax = axes[1, 0]
    calibration = []
    for i, (name, r) in enumerate(runs.items()):
        bins = np.minimum((r['prob']*10).astype(int), 9)
        points = []
        for b in range(10):
            mask = bins == b
            if not mask.any():
                continue
            item = dict(run=name, bin_low=b/10, bin_high=(b+1)/10, samples=int(mask.sum()),
                        mean_p_machine=float(r['prob'][mask].mean()),
                        observed_machine_fraction=float(y[mask].mean()))
            calibration.append(item)
            points.append(item)
        ax.plot([p['mean_p_machine'] for p in points], [p['observed_machine_fraction'] for p in points],
                '-o', ms=4, color=colors[i], label=name)
    ax.plot([0, 1], [0, 1], color='.65', ls='--', lw=1)
    ax.set(xlabel='Mean p_machine (10 fixed bins)', ylabel='Observed machine fraction',
           xlim=(0, 1), ylim=(0, 1), title='C  Reliability diagnostic; no calibration fitted')
    ax.legend(fontsize=8)
    with (out/'calibration.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(calibration[0]))
        writer.writeheader(); writer.writerows(calibration)
    ax = axes[1, 1]
    values = np.array([runs[n]['metrics']['macro_f1_all'] for n in names])
    intervals = np.array([runs[n]['metrics']['macro_f1_group_bootstrap_ci95'] for n in names])
    for i, name in enumerate(names):
        ax.vlines(i, intervals[i, 0], intervals[i, 1], color=colors[i], lw=2)
        ax.plot(i, values[i], 'o', color=colors[i])
        ax.annotate(f'{values[i]:.3f}\n[{intervals[i,0]:.3f}, {intervals[i,1]:.3f}]',
                    (i, intervals[i, 1]), xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9)
    ax.set(xticks=x, xticklabels=names, xlim=(-.5, len(names)-.5), ylim=(0, 1.12),
           ylabel='All macro F1', title='D  95% dev-group bootstrap intervals')
    fig.suptitle(f'Internal dev diagnostics: {len(rows)} samples; {seed_note}; exploratory\n'
                 'Intervals condition on fitted models; not training-seed uncertainty', fontsize=12)
    export_figure(fig,out,'error_analysis',args.panel_auditor); plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
