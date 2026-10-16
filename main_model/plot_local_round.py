"""Matplotlib comparisons from measured internal-dev nodes; no model inference."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from round_support import ROOT, csv_out, digest, dump, predictions, read_json
from analyze_performance import export_figure
from figure_audit.audit_panel_alignment import require_matplotlib_panel_alignment


def metrics_for_run(gold,path,config):
    import numpy as np
    from common import read_rows
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, log_loss, brier_score_loss
    if digest(gold)!=config['dev_sha256']:
        raise ValueError('Actual gold dev differs from frozen comparison version')
    rows = read_rows(gold,labeled=True)
    s = read_json(path/'run_summary.json')
    if s['status']!='PASS' or s['input_files']['dev']['sha256']!=config['dev_sha256']:
        raise ValueError('Run not accepted or dev hash differs')
    if s['threshold']!=.5 or s['a_evaluation']['short_by']!='word':
        raise ValueError('Threshold/temporary short-text evaluation mismatch')
    probs = np.array(predictions(path/'dev_predictions.csv',[r['sample_id'] for r in rows]))
    y = np.array([r['label'] for r in rows])
    short = np.array([len(r['text'].split())<100 for r in rows])
    pred = probs>=.5
    cm = confusion_matrix(y,pred,labels=[0,1])
    f1 = float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0))
    f1s = float(f1_score(y[short],pred[short],labels=[0,1],average='macro',zero_division=0)) if short.any() else None
    metrics = dict(macro_f1_all=f1,macro_f1_short_temporary=f1s,
        weighted_score_temporary=.7*f1+.3*f1s if f1s is not None else None,
        human_recall=float(cm[0,0]/cm[0].sum()),machine_recall=float(cm[1,1]/cm[1].sum()),
        roc_auc=float(roc_auc_score(y,probs)),log_loss=float(log_loss(y,probs)),
        brier=float(brier_score_loss(y,probs)))
    return metrics,dict(y=y,prob=probs,short=short,cm=cm,summary=s,
                        arguments=read_json(path/'arguments.json'))


def audit(out):
    from PIL import Image
    audit_dir = ROOT/'main_model/figure_audit'
    records = []
    if not (out/'figure_manifest.json').is_file():
        dump(out/'automatic_audit.json',dict(status='FAILED_INCOMPLETE_RENDER',manual_visual_review='PENDING_AFTER_RUN'))
        raise RuntimeError('Rendering did not finish; rerun plot phase before auditing')
    integrity=read_json(out/'figure_manifest.json')['files']
    for name,item in integrity.items():
        if digest(out/name)!=item['sha256']:
            raise ValueError(f'Figure/source export differs from rendered manifest: {name}')
    index = read_json(out/'figure_index.json')
    for source in ('plot_local_round.py','analyze_performance.py'):
        r = subprocess.run([sys.executable,'-X','utf8',str(audit_dir/'validate_figure.py'),str(ROOT/'main_model'/source),
                            '--backend','python','--json'],capture_output=True,text=True,encoding='utf-8')
        (out/(source+'.preflight.json')).write_text(r.stdout or json.dumps({'stderr':r.stderr}),encoding='utf-8')
        records.append(dict(source=source,audit='source',exit_code=r.returncode))
    for name in index['figures']:
        with Image.open(out/(name+'.png')) as raster:
            dpi=raster.info.get('dpi',(0,0))
            records.append(dict(figure=name,audit='PNG_300dpi',exit_code=0 if all(299<=v<=301 for v in dpi) else 1,dpi=dpi))
        svg=(out/(name+'.svg')).read_text(encoding='utf-8')
        records.append(dict(figure=name,audit='SVG_editable_text',exit_code=0 if '<text ' in svg else 1))
        alignment = out/(name+'.alignment.json')
        if not alignment.exists():
            records.append(dict(figure=name,audit='alignment',exit_code=2,reason='Missing rendered alignment report'))
        else:
            geometry = read_json(alignment)
            verdict=geometry.get('verdict')
            okay=verdict in ('PASS','NOT APPLICABLE') and (geometry.get('auditable') or not geometry.get('applicable'))
            records.append(dict(figure=name,audit='alignment',exit_code=0 if okay else 1,verdict=verdict))
        for script,options in [('audit_pdf_text.py',['--min-pt','5','--json']),
                ('audit_figure_collisions.py',['--strict','--json-out',str(out/(name+'.collision.json'))])]:
            r = subprocess.run([sys.executable,'-X','utf8',str(audit_dir/script),str(out/(name+'.pdf')),*options],
                               capture_output=True,text=True,encoding='utf-8')
            target = out/(name+('.text.json' if script=='audit_pdf_text.py' else '.collision.stdout.txt'))
            target.write_text(r.stdout or json.dumps(dict(stderr=r.stderr)),encoding='utf-8')
            records.append(dict(figure=name,audit=script,exit_code=r.returncode,stderr=r.stderr))
    failed = any(r['exit_code']!=0 for r in records)
    dump(out/'automatic_audit.json',dict(status='FAILED_OR_REVIEW_REQUIRED' if failed else 'PASS',
        records=records,manual_visual_review='PENDING_AFTER_RUN',
        source_advisories=dict(raster='300dpi PNG requested; TIFF not required; dpi is set by reused export_figure',
            dimensions='Report figures use readable physical widths; no journal column-width submission claim',
            uncertainty='Each seed is shown separately; no mean or uncertainty across training seeds is claimed'),
        note='Source checks, rendered axes alignment at 1.5pt, PDF glyph floor5pt and strict collision geometry; not human acceptance'))
    if failed:
        raise RuntimeError('Figure audit failed or requires review; see automatic_audit.json. Training is not restarted.')


def render(config,runs,out):
    if (out/'figure_manifest.json').exists() or any(out.glob('*.png')):
        raise FileExistsError('Use a new plot attempt; never overwrite completed figure exports')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve
    matplotlib.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    auditor = ROOT/'main_model/figure_audit/audit_panel_alignment.py'
    if not auditor.exists():
        raise FileNotFoundError('Bundled panel alignment auditor missing; no silent unaudited export')
    stats,detail = {},{}
    records = []
    for name,folder in runs.items():
        path = ROOT/folder
        metric,info = metrics_for_run(ROOT/config['dev_file'],path,config)
        stats[name],detail[name] = metric,info
        a,s = info['arguments'],info['summary']
        records.append(dict(run=name,length=a['max_length'],seed=a['seed'],steps=s['optimizer_updates'],**metric,
            active_training_seconds=s['training_seconds'],updates_per_second=s['optimizer_updates']/s['training_seconds'],
            peak_allocated_mib=s['peak_allocated_mib'],peak_reserved_mib=s['peak_reserved_mib'],
            before_save_inference_seconds=s.get('before_save_inference_seconds'),
            reload_inference_seconds=s.get('independent_reload_and_inference_seconds')))
    csv_out(out/'metrics_resources.csv',records)
    names = list(stats)
    labels = [name.replace('L','',1).replace('_s',' / s').replace('_u',' / ') for name in names]
    x = np.arange(len(names))
    colors = ['#487A9B','#BD795A','#6F8B6C','#8975A6','#A18C55','#587C80']
    note = 'internal-v2 dev | threshold 0.5 | *temporary short: <100 whitespace words'
    fig_names = []
    def save(fig,name):
        require_matplotlib_panel_alignment(fig,json_out=out/(name+'.alignment.json'),strict=True,
                                          tolerance_pt=1.5,gutter_tolerance_pt=1.5)
        fig.set_layout_engine('none')
        export_figure(fig,out,name)
        plt.close(fig)
        fig_names.append(name)
        dump(out/'figure_index.json',dict(figures=fig_names,manual_visual_review='PENDING_AFTER_RUN',
            metadata='Labels = length / seed / effective updates; no test or official scores',
            formats=['300dpi PNG','editable PDF','editable SVG']))
    def bars(ax,keys):
        width=.72/len(keys)
        for j,(key,label) in enumerate(keys):
            ax.bar(x+(j-(len(keys)-1)/2)*width,[stats[n][key] for n in names],width,color=colors[j],label=label)
        ax.set_xticks(x,labels,rotation=30,ha='right',rotation_mode='anchor')
        ax.set(ylim=(0,1.02),ylabel='Score')
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=len(keys),borderaxespad=0)
    fig,axs = plt.subplots(2,1,figsize=(max(9,len(names)*1.1),8),layout='constrained')
    bars(axs[0],[('macro_f1_all','All macro F1'),('macro_f1_short_temporary','Short macro F1*'),('weighted_score_temporary','Temporary S*')])
    bars(axs[1],[('human_recall','Human recall'),('machine_recall','Machine recall')])
    fig.suptitle('Fixed-node detection and class trade-off\n'+note,fontsize=10)
    save(fig,'performance')
    # Final 2000-step models carry paired length comparisons; seeds stay distinct.
    final = [n for n in names if detail[n]['summary']['optimizer_updates']==2000]
    if not final:
        final = names
    for name in final:
        fig,ax = plt.subplots(figsize=(5.8,5.6),layout='constrained')
        cm = detail[name]['cm']
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=cm.max())
        for i in (0,1):
            for j in (0,1):
                ax.text(j,i,f'{cm[i,j]}\n{cm[i,j]/cm[i].sum():.1%}',ha='center',va='center',
                        fontsize=12,color='white' if cm[i,j]>.6*cm.max() else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['human','machine'],yticklabels=['human','machine'],
               xlabel='Predicted',ylabel='True',title=name+' | counts and row percentages')
        fig.suptitle(note,fontsize=8)
        save(fig,'confusion_'+name)
    fig,axs = plt.subplots(1,2,figsize=(11,5.4),layout='constrained')
    curves,calibration = [],[]
    for i,name in enumerate(final):
        info = detail[name]
        fpr,tpr,threshold = roc_curve(info['y'],info['prob'])
        axs[0].plot(fpr,tpr,color=colors[i%len(colors)],label=f'{name}: {stats[name]["roc_auc"]:.4f}')
        curves.extend(dict(run=name,fpr=float(a),tpr=float(b),threshold=float(t) if np.isfinite(t) else 'inf')
                      for a,b,t in zip(fpr,tpr,threshold))
        bins = np.minimum((info['prob']*10).astype(int),9)
        points = []
        for b in range(10):
            mask = bins==b
            if mask.any():
                item=dict(run=name,bin_low=b/10,bin_high=(b+1)/10,samples=int(mask.sum()),
                          mean_probability=float(info['prob'][mask].mean()),observed_fraction=float(info['y'][mask].mean()))
                points.append(item)
                calibration.append(item)
        axs[1].plot([p['mean_probability'] for p in points],[p['observed_fraction'] for p in points],
                    '-o',markersize=3,color=colors[i%len(colors)],label=name)
    for ax in axs:
        ax.plot([0,1],[0,1],ls='--',color='.7',lw=.8)
        ax.set(xlim=(0,1),ylim=(0,1.02))
    axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='A  Ranking')
    axs[1].set(xlabel='Mean p_machine (10 fixed bins)',ylabel='Observed machine fraction',title='B  Reliability (not calibrated)')
    for ax in axs:
        ax.legend(loc='lower right',fontsize=7)
    fig.suptitle('Ranking and probability reliability\n'+note,fontsize=10)
    csv_out(out/'roc.csv',curves)
    csv_out(out/'reliability.csv',calibration)
    save(fig,'roc_reliability')
    fig,axs = plt.subplots(1,2,figsize=(max(11,len(names)*2.3),5.4),layout='constrained')
    for ax,key,title in zip(axs,['log_loss','brier'],['A  Log loss','B  Brier score']):
        ax.bar(x,[stats[n][key] for n in names],color=colors[0])
        ax.set_xticks(x,labels,rotation=30,ha='right',rotation_mode='anchor')
        ax.set(title=title,ylabel='Lower is better',ylim=(0,None))
    fig.suptitle('Probability quality at all measured nodes\n'+note,fontsize=10)
    save(fig,'probability_quality')
    # Deduplicate ONLY byte-equivalent histories from the same seed AND length.
    selected = {}
    for name,info in detail.items():
        a = info['arguments']
        key = (a['max_length'],a['seed'])
        if key not in selected or info['summary']['optimizer_updates']>detail[selected[key]]['summary']['optimizer_updates']:
            selected[key]=name
    loss_source = []
    fig,ax = plt.subplots(figsize=(10,5.8),layout='constrained')
    for i,((length,seed),name) in enumerate(selected.items()):
        folder = ROOT/runs[name]
        path = folder/'training_log.jsonl'
        if not path.exists():
            raise ValueError('Training history missing; cannot silently omit requested loss curve')
        history = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for other,info in detail.items():
            a = info['arguments']
            if (a['max_length'],a['seed'])==(length,seed) and other!=name:
                old = [json.loads(line) for line in (ROOT/runs[other]/'training_log.jsonl').read_text(encoding='utf-8').splitlines()]
                # Timing is operational and need not be byte-identical across saved copies.
                if any((p['sample_order_hash'],p['loss'])!=(q['sample_order_hash'],q['loss']) for p,q in zip(old,history)) or len(old)>len(history):
                    raise ValueError('Same-seed/length logs are not a verified prefix; refuse to merge curves')
        entries = [e for e in history if not e['skipped']]
        values = np.array([e.get('unweighted_loss',e['loss']) for e in entries])
        if len(values)>=10:
            avg = np.convolve(values,np.ones(10)/10,mode='valid')
            steps = [e['optimizer_step'] for e in entries[9:]]
            ax.plot(steps,avg,color=colors[i%len(colors)],label=f'L{length} seed{seed} (one trajectory)')
        loss_source.extend(dict(length=length,seed=seed,step=e['optimizer_step'],attempt=e['attempt'],
                                skipped=e['skipped'],loss=e['loss'],unweighted_loss=e.get('unweighted_loss',e['loss'])) for e in history)
    ax.set(xlabel='Effective optimizer update',ylabel='Unweighted CE (10-update mean)',title='One curve per length / seed; verified prefixes omitted')
    ax.legend()
    fig.suptitle('Training trajectory\n'+note,fontsize=10)
    csv_out(out/'training_loss.csv',loss_source)
    save(fig,'training_loss')
    fig,axs = plt.subplots(1,2,figsize=(11,5.8),layout='constrained')
    final_records = [r for r in records if r['run'] in final]
    xx = np.arange(len(final_records))
    for ax,key,title in zip(axs,['updates_per_second','peak_allocated_mib'],['A  Active training speed','B  Peak allocated memory']):
        ax.bar(xx,[r[key] for r in final_records],color=colors[0])
        ax.set_xticks(xx,[r['run'] for r in final_records],rotation=25,ha='right',rotation_mode='anchor')
        ax.set(title=title,ylabel='Updates / active second' if key=='updates_per_second' else 'MiB',ylim=(0,None))
    fig.suptitle('Measured training resources; cumulative peaks; same effective endpoint\n'+note,fontsize=10)
    save(fig,'resources')
    dump(out/'figure_manifest.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p))
        for p in out.iterdir() if p.suffix in ('.png','.svg','.pdf','.csv')},manual_visual_review='PENDING_AFTER_RUN'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--runs')
    p.add_argument('--output',required=True)
    p.add_argument('--audit-only',action='store_true')
    a=p.parse_args()
    directory=Path(a.output)
    if a.audit_only:
        try:
            audit(directory)
        except BaseException as exc:
            previous=read_json(directory/'automatic_audit.json') if (directory/'automatic_audit.json').exists() else None
            dump(directory/'automatic_audit.json',dict(status='FAILED_OR_REVIEW_REQUIRED',error=repr(exc),
                audit_details=previous,manual_visual_review='PENDING_AFTER_RUN'))
            raise
    else:
        render(read_json(a.config),read_json(a.runs),directory)
