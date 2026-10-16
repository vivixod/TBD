import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('/data/2025Zhutaotao/TBD/analysis/small_targets_full_20261016')
rows=list(csv.DictReader((base/'episodes.csv').open()));shots=list(csv.DictReader((base/'shots.csv').open()));loc=list(csv.DictReader((base/'localization.csv').open()))
full=[r for r in rows if r['variant']=='full'];key=lambda r:(r['dataset'],r['class_name'])
lookup={(r['id'],r['variant']):r for r in rows};groups=defaultdict(list)
for r in full:groups[key(r)].append(r)
classes=[('adam','aneurysm'),('btcv2','right_adrenal'),('btcv2','left_adrenal'),('btcv2','portal_vein')]
summary=[];patientrows=[]
for group in classes:
 rr=groups[group]
 if not rr:continue
 out=dict(dataset=group[0],class_name=group[1],n=len(rr),patients=len({r['patient'] for r in rr}),median_gt_area=float(np.median([float(r['gt_area']) for r in rr])))
 for variant in ['full','full_roi64','crop64','full_roi128','crop128']:
  vv=[lookup[r['id'],variant] for r in rr]
  out[variant+'_mean_dice']=float(np.mean([float(r['dice']) for r in vv]));out[variant+'_pooled_dice']=200*sum(int(r['intersection']) for r in vv)/max(sum(int(r['pred_area'])+int(r['gt_area']) for r in vv),1)
 out['full_mean_precision']=float(np.mean([float(r['precision']) for r in rr]));out['full_mean_recall']=float(np.mean([float(r['recall']) for r in rr]));out['zero_overlap_pct']=100*float(np.mean([int(r['intersection'])==0 for r in rr]));out['empty_prediction_pct']=100*float(np.mean([int(r['pred_area'])==0 for r in rr]));out['median_area_ratio']=float(np.median([float(r['area_ratio']) for r in rr]));out['best_shot_mean_dice']=float(np.mean([float(r['best_single_dice']) for r in rr]));out['best_minus_ensemble']=float(np.mean([float(r['best_minus_ensemble']) for r in rr]));out['best_shot_win10_count']=sum(float(r['best_minus_ensemble'])>10 for r in rr)
 ll=[r for r in loc if key(r)==group and r['support_index']=='mean'];valid=[r for r in ll if r['peak_hit']!='']
 out['pseudo_peak_hit_pct']=100*float(np.mean([r['peak_hit']=='True' for r in valid])) if valid else None
 out['pseudo_top1_gt_coverage']=float(np.mean([float(r['top1pct_gt_grid_coverage']) for r in valid])) if valid else None
 for variant in ['crop64','crop128']:
  control='full_roi'+variant[4:]
  out[variant+'_wins_vs_roi']=sum(float(lookup[r['id'],variant]['dice'])>float(lookup[r['id'],control]['dice'])+1e-5 for r in rr)
 for patient in sorted({r['patient'] for r in rr}):
  vv=[r for r in rr if r['patient']==patient]
  row=dict(dataset=group[0],class_name=group[1],patient=patient,n=len(vv))
  for v in ['full','full_roi64','crop64','full_roi128','crop128']:
   row[v]=float(np.mean([float(lookup[r['id'],v]['dice']) for r in vv]))
  patientrows.append(row)
 summary.append(out)
print(json.dumps(summary,indent=2));print('PATIENTS',json.dumps(patientrows,indent=2))
with (base/'diagnostic_summary.json').open('w') as f:json.dump({'summary':summary,'patients':patientrows},f,indent=2)
for path,rr in [(base/'diagnostic_summary.csv',summary),(base/'patients.csv',patientrows)]:
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
names=['ADAM aneurysm','R. adrenal','L. adrenal','Portal vein'];x=np.arange(len(summary));variants=['full','full_roi64','crop64','full_roi128','crop128'];colors=['#777','#a8c4cf','#247fae','#e9c296','#cd832f']
for i,(v,c) in enumerate(zip(variants,colors)):
 axes[0,0].bar(x+(i-2)*.15,[r[v+'_mean_dice'] for r in summary],width=.14,label=v,color=c)
axes[0,0].set_xticks(x,names);axes[0,0].set_ylabel('Mean episode Dice (%)');axes[0,0].set_title('GT-guided crop diagnostics');axes[0,0].legend(fontsize=8,ncol=2);axes[0,0].grid(axis='y',alpha=.2)
axes[0,1].bar(x-.17,[r['full_mean_precision'] for r in summary],width=.32,label='Precision');axes[0,1].bar(x+.17,[r['full_mean_recall'] for r in summary],width=.32,label='Recall');axes[0,1].set_xticks(x,names);axes[0,1].set_ylabel('Mean (%)');axes[0,1].set_title('Full-image prediction');axes[0,1].legend();axes[0,1].grid(axis='y',alpha=.2)
axes[1,0].bar(x-.17,[r['full_mean_dice'] for r in summary],width=.32,label='Mean logits ensemble');axes[1,0].bar(x+.17,[r['best_shot_mean_dice'] for r in summary],width=.32,label='Best single support (oracle)');axes[1,0].set_xticks(x,names);axes[1,0].set_ylabel('Mean Dice (%)');axes[1,0].set_title('Support aggregation diagnostic');axes[1,0].legend(fontsize=8);axes[1,0].grid(axis='y',alpha=.2)
for group,label,c in zip(classes,names,['#aa5555','#4c90b8','#65a785','#bc873e']):
 rr=groups[group]
 axes[1,1].scatter([float(r['gt_area']) for r in rr],[float(r['dice']) for r in rr],label=label,c=c,alpha=.8)
axes[1,1].set_xscale('log');axes[1,1].set_xlabel('GT pixels in 512 input');axes[1,1].set_ylabel('Full-image Dice (%)');axes[1,1].set_title('Size vs segmentation');axes[1,1].legend(fontsize=8);axes[1,1].grid(alpha=.2)
fig.suptitle('Fixed unique queries; oracle crops and best-shot scores are diagnostics only',fontsize=13)
fig.savefig(base/'diagnostic_summary.png',dpi=180)
