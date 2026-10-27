import csv,json
from pathlib import Path
from collections import Counter
root=Path('C:/Users/FORYOUCOM/Desktop/Semi-supervised-learning-main/Semi-supervised-learning-main/results')
folder=root/'validation_weak_tail_diagnostic'
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
rows=sum([read(folder/f'seed{s}_online_tail_observations.csv') for s in (1,2)],[])
for r in rows:
    for k in ('target','minor_pred','major_pred','minor_raw_pred','train_source_index','seed','repeat'):r[k]=int(r[k])
    for k in ('minor_conf','major_conf'):r[k]=float(r[k])
    for k in ('correct','raw_correct','major_correct','self_selected','major_to_minor','minor_to_major','minor_goal_conflict'):r[k]=r[k]=='True'
def stats(rr):
    return dict(n=len(rr),correct=sum(r['correct'] for r in rr),self_selected=sum(r['self_selected'] for r in rr),
        wrong_self=sum(r['self_selected'] and not r['correct'] for r in rr),
        wrong_destinations=dict(Counter(r['minor_pred'] for r in rr if not r['correct'])))
wrong=[r for r in rows if r['self_selected'] and not r['correct']]
report={'classes':{str(c):stats([r for r in rows if r['target']==c]) for c in (7,8,9)}}
report['wrong_self_major']={
    'same_wrong':sum(r['major_pred']==r['minor_pred'] for r in wrong),
    'different_wrong':sum(not r['major_correct'] and r['major_pred']!=r['minor_pred'] for r in wrong),
    'major_correct':sum(r['major_correct'] for r in wrong),
    'major_correct_but_no_major_to_minor':sum(r['major_correct'] and not r['major_to_minor'] for r in wrong),
    'conflicting_cross':sum(r['minor_goal_conflict'] for r in wrong),
    'reinforcing_same_wrong_cross':sum(r['major_to_minor'] and r['major_pred']==r['minor_pred'] for r in wrong)}
report['self_confidence_bins']={}
for lo,hi in ((.5,.7),(.7,.9),(.9,.95),(.95,1.000001)):
    rr=[r for r in rows if r['self_selected'] and lo<=r['minor_conf']<hi]
    report['self_confidence_bins'][str((lo,hi))]=stats(rr)
report['correct_self_rejected']=dict(Counter(r['self_reason'] for r in rows if r['correct'] and not r['self_selected']))
report['wrong_self_correction_effect']={
    'changed':sum(r['minor_raw_pred']!=r['minor_pred'] for r in wrong),
    'created_from_correct':sum(r['raw_correct'] for r in wrong)}
report['counterfactual_cross_threshold_true_tail_only']={}
for threshold in (.95,.9,.85):
    sent=[]
    for r in rows:
        eligible=r['minor_pred']>=7 and r['minor_conf']>=threshold
        major_candidate=r['major_pred']<3 and r['major_conf']>=.95
        if eligible and not major_candidate:sent.append(r)
    report['counterfactual_cross_threshold_true_tail_only'][str(threshold)]={'sent':len(sent),'wrong':sum(not r['correct'] for r in sent)}
images=sum([read(folder/f'seed{s}_online_tail_images.csv') for s in (1,2)],[])
stable=[r for r in images if int(r['wrong_self_repeats'])==5]
report['stable_wrong_images']={'seed_image_pairs':len(stable),'unique_source_images':len(set(r['train_source_index'] for r in stable)),
    'classes':dict(Counter(r['target'] for r in stable)),
    'sources_both_seeds':sorted(k for k,v in Counter(r['train_source_index'] for r in stable).items() if v==2)}
report['seed_summary']={str(s):stats([r for r in rows if r['seed']==s]) for s in (1,2)}
report['limits']='Online frozen BN weak-augmentation validation audit, not actual training causality. True-Tail subset cannot estimate global cross precision. Repeat observations are dependent.'
(folder/'failure_evidence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
