"""Frozen held-out weak-augmentation audit; no gradients or parameter updates."""
import ast
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

BASE = Path('C:/Users/FORYOUCOM/Desktop')
MAIN = BASE/'Semi-supervised-learning-main/Semi-supervised-learning-main'
CODE = BASE/'Semi-supervised-learning-upload-worktree/dual real last/dual real last-2'
sys.path.insert(0, str(CODE))
from models.wideresnet import build_wideresnet
from lla import LearnableLogitAdjustment
from correction_gate import correction_gate
from dataset.cifar_val_split import stratified_train_val_split

def main():
    torch.set_num_threads(2)
    out = MAIN/'results/validation_weak_tail_diagnostic'
    out.mkdir(exist_ok=True)
    source = CODE/'train_head_tail_exchange_sampler_alpha_val_split.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    env = dict(torch=torch,math=math,np=np)
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
         and n.name in {'classwise_topk_mask','checked_class_counts'}],type_ignores=[]),str(source),'exec'),env)
    args = SimpleNamespace(num_classes=10,val_samples_per_class=500,classwise_quota='global',
        major_top_ratio=.2,minority_threshold_gamma=1.,min_threshold=.5)
    counts = torch.tensor([1612,967,580,348,209,124,74,44,26,16],device='cuda')
    base = datasets.CIFAR10(str(MAIN/'dual/data'),train=True,download=False)
    normalize = [transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.2471,.2435,.2616))]
    clean = transforms.Compose(normalize)
    weak = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32,padding=4,padding_mode='reflect')]+normalize)
    major=build_wideresnet(28,2,0,10).cuda().eval()
    minor=build_wideresnet(28,2,0,10).cuda().eval()
    lla=LearnableLogitAdjustment(10).cuda().eval()
    summaries=[]; provenance=[]
    for seed in (1,2):
        args.seed=seed
        train_ids,val_ids=stratified_train_val_split(base.targets,args)
        assert not np.intersect1d(train_ids,val_ids).size
        assert len(val_ids)==5000
        run=MAIN/f'results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed{seed}'
        checkpoint=run/'checkpoint.pth.tar'
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ckpt['epoch']==50 and ckpt['variant']=='dual_real_last_2_minor_lla_hard'
        lla.load_state_dict(ckpt['minor_lla'])
        labels=np.asarray(base.targets)[val_ids]
        old=list(csv.DictReader((run/'validation_disagreement/epoch_050/samples.csv').open(encoding='utf-8-sig')))
        assert np.array_equal(labels,np.array([int(r['target']) for r in old]))
        provenance.append(dict(seed=seed,checkpoint=str(checkpoint),sha256=digest,validation_indices_sha256=hashlib.sha256(val_ids.astype('<i8').tobytes()).hexdigest()))
        for mode in ('ema','online'):
            major.load_state_dict(ckpt['ema_state_dict_major' if mode=='ema' else 'state_dict_major'])
            minor.load_state_dict(ckpt['ema_state_dict_minor' if mode=='ema' else 'state_dict_minor'])
            all_rows=[]
            with torch.inference_mode():
                if mode=='ema':
                    clean_pred=[]
                    for start in range(0,5000,224):
                        batch=torch.stack([clean(Image.fromarray(base.data[j])) for j in val_ids[start:start+224]]).cuda()
                        clean_pred.extend(lla.pseudo_labels(minor(batch)).argmax(-1).cpu().tolist())
                    assert clean_pred==[int(r['minor_adjusted_pred']) for r in old], 'Clean baseline did not reproduce'
                for repeat in range(5):
                    aug_seed=10000+seed*100+repeat
                    random.seed(aug_seed);np.random.seed(aug_seed);torch.manual_seed(aug_seed)
                    order=np.random.RandomState(aug_seed).permutation(5000)
                    for start in range(0,5000,224):
                        ids=order[start:start+224]
                        x=torch.stack([weak(Image.fromarray(base.data[val_ids[i]])) for i in ids]).cuda()
                        zm=major(x); zn=minor(x)
                        cm,pm=zm.softmax(-1).max(-1)
                        cr,pr=zn.softmax(-1).max(-1)
                        cn,pn=lla.pseudo_labels(zn).max(-1)
                        self_m=cm.ge(.95)
                        baseline=env['classwise_topk_mask'](cn,pn,counts,args,.5).bool()
                        cand_m=cm.ge(.95)&pm.lt(3);cand_n=cn.ge(.95)&pn.ge(7)
                        conflict=cand_m&cand_n
                        send_m=cand_m&~conflict;send_n=cand_n&~conflict
                        gate=correction_gate(pm,pr,pn,pm.lt(3),pn.ge(7),send_m)
                        self_n=baseline&~gate
                        packed=torch.stack([pm,pr,pn,cm,cr,cn,self_m,baseline,self_n,send_m,send_n,gate,conflict],dim=1).cpu().tolist()
                        for i,v in zip(ids,packed):
                            m,r,n=int(v[0]),int(v[1]),int(v[2]); y=int(labels[i])
                            if y<7:continue
                            sm,bn,sn,sendm,sendn,g,c=map(bool,v[6:])
                            reason='sent' if sendn else 'predicted_non_tail' if n<7 else 'confidence_below_0.95' if v[5]<.95 else 'cross_conflict'
                            self_reason='selected' if sn else 'confidence_below_0.5' if v[5]<.5 else 'gate_suppressed' if g else 'quota'
                            all_rows.append(dict(seed=seed,mode=mode,repeat=repeat,validation_index=int(i),train_source_index=int(val_ids[i]),target=y,
                                major_pred=m,minor_raw_pred=r,minor_pred=n,major_conf=v[3],minor_raw_conf=v[4],minor_conf=v[5],
                                correct=n==y,raw_correct=r==y,major_correct=m==y,self_selected=sn,self_reason=self_reason,
                                cross_reason=reason,minor_to_major=sendn,major_to_minor=sendm,
                                minor_goal_conflict=sn and sendm and m!=n,major_goal_conflict=sm and sendn and m!=n))
                    print(f'seed={seed} mode={mode} repeat={repeat+1}/5',flush=True)
            name=f'seed{seed}_{mode}'
            with (out/f'{name}_tail_observations.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
            per_image=defaultdict(list)
            for r in all_rows:per_image[r['validation_index']].append(r)
            image_rows=[]
            for idx,rr in per_image.items():
                image_rows.append(dict(validation_index=idx,train_source_index=rr[0]['train_source_index'],target=rr[0]['target'],
                    predictions=','.join(str(r['minor_pred']) for r in rr),correct_repeats=sum(r['correct'] for r in rr),
                    wrong_self_repeats=sum(r['self_selected'] and not r['correct'] for r in rr),
                    cross_sent_repeats=sum(r['minor_to_major'] for r in rr),prediction_changes=len(set(r['minor_pred'] for r in rr))>1))
            with (out/f'{name}_tail_images.csv').open('w',newline='',encoding='utf-8-sig') as f:
                w=csv.DictWriter(f,fieldnames=list(image_rows[0]));w.writeheader();w.writerows(image_rows)
            s=dict(seed=seed,mode=mode,n=len(all_rows),correct=sum(r['correct'] for r in all_rows),
                cross_reasons=dict(Counter(r['cross_reason'] for r in all_rows)),self_reasons=dict(Counter(r['self_reason'] for r in all_rows)),
                wrong_self=sum(r['self_selected'] and not r['correct'] for r in all_rows),
                wrong_self_major_correct=sum(r['self_selected'] and not r['correct'] and r['major_correct'] for r in all_rows),
                wrong_self_created_by_correction=sum(r['self_selected'] and not r['correct'] and r['raw_correct'] for r in all_rows),
                wrong_cross=sum(r['minor_to_major'] and not r['correct'] for r in all_rows),
                minor_goal_conflicts=sum(r['minor_goal_conflict'] for r in all_rows),major_goal_conflicts=sum(r['major_goal_conflict'] for r in all_rows),
                unstable_images=sum(r['prediction_changes'] for r in image_rows),
                always_wrong_self_images=sum(r['wrong_self_repeats']==5 for r in image_rows),
                per_class={str(c):dict(correct=sum(r['correct'] for r in all_rows if r['target']==c),
                    wrong_self=sum(r['self_selected'] and not r['correct'] for r in all_rows if r['target']==c),
                    wrong_destinations=dict(Counter(r['minor_pred'] for r in all_rows if r['target']==c and not r['correct']))) for c in (7,8,9)})
            summaries.append(s)
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    report=dict(provenance=provenance,repeats=5,batch_size=224,weights_frozen=True,
        clean_baseline_reproduced=True,model_mode='eval (frozen BN); online and EMA weights separately',
        limitation='Not a historical training trace: no strong view or loss/backprop; no train-mode BN; balanced held-out batches. Final short batch retained.',summaries=summaries)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
