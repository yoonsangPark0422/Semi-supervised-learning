"""Inventory and losslessly archive available experiment records; no checkpoints/data."""
import csv,gzip,hashlib,io,json,os,re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

DESKTOP=Path('C:/Users/FORYOUCOM/Desktop')
REPO=DESKTOP/'Semi-supervised-learning-upload-worktree'
OUT=REPO/'reports/all-experiments-20261027'
ROOTS={
 'main':DESKTOP/'Semi-supervised-learning-main', 'decon':DESKTOP/'DeCon',
 'thesis':DESKTOP/'졸업논문 코드', 'legacy_machine':DESKTOP/'머신',
 'dual200':DESKTOP/'dual_imb10_ep200','fixmatch':DESKTOP/'FixMatch-pytorch',
 'fixmatch1':DESKTOP/'FixMatch-pytorch1','fixmatch4000':DESKTOP/'fixmatch_4000_imbalance_ratio_100',
 'proposed500':DESKTOP/'proposed_cifar10_4000_balanced_500epoch',
 'onedrive_dual':Path('C:/Users/FORYOUCOM/OneDrive/바탕 화면/대학 자료/머신/dual'),
 'existing_reports':REPO/'reports',
}
SKIP={'.git','.venv-cu128','.venv','venv','__pycache__','node_modules','data','datasets','cifar-10-batches-py','cifar-100-python','all-experiments-20261027'}
SUFFIX={'.csv','.json','.md','.log','.txt','.npz'}
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{30,})')

def read_csv(data):
 for encoding in ('utf-8-sig','cp949','utf-16'):
  try:return list(csv.DictReader(io.StringIO(data.decode(encoding))))
  except (UnicodeError,csv.Error):pass
 return []

def write_csv(path,rows,fields):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
 OUT.mkdir(exist_ok=True)
 manifest=[];excluded=[];dedup={};metrics=[];latest=[];errors=[];source_counts=Counter()
 for alias,root in ROOTS.items():
  if not root.exists():continue
  for parent,dirs,files in os.walk(root):
   dirs[:]=[d for d in dirs if d not in SKIP]
   for name in sorted(files):
    p=Path(parent)/name;rel=p.relative_to(root)
    context=str(p).lower()
    relevant=alias in ('existing_reports','onedrive_dual','dual200','fixmatch4000','proposed500') or any(t in context for t in ('result','report','summary','eval','diagnos','seed','epoch'))
    if not relevant:continue
    if name.endswith(('.pth','.pth.tar','.pt','.ckpt')) or name.startswith('events.out.tfevents'):
     excluded.append(dict(source_root=alias,source_relative=rel.as_posix(),bytes=p.stat().st_size,reason='checkpoint_or_tensorboard_local_only'));continue
    if p.suffix.lower() not in SUFFIX and not (p.suffix=='.py' and ('source' in name)):
     continue
    if p.suffix=='.npz' and name!='shared_indices.npz':continue
    data=p.read_bytes()
    if SECRET.search(data):
     excluded.append(dict(source_root=alias,source_relative=rel.as_posix(),bytes=len(data),reason='credential_pattern_not_published'));continue
    digest=hashlib.sha256(data).hexdigest()
    if digest in dedup:
     dest=dedup[digest];duplicate=True
    else:
     # Short per-directory IDs avoid Windows path-length limits; manifest retains full paths.
     directory_id=hashlib.sha256(rel.parent.as_posix().encode()).hexdigest()[:12]
     dest=OUT/'raw'/alias/directory_id/name
     if dest.suffix=='.log':dest=dest.with_name(dest.name+'.txt')
     if len(data)>250000:dest=dest.with_name(dest.name+'.gz')
     dest.parent.mkdir(parents=True,exist_ok=True)
     dest.write_bytes(gzip.compress(data,compresslevel=6,mtime=0) if dest.suffix=='.gz' else data)
     dedup[digest]=dest;duplicate=False
    record=dict(source_root=alias,source_relative=rel.as_posix(),source_bytes=len(data),sha256=digest,
                archived_file=dest.relative_to(OUT).as_posix(),compressed=dest.suffix=='.gz',deduplicated=duplicate)
    manifest.append(record);source_counts[alias]+=1
    if name not in ('train_metrics.csv','per_class_eval.csv','evaluation_metrics.csv'):continue
    rows=read_csv(data)
    if not rows:continue
    def epoch(row):
     try:return int(float(row.get('epoch','0')))
     except (ValueError,TypeError):return 0
    e=max(map(epoch,rows));end=[r for r in rows if epoch(r)==e]
    run=rel.parent.as_posix();key=f'{alias}/{run}'
    planned=re.findall(r'(?:^|[_/])(?:e|ep)(\d+)(?:[_/]|$)',run)
    planned=int(planned[-1]) if planned else None
    status=('reached_planned_epoch' if e>=planned else 'partial') if planned else 'latest_record_only'
    common=dict(run=key,last_epoch=e,planned_epoch=planned or '',status=status,source=record['archived_file'])
    latest.append(dict(common,metric_file=name))
    if name=='per_class_eval.csv':
     groups={}
     for r in end:groups.setdefault(r.get('model','unknown'),[]).append(r)
     for model,g in groups.items():
      try:
       supports=[float(r['support']) for r in g]
       accuracy=sum(float(r['accuracy'])*s for r,s in zip(g,supports))/sum(supports)
       f1=sum(float(r['f1']) for r in g)/len(g)
       total=sum(supports)
       scope=('official_test' if 'official_test' in run or 'official-test' in run or 'three_models_matched_fresh' in run or 'reduced10pct' in run
              else 'validation' if 'validation' in run or 'valsplit' in run or (len(g)==10 and all(s==500 for s in supports)) else 'unknown')
       groupscores={}
       for group,cs in [('head',(0,1,2)),('middle',(3,4,5,6)),('tail',(7,8,9))]:
        selected=[r for r in g if int(r['class']) in cs]
        groupscores[group+'_recall']=sum(float(r['accuracy']) for r in selected)/len(selected) if len(g)==10 and selected else ''
       metrics.append(dict(common,model=model,split=scope,n_eval=total,accuracy=accuracy,macro_f1=f1,**groupscores))
      except (ValueError,KeyError,ZeroDivisionError) as exc:errors.append({'run':key,'error':str(exc)})
    if len(manifest)%300==0:print('archived',len(manifest),flush=True)
 write_csv(OUT/'artifact_manifest.csv',manifest,['source_root','source_relative','source_bytes','sha256','archived_file','compressed','deduplicated'])
 write_csv(OUT/'local_only_artifacts.csv',excluded,['source_root','source_relative','bytes','reason'])
 write_csv(OUT/'run_inventory.csv',latest,['run','metric_file','last_epoch','planned_epoch','status','source'])
 write_csv(OUT/'final_metrics.csv',metrics,['run','model','split','last_epoch','planned_epoch','status','n_eval','accuracy','macro_f1','head_recall','middle_recall','tail_recall','source'])
 (OUT/'parse_notes.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
 counts=dict(source_files=len(manifest),unique_archived_files=len(dedup),deduplicated=len(manifest)-len(dedup),
             unique_metric_runs=len(set(r['run'] for r in latest)),model_metric_rows=len(metrics),
             raw_source_bytes=sum(r['source_bytes'] for r in manifest),archived_bytes=sum(p.stat().st_size for p in dedup.values()),
             source_counts=dict(source_counts),local_only_count=len(excluded),parse_notes=len(errors))
 (OUT/'archive_stats.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
 lines=['# 전체 모델별 최종 기록','', '분할·학습 조건이 다른 행을 직접 순위 비교하지 마세요. `unknown`은 근거가 부족한 분할입니다. 마지막 기록이며 완료를 보장하지 않습니다.','',
        '| 실행 | 모델 | 평가 | epoch | 정확도 | Macro-F1 | 원본 |','|---|---|---|---:|---:|---:|---|']
 for r in metrics:
  lines.append(f"| {r['run']} | {r['model']} | {r['split']} | {r['last_epoch']} | {r['accuracy']:.2f} | {r['macro_f1']:.2f} | [파일]({quote(r['source'],safe='/')}) |")
 (OUT/'CATALOG.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 # Verify every archived file against original-byte SHA256, once per content hash.
 for digest,p in dedup.items():
  data=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
  assert hashlib.sha256(data).hexdigest()==digest,str(p)
 print(json.dumps(counts),flush=True)

if __name__=='__main__':main()
