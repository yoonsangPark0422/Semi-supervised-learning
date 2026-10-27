"""Post-hoc frozen DeCon test diagnostic, not selection of a new test setting."""
import sys,json
from pathlib import Path
import torch
from torchvision import datasets,transforms
BASE=Path('C:/Users/FORYOUCOM/Desktop')
MAIN=BASE/'Semi-supervised-learning-main/Semi-supervised-learning-main'
sys.path.insert(0,str(BASE/'DeCon'))
from models.wideresnet import build_wideresnet
def main():
    torch.set_num_threads(2)
    dataset=datasets.CIFAR10(str(MAIN/'dual/data'),train=False,download=False,transform=transforms.Compose([
        transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.2471,.2435,.2616))]))
    loader=torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False,num_workers=0)
    model=build_wideresnet(28,2,0,10).cuda().eval()
    reports={}
    for experiment,expected in [('three_models_matched_fresh_cifar10_seed1',9072),('reduced10pct_decon_real_last2_cifar10_seed1',5385)]:
        checkpoint=MAIN/'results'/experiment/'decon/checkpoint.pth.tar'
        ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert ckpt['epoch']==50
        model.load_state_dict(ckpt['ema_state_dict'])
        qbal=ckpt['qbal'].cuda();qhat=ckpt['qhat']
        matrices={k:torch.zeros((10,10),dtype=torch.long) for k in ['standard_head','balanced_raw','balanced_adjusted']}
        flips={'wrong_to_right':0,'right_to_wrong':0,'changed':0}
        with torch.inference_mode():
            for x,y in loader:
                feature=model(x.cuda());raw=model.classify1(feature)
                predictions={'standard_head':model.classify(feature).argmax(-1).cpu(),
                    'balanced_raw':raw.argmax(-1).cpu(),
                    'balanced_adjusted':(raw-2*qbal.log()).argmax(-1).cpu()}
                for key,p in predictions.items():matrices[key]+=torch.bincount(y*10+p,minlength=100).reshape(10,10)
                a=predictions['balanced_raw'];b=predictions['balanced_adjusted']
                flips['changed']+=int((a!=b).sum());flips['wrong_to_right']+=int(((a!=y)&(b==y)).sum());flips['right_to_wrong']+=int(((a==y)&(b!=y)).sum())
        print('reproduction',experiment,int(matrices['balanced_adjusted'].trace()),expected,flush=True)
        assert int(matrices['balanced_adjusted'].trace())==expected
        report={'qhat':qhat.flatten().tolist(),'qbal':qbal.cpu().flatten().tolist(),'correction_flips':flips,'outputs':{}}
        for key,c in matrices.items():
            report['outputs'][key]={'accuracy':float(c.trace())/100,'per_class_recall':(c.diag().float()/10).tolist(),
                'predicted_counts':c.sum(0).tolist(),'confusion':c.tolist()}
        reports[experiment]=report
    out=MAIN/'results/reduced10pct_decon_real_last2_cifar10_seed1/decon_diagnostic.json'
    out.write_text(json.dumps(reports,indent=2),encoding='utf-8')
    for key,r in reports.items():
        print(key,flush=True)
        print(json.dumps({k:v for k,v in r.items() if k!='outputs'}))
        for name,o in r['outputs'].items():print(name,json.dumps({k:v for k,v in o.items() if k!='confusion'}))
        print('adjusted truck confusion',r['outputs']['balanced_adjusted']['confusion'][9])
if __name__=='__main__':main()
