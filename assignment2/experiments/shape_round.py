"""Second predeclared BN feature bank; fixed four three-node chains throughout."""
from pathlib import Path
import sys,json,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import explore as e


def local_moments(images,rows,cols,kind):
    rb=np.linspace(0,images.shape[1],rows+1,dtype=int)
    cb=np.linspace(0,images.shape[2],cols+1,dtype=int)
    features=[];edges=[]
    for r in range(rows):
        for c in range(cols):
            patch=images[:,rb[r]:rb[r+1],cb[c]:cb[c+1]].astype(float)/255
            yy,xx=np.indices(patch.shape[1:],dtype=float)
            xx/=max(patch.shape[2]-1,1);yy/=max(patch.shape[1]-1,1)
            total=patch.sum((1,2));safe=np.maximum(total,1e-12)
            cx=(patch*xx).sum((1,2))/safe;cy=(patch*yy).sum((1,2))/safe
            density=patch.mean((1,2))
            if kind=='centroids':
                features.extend([density,cx,cy]);edges.extend([(0.05,0.15,0.3,0.5),(0.2,0.4,0.6,0.8),(0.2,0.4,0.6,0.8)])
            else:
                dx=xx-cx[:,None,None];dy=yy-cy[:,None,None]
                vx=(patch*dx*dx).sum((1,2))/safe;vy=(patch*dy*dy).sum((1,2))/safe
                cov=(patch*dx*dy).sum((1,2))/safe
                anis=np.divide(vx-vy,vx+vy,out=np.zeros_like(vx),where=vx+vy>1e-12)
                corr=np.divide(cov,np.sqrt(vx*vy),out=np.zeros_like(vx),where=vx*vy>1e-12)
                features.extend([density,anis,corr]);edges.extend([(0.05,0.15,0.3,0.5),(-0.6,-0.2,0.2,0.6),(-0.6,-0.2,0.2,0.6)])
    return np.column_stack(features),edges


def strip_features(images,topo):
    bounds=np.linspace(0,images.shape[1],4,dtype=int)
    features=[];edges=[]
    for a,b in zip(bounds,bounds[1:]):
        patch=images[:,a:b,:].astype(float)/255
        features.extend([patch[:,:,:patch.shape[2]//2].mean((1,2)),patch[:,:,patch.shape[2]//2:].mean((1,2))])
        yy,xx=np.indices(patch.shape[1:],dtype=float)
        xx/=max(patch.shape[2]-1,1);yy/=max(patch.shape[1]-1,1)
        total=np.maximum(patch.sum((1,2)),1e-12)
        cx=(patch*xx).sum((1,2))/total;cy=(patch*yy).sum((1,2))/total
        dx=xx-cx[:,None,None];dy=yy-cy[:,None,None]
        cov=(patch*dx*dy).sum((1,2));vx=(patch*dx*dx).sum((1,2));vy=(patch*dy*dy).sum((1,2))
        corr=np.divide(cov,np.sqrt(vx*vy),out=np.zeros_like(vx),where=vx*vy>1e-12)
        features.append(corr)
        edges.extend([(0.05,0.15,0.3,0.5),(0.05,0.15,0.3,0.5),(-0.6,-0.2,0.2,0.6)])
    features.extend(topo.T)
    edges.extend([(0.45,0.6,0.75,0.9),(0.001,0.03,0.1,0.22),(0.001,0.03,0.1,0.22)])
    return np.column_stack(features),edges


def run_shapes(X,y,train,valid):
    allrows=[]
    (e.OUT/'shape_protocol.json').write_text(json.dumps({
      'declared_before_round':True,'graph':list(e.PARENTS),
      'families':['4x3 or 3x4 regional densities','3x3 densities plus aspect/holes',
                  'four quadrant density and centroids','four quadrant density and central moments',
                  'three strips left/right density and correlation plus aspect/holes'],
      'geometry':['square box 24','square box 24 on 28 canvas, mass-centered'],
      'deskew':[False,True], 'binning':['fixed semantic thresholds','training-only five-state quantiles'],
      'test_files_read':False},indent=2)+'\n',encoding='utf-8')
    for skew in (False,True):
        pixels=e.deskew_pixels(X) if skew else X
        topo=e.topology(pixels)
        for com in (False,True):
            images=e.resize_box(pixels,24,64,False,28 if com else None,com)
            prefix=f'shape_skew{int(skew)}_com{int(com)}'
            for r,c in ((4,3),(3,4)):
                f=e.regional_features(images,r,c)
                allrows+=e.evaluate_features(f'{prefix}_density{r}{c}',f,y,train,valid)
            f=np.column_stack([e.regional_features(images,3,3),topo])
            edges=[(0.05,0.15,0.3,0.5)]*9+[(0.45,0.6,0.75,0.9),(0.001,0.03,0.1,0.22),(0.001,0.03,0.1,0.22)]
            allrows+=e.evaluate_features(prefix+'_density_holes',f,y,train,valid,edges=edges)
            for kind in ('centroids','central'):
                f,edges=local_moments(images,2,2,kind)
                allrows+=e.evaluate_features(prefix+'_'+kind,f,y,train,valid,edges=edges)
            f,edges=strip_features(images,topo)
            allrows+=e.evaluate_features(prefix+'_strips',f,y,train,valid,edges=edges)
    return allrows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=e.OUT)
    args=parser.parse_args()
    e.OUT=args.output_dir
    e.OUT.mkdir(parents=True,exist_ok=True)
    X,y=e.previous.load_training_only(args.data_dir)
    train,valid=e.previous.validation_split(y)
    run_shapes(X,y,train,valid)
