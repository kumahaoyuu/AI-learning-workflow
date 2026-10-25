"""Training-only exploration for the third version. Never reads MNIST test files."""
from pathlib import Path
import sys,json,argparse,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from history.previous import optimize as previous
from naive_bayes import deskew_pixels

OUT=ROOT/'experiment-cache'
PARENTS=(None,0,1,None,3,4,None,6,7,None,9,10)


def resize_box(X,size=21,threshold=64,keep_aspect=False,canvas=None,center_mass=False):
    """Deterministic gray bilinear box sampling, optionally onto a 28x28 canvas."""
    output=np.empty((len(X),canvas or size,canvas or size),dtype=np.float32)
    for start in range(0,len(X),512):
        raw=X[start:start+512].reshape(-1,28,28).astype(np.float64)
        mask=raw>threshold
        rows,cols=mask.any(2),mask.any(1)
        nonempty=rows.any(1)
        top,left=rows.argmax(1),cols.argmax(1)
        height=28-top-rows[:,::-1].argmax(1)
        width=28-left-cols[:,::-1].argmax(1)
        oh=np.full(len(raw),size)
        ow=np.clip(np.rint(size*width/height),1,(canvas or size)-2).astype(int) if keep_aspect else oh
        side=canvas or size
        oy=(side-oh)//2 if canvas else np.zeros(len(raw),int)
        ox=(side-ow)//2 if canvas else np.zeros(len(raw),int)
        yy=np.arange(side)[None,:,None]
        xx=np.arange(side)[None,None,:]
        sy=top[:,None,None]+np.clip((yy-oy[:,None,None]+0.5)*height[:,None,None]/oh[:,None,None]-0.5,0,height[:,None,None]-1)
        sx=left[:,None,None]+np.clip((xx-ox[:,None,None]+0.5)*width[:,None,None]/ow[:,None,None]-0.5,0,width[:,None,None]-1)
        y0,x0=np.floor(sy).astype(int),np.floor(sx).astype(int)
        y1,x1=np.minimum(y0+1,27),np.minimum(x0+1,27)
        fy,fx=sy-y0,sx-x0
        ids=np.arange(len(raw))[:,None,None]
        sampled=(raw[ids,y0,x0]*(1-fx)*(1-fy)+raw[ids,y0,x1]*fx*(1-fy)
                +raw[ids,y1,x0]*(1-fx)*fy+raw[ids,y1,x1]*fx*fy)
        valid=(yy>=oy[:,None,None])&(yy<(oy+oh)[:,None,None])&(xx>=ox[:,None,None])&(xx<(ox+ow)[:,None,None])
        sampled*=valid&nonempty[:,None,None]
        if center_mass:
            mass=sampled.sum((1,2))
            denom=np.maximum(mass,1)
            cy=(sampled*yy).sum((1,2))/denom
            cx=(sampled*xx).sum((1,2))/denom
            sy=yy+cy[:,None,None]-(side-1)/2
            sx=xx+cx[:,None,None]-(side-1)/2
            y0,x0=np.floor(sy).astype(int),np.floor(sx).astype(int)
            y1,x1=y0+1,x0+1
            fy,fx=sy-y0,sx-x0
            recentered=np.zeros_like(sampled)
            for iy,wy in ((y0,1-fy),(y1,fy)):
                for ix,wx in ((x0,1-fx),(x1,fx)):
                    recentered+=sampled[ids,np.clip(iy,0,side-1),np.clip(ix,0,side-1)]*wy*wx*((iy>=0)&(iy<side)&(ix>=0)&(ix<side))
            sampled=recentered
        output[start:start+len(raw)]=sampled
    return output


def blur(images):
    padded=np.pad(images,((0,0),(1,1),(1,1)))
    out=np.zeros_like(images)
    for dy,wy in ((0,1),(1,2),(2,1)):
        for dx,wx in ((0,1),(1,2),(2,1)):
            out+=padded[:,dy:dy+images.shape[1],dx:dx+images.shape[2]]*(wy*wx/16)
    return out


def grad_hist(images,rows,cols,n_angles=4,soft=True):
    """Fixed unsigned gradient orientation mass in explicitly located regions."""
    X=images.astype(np.float32)/255
    gx=np.zeros_like(X);gy=np.zeros_like(X)
    gx[:,:,1:-1]=(X[:,:,2:]-X[:,:,:-2])/2
    gy[:,1:-1,:]=(X[:,2:,:]-X[:,:-2,:])/2
    mag=np.sqrt(gx*gx+gy*gy)
    angle=np.mod(np.arctan2(gy,gx),np.pi)*n_angles/np.pi
    low=np.floor(angle).astype(np.int64)%n_angles
    fraction=angle-np.floor(angle)
    hist=[]
    rb=np.linspace(0,X.shape[1],rows+1,dtype=int)
    cb=np.linspace(0,X.shape[2],cols+1,dtype=int)
    for r in range(rows):
        for c in range(cols):
            sl=(slice(None),slice(rb[r],rb[r+1]),slice(cb[c],cb[c+1]))
            columns=[]
            for d in range(n_angles):
                weights=(low[sl]==d).astype(float)
                if soft:
                    weights=weights*(1-fraction[sl])+((low[sl]+1)%n_angles==d)*fraction[sl]
                columns.append((mag[sl]*weights).sum((1,2)))
            hist.append(np.stack(columns,axis=1))
    return np.stack(hist,axis=1)


def hog_features(images,rows=2,cols=2,n_angles=4,soft=True):
    h=grad_hist(images,rows,cols,n_angles,soft)
    total=h.sum(2,keepdims=True)
    norm=np.divide(h,total,out=np.zeros_like(h),where=total>1e-12)
    return norm[:,:,:-1].reshape(len(images),-1)


def topology(X,size=21):
    """Original aspect and top/bottom hole fractions, with a consistent box map."""
    return previous.continuous_features(X,'normalized','holes')[:,9:]


def regional_features(images,rows,cols):
    rb=np.linspace(0,images.shape[1],rows+1,dtype=int)
    cb=np.linspace(0,images.shape[2],cols+1,dtype=int)
    return np.column_stack([images[:,rb[r]:rb[r+1],cb[c]:cb[c+1]].mean((1,2))/255
                            for r in range(rows) for c in range(cols)])


def evaluate_features(name,features,y,train,valid,modes=('uniform','quantile'),edges=None):
    assert features.shape==(len(y),12),features.shape
    np.save(OUT/(name+'.npy'),features.astype(np.float64))
    records=[]
    for mode in modes:
        if mode=='quantile':
            e=previous.bin_edges(features[train],'quantile','holes')
        else:
            e=np.asarray(edges if edges is not None else [(0.05,0.15,0.3,0.5)]*12)
        states=previous.discretize_features(features,e)
        for alpha in (0.1,1.0):
            pred=previous.scores_from_features(states[train],y[train],states[valid],alpha,PARENTS).argmax(1)
            row={'model':'bayes_net','features':name,'binning':mode,'alpha':alpha,'edges':e.tolist(),**previous.metrics(y[valid],pred)}
            records.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='edges'}),flush=True)
            with (OUT/'bn_results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    return records


def nb_round(X,y,train,valid):
    allrows=[]
    skewed=deskew_pixels(X)
    configurations=[(20,False,False,False),(20,False,True,False),(22,False,False,False),
                    (22,False,True,False),(24,False,True,False),(20,True,True,False),
                    (20,False,True,True),(22,False,True,True)]
    for size,aspect,com,smooth in configurations:
        images=resize_box(skewed,size,64,aspect,28,com)
        if smooth:images=blur(images)
        pixels=images.reshape(-1,784)
        for n_states in (4,8,16):
            counts=previous.pixel_counts(pixels[train],y[train],n_states)
            for alpha in (0.1,1.0):
                pred=previous.pixel_predict(counts,y[train],pixels[valid],alpha,n_states)
                row={'model':'naive_bayes','size':size,'keep_aspect':aspect,'center_mass':com,'blur':smooth,
                     'n_states':n_states,'alpha':alpha,**previous.metrics(y[valid],pred)}
                allrows.append(row)
                with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
    return allrows


def bn_round(X,y,train,valid):
    allrows=[]
    for deskew in (False,True):
        pixels=deskew_pixels(X) if deskew else X
        images=resize_box(pixels,24,64)
        topo=topology(pixels)
        prefix='skew' if deskew else 'raw'
        f=np.column_stack([regional_features(images,3,3),topo])
        edges=[(0.05,0.15,0.3,0.5)]*9+[(0.45,0.6,0.75,0.9),(0.001,0.03,0.1,0.22),(0.001,0.03,0.1,0.22)]
        allrows+=evaluate_features(prefix+'_density_holes',f,y,train,valid,edges=edges)
        for r,c,angles in ((2,2,4),(2,3,3),(3,2,3)):
            for soft in (True,False):
                f=hog_features(images,r,c,angles,soft)
                allrows+=evaluate_features(f'{prefix}_hog_{r}{c}_{angles}_soft{int(soft)}',f,y,train,valid)
        for r,c in ((3,3),):
            h=grad_hist(images,r,c,4,True)
            total=h.sum(2)
            # Continuous dominant angle of the unsigned gradient, via doubled
            # circular moment. A zero-gradient region is assigned angle zero.
            theta=np.arange(4)*np.pi/4
            cosine=(h*np.cos(2*theta)).sum(2)
            sine=(h*np.sin(2*theta)).sum(2)
            dominant=(np.mod(np.arctan2(sine,cosine),2*np.pi)/(2*np.pi))
            f=np.column_stack([dominant,topo])
            edges=[(0.2,0.4,0.6,0.8)]*9+[(0.45,0.6,0.75,0.9),(0.001,0.03,0.1,0.22),(0.001,0.03,0.1,0.22)]
            allrows+=evaluate_features(prefix+'_orientation_holes',f,y,train,valid,edges=edges)
    return allrows


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('round',choices=['nb','bn'])
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output_dir
    OUT.mkdir(parents=True,exist_ok=True)
    X,y=previous.load_training_only(args.data_dir)
    train,valid=previous.validation_split(y)
    before=time.perf_counter()
    rows=nb_round(X,y,train,valid) if args.round=='nb' else bn_round(X,y,train,valid)
    print('BEST',json.dumps(max(rows,key=lambda r:(r['macro_f1'],r['accuracy']))),flush=True)
    print('SECONDS',time.perf_counter()-before,flush=True)
