"""Generic AP50 evaluator. No hidden labels are embedded."""
import numpy as np
import pandas as pd
PRED_COLUMNS=['image_id','class_id','confidence','x_min','y_min','x_max','y_max']
BOX_COLUMNS=['x_min','y_min','x_max','y_max']
def validate(pred,images):
    if list(pred.columns)!=PRED_COLUMNS:raise ValueError('Wrong columns/order.')
    if pred.image_id.isna().any() or not pred.image_id.isin(images.image_id).all():raise ValueError('Unknown/missing image ID.')
    vals=pred[PRED_COLUMNS[1:]].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(vals).all():raise ValueError('Non-finite/non-numeric prediction.')
    if not pred.class_id.isin([0]).all():raise ValueError('Only class_id 0 is allowed.')
    if not pred.confidence.between(0,1).all():raise ValueError('Confidence must be in [0,1].')
    if pred.groupby('image_id').size().gt(300).any():raise ValueError('At most 300 predictions/image.')
    if pred.duplicated(PRED_COLUMNS).any():raise ValueError('Exact duplicate detection rows.')
    dims=images.set_index('image_id').loc[pred.image_id] if len(pred) else images.iloc[:0]
    x=pred[BOX_COLUMNS].to_numpy(float)
    if len(x) and not ((x[:,0]>=0)&(x[:,1]>=0)&(x[:,2]>x[:,0])&(x[:,3]>x[:,1])&(x[:,2]<=dims.width.to_numpy())&(x[:,3]<=dims.height.to_numpy())).all():raise ValueError('Invalid/out-of-bounds box.')

def overlap(box,others,ioa=False):
    if not len(others):return np.empty(0)
    lt=np.maximum(box[:2],others[:,:2]);rb=np.minimum(box[2:],others[:,2:]);inter=np.maximum(0,rb-lt).prod(1)
    area=(box[2]-box[0])*(box[3]-box[1])
    den=area if ioa else area+(others[:,2]-others[:,0])*(others[:,3]-others[:,1])-inter
    return inter/np.maximum(den,1e-12)

def evaluate(pred,truth,images,ignore_regions=None):
    validate(pred,images);ids=set(images.image_id);truth=truth[truth.image_id.isin(ids)]
    if ignore_regions is None:ignore_regions=pd.DataFrame(columns=['image_id']+BOX_COLUMNS)
    gt={mid:g[BOX_COLUMNS].to_numpy(float) for mid,g in truth.groupby('image_id')}
    ign={mid:g[BOX_COLUMNS].to_numpy(float) for mid,g in ignore_regions[ignore_regions.image_id.isin(ids)].groupby('image_id')}
    matched={mid:np.zeros(len(v),dtype=bool) for mid,v in gt.items()};tps=[];fps=[];ignored=0
    for r in pred.sort_values('confidence',ascending=False,kind='stable').itertuples(index=False):
        box=np.array([r.x_min,r.y_min,r.x_max,r.y_max]);mid=r.image_id;over=overlap(box,gt.get(mid,np.empty((0,4))))
        free=np.flatnonzero((over>=.5)&~matched.get(mid,np.empty(0,dtype=bool)))
        if len(free):matched[mid][free[np.argmax(over[free])]]=True;tps.append(1);fps.append(0)
        elif np.any(overlap(box,ign.get(mid,np.empty((0,4))),True)>=.5):ignored+=1
        else:tps.append(0);fps.append(1)
    tp=np.cumsum(tps);fp=np.cumsum(fps);total=len(truth)
    recall=tp/max(total,1);precision=tp/np.maximum(tp+fp,1)
    ap=float(np.mean([precision[recall>=r].max(initial=0) for r in np.linspace(0,1,101)])) if total else 0.
    return {'ap50':ap,'score_100':100*ap,'gt_boxes':total,'submitted_boxes':len(pred),'ignored_predictions':ignored,'true_positives':int(tp[-1]) if len(tp) else 0}
