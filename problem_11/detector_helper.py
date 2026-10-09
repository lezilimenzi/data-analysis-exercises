"""Optional local YOLOv8n CPU inference. Contains no task labels or tile pipeline.

The caller chooses preprocessing/crops and combines their predictions.
"""
from pathlib import Path
import numpy as np
from PIL import Image

def nms(boxes,scores,iou_threshold=.5):
    boxes=np.asarray(boxes,dtype=float).reshape(-1,4);scores=np.asarray(scores,dtype=float)
    if not len(boxes):return np.empty(0,dtype=int)
    order=np.argsort(-scores,kind='stable');keep=[]
    area=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    while len(order):
        k=order[0];keep.append(k);rest=order[1:]
        lt=np.maximum(boxes[k,:2],boxes[rest,:2]);rb=np.minimum(boxes[k,2:],boxes[rest,2:])
        inter=np.maximum(rb-lt,0).prod(1);union=area[k]+area[rest]-inter
        overlap=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        order=rest[overlap<=iou_threshold]
    return np.asarray(keep,dtype=int)

class Detector:
    def __init__(self,model_path=None,threads=2):
        import onnxruntime as ort
        p=Path(model_path) if model_path else Path(__file__).resolve().parent/'models/yolov8n.onnx'
        opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(p),sess_options=opts,providers=['CPUExecutionProvider'])
        self.input_name=self.session.get_inputs()[0].name

    def detect(self,image,conf=.03,iou=.5,classes=(2,5,7)):
        """PIL RGB image -> Nx6 [xmin,ymin,xmax,ymax,confidence,COCO_class].

        Coordinates refer to the supplied image (or crop), NOT its parent.
        Class 2=car, 5=bus, 7=truck. Other pretrained models are permitted.
        """
        if isinstance(image,(str,Path)):
            with Image.open(image) as im:image=im.convert('RGB')
        image=image.convert('RGB');w,h=image.size;scale=min(640/w,640/h)
        rw,rh=round(w*scale),round(h*scale);left=(640-rw)//2;top=(640-rh)//2
        canvas=Image.new('RGB',(640,640),(114,114,114));canvas.paste(image.resize((rw,rh),Image.Resampling.BILINEAR),(left,top))
        tensor=np.asarray(canvas,dtype=np.float32).transpose(2,0,1)[None]/255
        raw=self.session.run(None,{self.input_name:np.ascontiguousarray(tensor)})[0][0].T
        cid=raw[:,4:].argmax(1);scores=raw[np.arange(len(raw)),4+cid];mask=(scores>=conf)&np.isin(cid,classes)
        raw=raw[mask];scores=scores[mask];cid=cid[mask]
        boxes=np.column_stack([raw[:,0]-raw[:,2]/2,raw[:,1]-raw[:,3]/2,raw[:,0]+raw[:,2]/2,raw[:,1]+raw[:,3]/2])
        boxes[:,[0,2]]=(boxes[:,[0,2]]-left)/(rw/w);boxes[:,[1,3]]=(boxes[:,[1,3]]-top)/(rh/h)
        boxes[:,[0,2]]=boxes[:,[0,2]].clip(0,w);boxes[:,[1,3]]=boxes[:,[1,3]].clip(0,h)
        good=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1]);boxes,scores,cid=boxes[good],scores[good],cid[good]
        keep=[]
        for c in classes:
            ix=np.flatnonzero(cid==c);keep.extend(ix[nms(boxes[ix],scores[ix],iou)])
        keep=np.array(keep,dtype=int)
        return np.column_stack([boxes[keep],scores[keep],cid[keep]]).astype(np.float32)
