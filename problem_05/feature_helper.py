"""Frozen image encoder. No task labels, classifier, or test answers are used."""
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent

def extract_features(paths,model_path=None,cache_path=None,verbose=True):
    """Returns (len(paths), 1280) float32 features in the same path order.

    Local model only; uses CPU. Resizes RGB images to 224x224, normalizes
    with ImageNet mean/std, and reads the final pooled feature vector.
    Optional cache is checked against model/image contents and preprocessing.
    """
    import onnxruntime as ort
    paths=[Path(p) for p in paths]
    if not paths:raise ValueError('No images supplied.')
    model_path=Path(model_path) if model_path else ROOT/'models/image_encoder.onnx'
    h=hashlib.sha256(b'p05-rgb-bilinear224-imagenet-v1')
    h.update(model_path.read_bytes())
    for p in paths:h.update(hashlib.sha256(p.read_bytes()).digest())
    signature=h.hexdigest()
    if cache_path is not None and Path(cache_path).exists():
        with np.load(cache_path,allow_pickle=False) as data:
            if data['signature'].item()==signature and data['features'].shape==(len(paths),1280):
                if verbose:print('Using cached image features.')
                return data['features'].copy()
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(model_path),sess_options=opts,providers=['CPUExecutionProvider'])
    input_name=session.get_inputs()[0].name
    mean=np.array([.485,.456,.406],dtype=np.float32).reshape(3,1,1)
    std=np.array([.229,.224,.225],dtype=np.float32).reshape(3,1,1)
    result=[]
    for i,p in enumerate(paths):
        with Image.open(p) as im:
            rgb=np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR),dtype=np.float32)/255.0
        tensor=np.ascontiguousarray(((rgb.transpose(2,0,1)-mean)/std)[None])
        vector=session.run(None,{input_name:tensor})[0].reshape(-1)
        if vector.shape!=(1280,) or not np.isfinite(vector).all():raise ValueError('Invalid feature output.')
        result.append(vector)
        if verbose and ((i+1)%100==0 or i+1==len(paths)):print(f'Images processed: {i+1}/{len(paths)}')
    features=np.stack(result).astype(np.float32)
    if cache_path is not None:np.savez_compressed(cache_path,features=features,signature=signature)
    return features
