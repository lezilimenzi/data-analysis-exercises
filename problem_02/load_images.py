"""Image loading helpers only. The learner supplies validation and the model."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

DEFAULT_ROOT=Path(__file__).resolve().parent

def read_image(path):
    with Image.open(path) as image:
        if image.size != (28,28):
            raise ValueError(f'Unexpected image size: {path}')
        return np.asarray(image.convert('L'),dtype=np.float32)/255.0

def load_train(root=DEFAULT_ROOT):
    root=Path(root)
    classes=json.loads((root/'classes.json').read_text(encoding='utf-8'))
    images,labels=[],[]
    for item in classes:
        for path in sorted((root/item['folder']).glob('*.png')):
            images.append(read_image(path)); labels.append(item['label'])
    return np.stack(images),np.asarray(labels,dtype=np.int64)

def load_test(root=DEFAULT_ROOT):
    paths=sorted((Path(root)/'test').glob('*.png'))
    return np.stack([read_image(p) for p in paths]),[p.stem for p in paths]

def save_submission(image_ids, predictions, path='submission.json'):
    values=np.asarray(predictions)
    if values.shape != (len(image_ids),):
        raise ValueError('One label is required for every image ID.')
    if values.dtype.kind not in 'iu' or not np.isin(values,range(7)).all():
        raise ValueError('Predictions must be integer class labels 0..6.')
    if len(set(image_ids)) != len(image_ids):
        raise ValueError('Duplicate image IDs.')
    rows=[{'image_id':str(k),'label':int(v)} for k,v in zip(image_ids,values)]
    Path(path).write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    x,y=load_train(); xt,ids=load_test()
    print('train:',x.shape,'labels:',y.shape,'test:',xt.shape)
    print('These helpers do not train a model or produce predictions.')
