"""Huấn luyện thực nghiệm mới: python -m scripts.tieuluan.run_course_experiments.

Có thể tiếp tục lần chạy bị gián đoạn; --overwrite mới ghi đè kết quả mới.
Thực nghiệm tài chính trong full/ không bao giờ bị sửa.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
from src.tieuluan.course.models import KINDS, build_models, loss_and_gradient, logits, probabilities
from src.tieuluan.scratch.optim import Adam, clip_grad_norm

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/tieuluan/course'
OUT=ROOT/'results/tieuluan/course'
MODELS=ROOT/'models/tieuluan/course'
REG=1e-3


def stepper(model, fw, kind):
    if fw=='scratch':
        params=[p for p in model.parameters() if not(kind in ('rnn','lstm') and p.name=='bias_hh')]
        optimizer=Adam(params,lr=1e-3,eps=1e-8)
        def step(x,y):
            model.zero_grad()
            z=model.forward(x)
            loss,grad=loss_and_gradient(z,y,kind)
            model.backward(grad.astype('float32'))
            if kind=='svm':
                weight=params[0]
                loss+=REG/2*float(np.sum(weight.value**2))
                weight.grad+=REG*weight.value
            clip_grad_norm(params,1.0)
            optimizer.step()
            return loss
    elif fw=='pytorch':
        import torch
        params=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.Adam(params,lr=1e-3,eps=1e-8)
        def step(x,y):
            optimizer.zero_grad()
            z=model(torch.from_numpy(x))
            target=torch.from_numpy(y)
            if kind.startswith('cnn'):
                loss=torch.nn.functional.cross_entropy(z,target.long())
            elif kind=='svm':
                loss=torch.relu(1-(2*target[:,None]-1)*z).mean()+REG/2*(params[0]**2).sum()
            else:
                loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target[:,None].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.0)
            optimizer.step()
            return float(loss.detach())
    else:
        import tensorflow as tf
        import keras
        optimizer=keras.optimizers.Adam(learning_rate=1e-3,epsilon=1e-8,global_clipnorm=1.0)
        @tf.function(reduce_retracing=True)
        def graph(x,y):
            with tf.GradientTape() as tape:
                z=model(x,training=True)
                if kind.startswith('cnn'):
                    loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=z))
                elif kind=='svm':
                    signed=2*tf.cast(y[:,None],tf.float32)-1
                    loss=tf.reduce_mean(tf.nn.relu(1-signed*z))+REG/2*tf.reduce_sum(model.trainable_variables[0]**2)
                else:
                    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y[:,None],tf.float32),logits=z))
            grads=tape.gradient(loss,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))
            return loss
        def step(x,y):
            return float(graph(x,y).numpy())
    return step


def snapshot(model,fw):
    if fw=='keras':
        return model.get_weights()
    return copy.deepcopy(model.state_dict())


def restore(model,fw,state):
    if fw=='keras':
        model.set_weights(state)
    else:
        model.load_state_dict(state)


def fit(model,fw,kind,data,epochs):
    step=stepper(model,fw,kind)
    rng=np.random.default_rng(1011)
    best=float('inf'); wait=0; state=None; history=[]
    started=time.perf_counter()
    for epoch in range(epochs):
        order=rng.permutation(len(data['train_y']))
        total=0.
        if fw in ('scratch','pytorch'):
            model.train()
        for offset in range(0,len(order),128):
            ids=order[offset:offset+128]
            total+=step(data['train_x'][ids],data['train_y'][ids])*len(ids)
        val=loss_and_gradient(logits(model,fw,data['val_x']),data['val_y'],kind)[0]
        history.append({'epoch':epoch+1,'train_loss':total/len(order),'val_loss':val})
        if not np.isfinite(val):
            raise ValueError('Hàm mất mát không hữu hạn')
        if val<best-1e-6:
            best=val; wait=0; state=snapshot(model,fw); best_epoch=epoch+1
        else:
            wait+=1
        if wait>=5:
            break
    restore(model,fw,state)
    return {'history':history,'best_epoch':best_epoch,'training_seconds':time.perf_counter()-started}


def evaluate(y,z,kind):
    scores=probabilities(z)
    pred=np.argmax(z,axis=1) if z.shape[1]>1 else (z[:,0]>=0).astype(int)
    result={'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
        'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
        'confusion_matrix':confusion_matrix(y,pred).tolist()}
    result['roc_auc']=roc_auc_score(y,z[:,0]) if z.shape[1]==1 else roc_auc_score(y,scores,multi_class='ovr',average='macro')
    return result,pred


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+')
    parser.add_argument('--frameworks',nargs='+',default=['scratch','pytorch','keras'])
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf8'))
    for key,meta in manifest.items():
        if args.datasets and key not in args.datasets:
            continue
        with np.load(DATA/f'{key}.npz') as archive:
            data={k:archive[k] for k in archive.files}
        for kind in KINDS[meta['group']]:
            needed=[fw for fw in args.frameworks if args.overwrite or not(OUT/f'{key}__{kind}__{fw}.json').exists()]
            if not needed:
                continue
            models=build_models(kind,data['train_x'].shape[1:],11,frameworks=needed)
            for fw,model in models.items():
                name=f'{key}__{kind}__{fw}'
                print('Huấn luyện',name,flush=True)
                result=fit(model,fw,kind,data,12 if meta['group']=='image' else 40)
                z=logits(model,fw,data['test_x'])
                metrics,pred=evaluate(data['test_y'],z,kind)
                result.update({'dataset':key,'kind':kind,'framework':fw,'seed':11,
                    'test':metrics,'data_sha256':meta['prepared_sha256'],
                    'threshold':0,'decision':'argmax' if kind.startswith('cnn') else 'logit >= 0',
                    'svm_l2':REG if kind=='svm' else None})
                if fw=='scratch':
                    model.save(MODELS/f'{name}.npz')
                elif fw=='keras':
                    model.save_weights(MODELS/f'{name}.weights.h5')
                else:
                    import torch
                    torch.save(model.state_dict(),MODELS/f'{name}.pt')
                np.savez_compressed(OUT/f'{name}_predictions.npz',test_logits=z,test_prediction=pred,
                    test_y=data['test_y'],test_ids=data['test_ids'])
                (OUT/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
                print('Hoàn thành',name,round(metrics['balanced_accuracy'],4),flush=True)
    code_files=['src/tieuluan/course/models.py','scripts/tieuluan/prepare_course_data.py',
                'scripts/tieuluan/run_course_experiments.py']
    (OUT/'run_manifest.json').write_text(json.dumps({'seed':11,'batch_size':128,'learning_rate':.001,
        'files':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code_files}},indent=2),encoding='utf8')


if __name__=='__main__':
    main()
