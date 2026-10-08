"""Vòng huấn luyện hiện đầy đủ trong notebook; lưu checkpoint sau mỗi epoch.

Checkpoint pickle chỉ được đọc từ tệp cục bộ do chính notebook tạo.
Không tải hoặc mở checkpoint pickle từ người khác.
"""
import copy
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix
from src.tieuluan.course.large_data import batch_input,enforce_size
from src.tieuluan.course.models import build_models,loss_and_gradient,probabilities
from src.tieuluan.scratch.optim import Adam,clip_grad_norm


class BatchTrainer:
    def __init__(self,model,framework,kind,learning_rate=.001):
        self.model,self.framework,self.kind=model,framework,kind
        if framework=='scratch':
            self.params=[p for p in model.parameters() if not(kind in ('rnn','lstm') and p.name=='bias_hh')]
            self.optimizer=Adam(self.params,lr=learning_rate,eps=1e-8)
        elif framework=='pytorch':
            import torch
            self.params=[p for p in model.parameters() if p.requires_grad]
            self.optimizer=torch.optim.Adam(self.params,lr=learning_rate,eps=1e-8)
        else:
            import keras
            import tensorflow as tf
            self.optimizer=keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-8,global_clipnorm=1.)
            self.optimizer.build(model.trainable_variables)
            self.graph=tf.function(self.keras_step,reduce_retracing=True)

    def keras_step(self,x,y):
        import tensorflow as tf
        with tf.GradientTape() as tape:
            z=self.model(x,training=True)
            if self.kind.startswith('cnn'):
                loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=z))
            elif self.kind=='svm':
                target=2*tf.cast(y[:,None],tf.float32)-1
                loss=tf.reduce_mean(tf.nn.relu(1-target*z))+.0005*tf.reduce_sum(self.model.trainable_variables[0]**2)
            else:
                loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y[:,None],tf.float32),logits=z))
        gradients=tape.gradient(loss,self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,self.model.trainable_variables))
        return loss

    def step(self,x,y):
        if self.framework=='keras':
            return float(self.graph(x,y).numpy())
        self.model.train()
        if self.framework=='scratch':
            self.model.zero_grad()
            z=self.model.forward(x)
            loss,gradient=loss_and_gradient(z,y,self.kind)
            self.model.backward(gradient.astype('float32'))
            if self.kind=='svm':
                loss+=.0005*float(np.sum(self.params[0].value**2))
                self.params[0].grad+=.001*self.params[0].value
            clip_grad_norm(self.params,1.)
            self.optimizer.step()
            return loss
        import torch
        self.optimizer.zero_grad()
        z=self.model(torch.from_numpy(x)); target=torch.from_numpy(y)
        if self.kind.startswith('cnn'):
            loss=torch.nn.functional.cross_entropy(z,target.long())
        elif self.kind=='svm':
            loss=torch.relu(1-(2*target[:,None]-1)*z).mean()+.0005*(self.params[0]**2).sum()
        else:
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z,target[:,None].float())
        loss.backward(); torch.nn.utils.clip_grad_norm_(self.params,1.)
        self.optimizer.step()
        return float(loss.detach())

    def state(self):
        if self.framework=='scratch':
            return copy.deepcopy({'m':self.optimizer.m,'v':self.optimizer.v,'t':self.optimizer.t})
        if self.framework=='pytorch':
            return copy.deepcopy(self.optimizer.state_dict())
        return [v.numpy() for v in self.optimizer.variables]

    def restore(self,state):
        if self.framework=='scratch':
            self.optimizer.m,self.optimizer.v,self.optimizer.t=copy.deepcopy(state['m']),copy.deepcopy(state['v']),state['t']
        elif self.framework=='pytorch':
            self.optimizer.load_state_dict(state)
        else:
            for variable,value in zip(self.optimizer.variables,state,strict=True):
                variable.assign(value)


def model_state(model,framework):
    return model.get_weights() if framework=='keras' else copy.deepcopy(model.state_dict())


def restore_model(model,framework,state):
    model.set_weights(state) if framework=='keras' else model.load_state_dict(state)


def infer_batches(model,framework,x,batch_size=256):
    outputs=[]
    if framework!='keras':
        model.eval()
    for start in range(0,len(x),batch_size):
        xb=batch_input(x[start:start+batch_size])
        if framework=='scratch':
            value=model.forward(xb)
        elif framework=='keras':
            value=model(xb,training=False).numpy()
        else:
            import torch
            with torch.no_grad():
                value=model(torch.from_numpy(xb)).numpy()
        outputs.append(value)
    return np.concatenate(outputs)


def fit_large(model,framework,kind,data,config,checkpoint,signature):
    """Khôi phục cả trọng số, Adam, RNG, patience, checkpoint tốt nhất và lịch sử."""
    trainer=BatchTrainer(model,framework,kind,config['learning_rate'])
    rng=np.random.default_rng(config['seed']+1000)
    state={'signature':signature,'history':[],'best_loss':float('inf'),'best_model':None,
           'best_epoch':0,'wait':0,'training_seconds':0.}
    checkpoint=Path(checkpoint)
    if checkpoint.exists():
        with checkpoint.open('rb') as handle:
            saved=pickle.load(handle)
        if saved['signature']!=signature:
            raise ValueError('Checkpoint khác dữ liệu/mã/cấu hình. Chọn RUN_ID mới; không ghi đè lặng lẽ.')
        state=saved
        restore_model(model,framework,state['model'])
        trainer.restore(state['optimizer'])
        rng.bit_generator.state=state['rng']
        print('Tiếp tục sau epoch',len(state['history']))
    for epoch in range(len(state['history']),config['epochs']):
        if state['wait']>=config['patience']:
            break
        started=time.perf_counter(); total=0.
        order=rng.permutation(len(data['train_y']))
        for offset in range(0,len(order),config['batch_size']):
            ids=order[offset:offset+config['batch_size']]
            total+=trainer.step(batch_input(data['train_x'][ids]),np.asarray(data['train_y'][ids],dtype='int64'))*len(ids)
        val=loss_and_gradient(infer_batches(model,framework,data['val_x']),data['val_y'],kind)[0]
        if not np.isfinite(val):
            raise ValueError('Validation loss không hữu hạn; không lưu kết quả sai.')
        elapsed=time.perf_counter()-started
        state['history'].append({'epoch':epoch+1,'train_loss':total/len(order),'val_loss':val,'seconds':elapsed})
        state['training_seconds']+=elapsed
        if val<state['best_loss']-1e-6:
            state.update(best_loss=val,best_model=model_state(model,framework),best_epoch=epoch+1,wait=0)
        else:
            state['wait']+=1
        state.update(model=model_state(model,framework),optimizer=trainer.state(),rng=rng.bit_generator.state)
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        temporary=checkpoint.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            pickle.dump(state,handle,protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(checkpoint)
        print(f"Epoch {epoch+1}/{config['epochs']} | học {total/len(order):.4f} | validation {val:.4f} | {elapsed:.1f} giây",flush=True)
    restore_model(model,framework,state['best_model'])
    return {k:state[k] for k in ('history','best_epoch','best_loss','training_seconds')}


def evaluate_large(y,z):
    pred=z.argmax(1) if z.shape[1]>1 else (z[:,0]>=0).astype(int)
    result={'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
        'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
        'confusion_matrix':confusion_matrix(y,pred).tolist()}
    # ROC-AUC đa lớp chỉ có nghĩa khi tất cả các lớp đều có trong tập test.
    if z.shape[1]==1:
        result['roc_auc']=roc_auc_score(y,z[:,0])
    elif len(np.unique(y))==z.shape[1]:
        result['roc_auc']=roc_auc_score(y,probabilities(z),multi_class='ovr',average='macro')
    return result,pred


def run_experiment(root,key,kind,framework,data,meta,config,run_id,code_hash):
    enforce_size(len(data['train_y']))
    folder=Path(root)/'results/tieuluan/course_large'/run_id
    folder.mkdir(parents=True,exist_ok=True)
    stem=f'{key}__{kind}__{framework}__{config["seed"]}'
    signature=hashlib.sha256(json.dumps({'arrays':meta['array_hashes'],'config':config,'code':code_hash,
        'key':key,'kind':kind,'framework':framework},sort_keys=True).encode()).hexdigest()
    record_path=folder/f'{stem}.json'
    if record_path.exists():
        record=json.loads(record_path.read_text(encoding='utf8'))
        if record['signature']!=signature:
            raise ValueError('Kết quả cũ khác cấu hình. Đổi RUN_ID để tách thực nghiệm.')
        if not (folder/f'{stem}_predictions.npz').exists():
            raise ValueError('Thiếu tệp dự báo của lượt chạy đã hoàn thành.')
        print('Đã hoàn thành, bỏ qua:',stem)
        return record
    print('Bắt đầu:',stem,flush=True)
    model=build_models(kind,tuple(meta['input_shape']),config['seed'],frameworks=(framework,),n_classes=len(meta['classes']))[framework]
    history=fit_large(model,framework,kind,data,config,folder/'checkpoints'/f'{stem}.pkl',signature)
    z=infer_batches(model,framework,data['test_x'])
    metrics,pred=evaluate_large(data['test_y'],z)
    baselines={}
    majority=int(np.bincount(data['train_y']).argmax())
    for name,baseline in [('majority',np.full(len(pred),majority)),('persistence',data.get('test_persistence'))]:
        if baseline is not None:
            baselines[name]={'accuracy':accuracy_score(data['test_y'],baseline),
                'balanced_accuracy':balanced_accuracy_score(data['test_y'],baseline)}
    model_dir=Path(root)/'models/tieuluan/course_large'/run_id
    model_dir.mkdir(parents=True,exist_ok=True)
    if framework=='scratch':
        model.save(model_dir/f'{stem}.npz')
    elif framework=='keras':
        model.save_weights(model_dir/f'{stem}.weights.h5')
    else:
        import torch
        torch.save(model.state_dict(),model_dir/f'{stem}.pt')
    np.savez_compressed(folder/f'{stem}_predictions.npz',y=data['test_y'],logits=z,prediction=pred,ids=data['test_ids'])
    record={'dataset':key,'kind':kind,'framework':framework,'config':config,'signature':signature,
        'code_hash':code_hash,'metadata':meta,'metrics':metrics,'baselines':baselines,**history}
    temporary=record_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf8')
    temporary.replace(record_path)
    return record
