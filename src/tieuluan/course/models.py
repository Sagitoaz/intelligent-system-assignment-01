"""Ba cách cài đặt dùng chung kiến trúc, khởi tạo và hàm mục tiêu.

Chỉ nhập thư viện học sâu khi được yêu cầu; suy luận NumPy dùng được trên Render.
"""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
from ..scratch.core import Sequential
from ..scratch.layers import Dense, ReLU, Conv2d, MaxPool2d, Flatten
from ..experiment_models import (scratch_model as recurrent_scratch,
    torch_model as recurrent_torch, keras_model as recurrent_keras,
    copy_weights_to_torch, copy_weights_to_keras)

KINDS = {'tabular': ('logistic','svm','mlp'),
         'image': ('cnn4','cnn8','cnndeep'), 'sequence': ('rnn','lstm','gru')}


def scratch_model(kind, shape, seed=11, n_classes=10):
    rng = np.random.default_rng(seed)
    if kind in ('rnn','lstm','gru'):
        return recurrent_scratch(kind, shape, seed)
    if kind in ('logistic','svm'):
        return Sequential(Dense(shape[0], 1, rng=rng))
    if kind == 'mlp':
        return Sequential(Dense(shape[0],16,rng=rng), ReLU(), Dense(16,1,rng=rng))
    filters = 8 if kind == 'cnn8' else 4
    layers = [Conv2d(1,filters,3,rng=rng), ReLU(), MaxPool2d(2)]
    side = (shape[-1]-2)//2
    if kind == 'cnndeep':
        layers += [Conv2d(filters,8,3,rng=rng), ReLU(), MaxPool2d(2)]
        filters, side = 8, (side-2)//2
    return Sequential(*layers, Flatten(), Dense(filters*side*side,n_classes,rng=rng))


def torch_model(kind, shape, n_classes=10):
    import torch
    from torch import nn
    torch.set_num_threads(2)
    if kind in ('rnn','lstm','gru'):
        return recurrent_torch(kind,shape)
    if kind in ('logistic','svm'):
        return nn.Sequential(nn.Linear(shape[0],1))
    if kind == 'mlp':
        return nn.Sequential(nn.Linear(shape[0],16), nn.ReLU(), nn.Linear(16,1))
    filters = 8 if kind == 'cnn8' else 4
    layers = [nn.Conv2d(1,filters,3), nn.ReLU(), nn.MaxPool2d(2)]
    side = (shape[-1]-2)//2
    if kind == 'cnndeep':
        layers += [nn.Conv2d(filters,8,3), nn.ReLU(), nn.MaxPool2d(2)]
        filters, side = 8, (side-2)//2
    return nn.Sequential(*layers,nn.Flatten(),nn.Linear(filters*side*side,n_classes))


def keras_model(kind, shape, n_classes=10):
    import keras
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except RuntimeError:
        pass
    if kind in ('rnn','lstm','gru'):
        return recurrent_keras(kind,shape)
    x = keras.Input(shape=shape)
    z = x
    if kind == 'mlp':
        z = keras.layers.Dense(16,activation='relu')(z)
    if kind.startswith('cnn'):
        z = keras.layers.Permute((2,3,1))(z)
        z = keras.layers.Conv2D(8 if kind=='cnn8' else 4,3,activation='relu')(z)
        z = keras.layers.MaxPooling2D(2)(z)
        if kind == 'cnndeep':
            z = keras.layers.Conv2D(8,3,activation='relu')(z)
            z = keras.layers.MaxPooling2D(2)(z)
        z = keras.layers.Permute((3,1,2))(z)
        z = keras.layers.Flatten()(z)
    return keras.Model(x,keras.layers.Dense(n_classes if kind.startswith('cnn') else 1)(z))


def build_models(kind, shape, seed=11, frameworks=('scratch','keras','pytorch'), n_classes=10):
    source = scratch_model(kind,shape,seed,n_classes)
    models = {'scratch':source}
    if 'pytorch' in frameworks:
        models['pytorch'] = torch_model(kind,shape,n_classes)
        copy_weights_to_torch(source,models['pytorch'],kind)
    if 'keras' in frameworks:
        models['keras'] = keras_model(kind,shape,n_classes)
        copy_weights_to_keras(source,models['keras'],kind)
    return {fw:models[fw] for fw in frameworks}


def loss_and_gradient(z, y, kind):
    """Loss dữ liệu; SVM thêm phạt L2 cho trọng số ở vòng cập nhật."""
    z = np.asarray(z,dtype=np.float64)
    y = np.asarray(y,dtype=int).reshape(-1)
    if kind.startswith('cnn'):
        shifted = z-z.max(axis=1,keepdims=True)
        exp = np.exp(shifted)
        probability = exp/exp.sum(axis=1,keepdims=True)
        loss = np.mean(np.log(exp.sum(axis=1))-shifted[np.arange(len(y)),y])
        probability[np.arange(len(y)),y] -= 1
        return float(loss), probability/len(y)
    signed = (2*y-1)[:,None]
    if kind == 'svm':
        margin = 1-signed*z
        return float(np.maximum(0,margin).mean()), -signed*(margin>0)/len(y)
    target = y[:,None]
    probability = .5*(1+np.tanh(z/2))
    return float(np.mean(np.logaddexp(0,z)-target*z)), (probability-target)/len(y)


def logits(model, framework, x, batch_size=128):
    output=[]
    if framework in ('scratch','pytorch'):
        model.eval()
    for offset in range(0,len(x),batch_size):
        batch=x[offset:offset+batch_size]
        if framework=='scratch':
            value=model.forward(batch)
        elif framework=='keras':
            value=model(batch,training=False).numpy()
        else:
            import torch
            with torch.no_grad():
                value=model(torch.from_numpy(batch)).numpy()
        output.append(value)
    return np.concatenate(output)


def probabilities(z):
    if z.shape[1]==1:
        p=.5*(1+np.tanh(z[:,0]/2))
        return np.column_stack([1-p,p])
    exp=np.exp(z-z.max(axis=1,keepdims=True))
    return exp/exp.sum(axis=1,keepdims=True)
