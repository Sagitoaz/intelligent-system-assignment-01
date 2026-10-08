"""Kiểm thử nhỏ, không huấn luyện đầy đủ và không gọi mạng."""
import numpy as np
import pytest
from src.tieuluan.course.large_data import temporal_arrays, enforce_size, batch_input
from src.tieuluan.course.models import build_models, logits


def test_training_size_is_strict():
    with pytest.raises(ValueError,match='100.000'):
        enforce_size(100000)
    enforce_size(100001)


def test_time_windows_do_not_cross_split_or_missing_values():
    values=np.arange(1000,dtype=float)
    values[100]=np.nan
    arrays,meta=temporal_arrays(values,window=20)
    for split in ('train','val','test'):
        assert np.isfinite(arrays[f'{split}_x']).all()
        assert (arrays[f'{split}_y']==1).all()
        ids=arrays[f'{split}_ids']
        assert not np.any((ids>=100)&(ids<=120))
    assert arrays['train_ids'].max()<arrays['val_ids'].min()-20
    assert arrays['val_ids'].max()<arrays['test_ids'].min()-20


def test_image_scaling_only_applied_to_bytes():
    np.testing.assert_allclose(batch_input(np.array([0,255],dtype='uint8')),[0,1])
    np.testing.assert_allclose(batch_input(np.array([0,2],dtype='float32')),[0,2])


def test_49_class_framework_parity():
    models=build_models('cnn4',(1,28,28),11,n_classes=49)
    x=np.random.default_rng(1).normal(size=(2,1,28,28)).astype('float32')
    expected=logits(models['scratch'],'scratch',x)
    assert expected.shape==(2,49)
    for fw in ('pytorch','keras'):
        np.testing.assert_allclose(logits(models[fw],fw,x),expected,atol=2e-6)


@pytest.mark.parametrize('framework',['scratch','pytorch','keras'])
def test_checkpoint_preserves_optimizer_and_next_update(framework):
    from src.tieuluan.course.large_training import BatchTrainer,model_state,restore_model
    x=np.random.default_rng(4).normal(size=(4,3)).astype('float32')
    y=np.array([0,1,0,1],dtype='int64')
    first=build_models('mlp',(3,),11,frameworks=(framework,))[framework]
    trainer=BatchTrainer(first,framework,'mlp')
    trainer.step(x,y)
    weights=model_state(first,framework); optimizer=trainer.state()
    trainer.step(x,y)
    expected=logits(first,framework,x)
    second=build_models('mlp',(3,),11,frameworks=(framework,))[framework]
    restored=BatchTrainer(second,framework,'mlp')
    restore_model(second,framework,weights); restored.restore(optimizer)
    restored.step(x,y)
    np.testing.assert_allclose(logits(second,framework,x),expected,atol=1e-6)


def test_notebooks_contain_executable_model_code():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for filename in ('02_ml.ipynb','03_cnn.ipynb','04_rnn.ipynb'):
        notebook=json.loads((root/'notebooks/tieuluan'/filename).read_text(encoding='utf8'))
        namespace={}
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='code': continue
            source=''.join(cell['source'])
            compile(source,filename+':'+str(i),'exec')
            if 'definition' in cell.get('metadata',{}).get('tags',[]):
                exec(source,namespace)
        for name in ('Conv2d','LSTM','GRU','BatchTrainer','load_csv_dataset','fit_large'):
            assert name in namespace
        models=namespace['build_models']('cnn4',(1,28,28),11,n_classes=49)
        x=np.zeros((2,1,28,28),dtype='float32')
        expected=namespace['logits'](models['scratch'],'scratch',x)
        for fw in ('keras','pytorch'):
            np.testing.assert_allclose(namespace['logits'](models[fw],fw,x),expected,atol=2e-6)
