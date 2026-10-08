"""Kiểm tra tính đúng của thực nghiệm môn học đa lĩnh vực."""
import numpy as np
import pytest

from src.tieuluan.course.models import loss_and_gradient, build_models, logits


@pytest.mark.parametrize('kind', ['logistic', 'svm', 'cnn4'])
def test_loss_gradient(kind):
    rng = np.random.default_rng(12)
    z = rng.normal(size=(4, 10 if kind == 'cnn4' else 1))
    y = np.array([0, 1, 1, 0])
    _, grad = loss_and_gradient(z, y, kind)
    for index in np.ndindex(z.shape):
        plus, minus = z.copy(), z.copy()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        numerical = (loss_and_gradient(plus, y, kind)[0] - loss_and_gradient(minus, y, kind)[0]) / 2e-5
        assert grad[index] == pytest.approx(numerical, abs=1e-7)


@pytest.mark.parametrize('kind,shape', [('logistic',(5,)), ('svm',(5,)), ('mlp',(5,)),
    ('cnn4',(1,28,28)), ('cnn8',(1,28,28)), ('cnndeep',(1,28,28)),
    ('rnn',(20,1)), ('lstm',(20,1)), ('gru',(20,1))])
def test_framework_initial_predictions_match(kind, shape):
    models = build_models(kind, shape, 11)
    x = np.random.default_rng(42).normal(size=(3, *shape)).astype('float32')
    expected = logits(models['scratch'], 'scratch', x)
    for fw in ('keras','pytorch'):
        np.testing.assert_allclose(logits(models[fw], fw, x), expected, atol=2e-6, rtol=1e-5)


def test_extreme_logits_finite():
    for kind, z in [('cnn4',np.array([[1e4,-1e4]])), ('logistic',np.array([[-1e4]]))]:
        loss, gradient = loss_and_gradient(z, np.array([0]), kind)
        assert np.isfinite(loss) and np.isfinite(gradient).all()


def test_prepared_datasets_are_disjoint_and_finite():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'data/tieuluan/course'
    meta=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    assert len(meta)==6
    for key,item in meta.items():
        with np.load(root/f'{key}.npz') as data:
            for split in ('train','val','test'):
                assert np.isfinite(data[f'{split}_x']).all()
                assert len(data[f'{split}_x'])==item['splits'][split]['n']
                assert len(np.unique(data[f'{split}_y']))==len(item['classes'])
            assert not set(data['train_ids']) & set(data['val_ids'])
            if item['group']!='image':
                assert not set(data['train_ids']) & set(data['test_ids'])
                assert not set(data['val_ids']) & set(data['test_ids'])
            if item['group']=='sequence':
                assert max(data['train_ids'])<min(data['val_ids'])<min(data['test_ids'])
            if item['group']=='image':
                assert data['test_x'].shape==(10000,1,28,28)
