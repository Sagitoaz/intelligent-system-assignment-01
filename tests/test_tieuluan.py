import numpy as np
import pandas as pd
from src.tieuluan.experiment_data import market_samples, gaf_image, split_by_dates
from src.tieuluan.scratch.core import Sequential
from src.tieuluan.scratch.layers import Dense, Tanh, Conv2d, Flatten
from src.tieuluan.scratch.recurrent import SimpleRNN, LSTM, GRU, LastStep
from src.tieuluan.scratch.losses import MSELoss
from src.tieuluan.scratch.gradcheck import check_model
import pytest


def test_prepared_archives_load_without_pickle():
    from src.tieuluan.experiment_data import OUT
    for path in OUT.glob('*.npz'):
        with np.load(path,allow_pickle=False) as data:
            for key in data.files:
                assert data[key].dtype != object, (path.name,key)


def test_resume_rejects_changed_data_or_environment(tmp_path):
    from src.tieuluan.run_provenance import make_manifest, ensure_manifest
    data=tmp_path/'data.txt';data.write_text('original')
    first=make_manifest([data],{'epochs':20,'numpy':'v1'})
    ensure_manifest(tmp_path,first)
    data.write_text('changed')
    changed=make_manifest([data],{'epochs':20,'numpy':'v1'})
    assert first['fingerprint'] != changed['fingerprint']
    with pytest.raises(ValueError,match='incompatible'):
        ensure_manifest(tmp_path,changed)
    assert first['fingerprint'] != make_manifest([data],{'epochs':20,'numpy':'v2'})['fingerprint']


def test_display_math_render(tmp_path):
    from scripts.tieuluan.build_report import render_equation
    image=render_equation(r'g=\\frac{1}{n}\\sum_i x_i'.replace('\\\\','\\'),tmp_path)
    assert image.exists() and image.stat().st_size>100


def test_changed_raw_snapshot_rejected(tmp_path):
    import hashlib,json
    from src.tieuluan.run_provenance import check_raw_snapshot
    processed=tmp_path/'data/tieuluan/processed';processed.mkdir(parents=True)
    raw=tmp_path/'data/tieuluan/raw/indices';raw.mkdir(parents=True)
    p=raw/'sp500.csv';p.write_text('old')
    (processed/'inventory.json').write_text(json.dumps({'sp500':{'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}}))
    check_raw_snapshot(tmp_path)
    p.write_text('new')
    with pytest.raises(ValueError,match='raw snapshot'):
        check_raw_snapshot(tmp_path)


def test_samples_do_not_use_future():
    frame = pd.DataFrame({'date': pd.date_range('2019-11-01', periods=100), 'close': np.arange(100.)+100})
    a = market_samples(frame)
    frame.loc[70:, 'close'] *= 3
    b = market_samples(frame)
    np.testing.assert_array_equal(a['sequence'][:40], b['sequence'][:40])
    np.testing.assert_array_equal(a['image'][:40], b['image'][:40])
    assert (a['target_date'] > a['date']).all()


def test_split_purges_boundary_target():
    dates = np.array(['2019-12-30','2019-12-31','2022-12-31'],dtype='datetime64[D]')
    target = np.array(['2019-12-31','2020-01-02','2023-01-01'],dtype='datetime64[D]')
    masks = split_by_dates(dates,target)
    assert masks['train'].tolist() == [True,False,False]
    assert not any(mask[1] or mask[2] for mask in masks.values())


def test_constant_gaf_finite():
    image = gaf_image(np.ones(20))
    assert image.shape == (20,20)
    assert np.isfinite(image).all()
    assert abs(image).max() <= 1.000001


def test_gaf_distinguishes_rising_from_falling():
    # Với chuẩn hóa [-1, 1], chuỗi và ảnh phản chiếu của nó cho cùng một GASF; chuẩn hóa [0, 1] thì không.
    rising = np.linspace(100, 110, 20)
    falling = rising[::-1]
    assert not np.allclose(gaf_image(rising), gaf_image(falling))
    mirrored = rising.max() + rising.min() - rising
    assert not np.allclose(gaf_image(rising), gaf_image(mirrored))


@pytest.mark.parametrize('kind',[SimpleRNN,LSTM,GRU])
def test_recurrent_gradients(kind):
    rng = np.random.default_rng(8)
    model = Sequential(kind(2,3,rng=rng,dtype=np.float64),LastStep(),Dense(3,1,rng=rng,dtype=np.float64))
    errors = check_model(model,MSELoss(),rng.normal(size=(2,4,2)),rng.normal(size=(2,1)))
    assert max(errors.values()) < 1e-4, errors


def test_convolution_gradients():
    rng = np.random.default_rng(8)
    model = Sequential(Conv2d(1,2,3,rng=rng,dtype=np.float64),Tanh(),Flatten(),Dense(18,1,rng=rng,dtype=np.float64))
    errors = check_model(model,MSELoss(),rng.normal(size=(2,1,5,5)),rng.normal(size=(2,1)))
    assert max(errors.values()) < 1e-4, errors


@pytest.mark.parametrize('kind,shape',[('mlp',(3,5)),('cnn4',(3,1,20,20)),('lstm',(3,20,1))])
def test_framework_forward_equivalence(kind,shape):
    from src.tieuluan.experiment_models import build_models, predict_logits
    models = build_models(kind,shape[1:],11)
    x = np.random.default_rng(3).normal(size=shape).astype(np.float32)
    expected = predict_logits(models['scratch'],'scratch',x)
    for framework in ['pytorch','keras']:
        np.testing.assert_allclose(predict_logits(models[framework],framework,x),expected,atol=2e-6,rtol=2e-5)
