import numpy as np

from src.assignment04.full_data import _manifest_ids, load_full_splits, write_array


def test_write_array_is_reusable_and_memory_mapped(tmp_path):
    expected = np.arange(12, dtype=np.int32).reshape(3, 4)
    path = tmp_path / "train_ids.npy"
    write_array(path, expected)
    write_array(path, expected)
    loaded = np.load(path, mmap_mode="r")
    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, expected)


def test_load_full_comments_preserves_int32_ids(tmp_path):
    task_dir = tmp_path / "comments"
    task_dir.mkdir()
    for split, size in (("train", 3), ("validation", 2), ("test", 1)):
        np.save(task_dir / f"{split}_ids.npy", np.ones((size, 5), dtype=np.int32))
        np.save(task_dir / f"{split}_y.npy", np.zeros(size, dtype=np.int64))
    splits = load_full_splits(tmp_path, "comments")
    assert splits["train"]["ids"].dtype == np.int32
    assert [len(splits[name]["y"]) for name in ("train", "validation", "test")] == [3, 2, 1]


def test_manifest_ids_accepts_pandas_numeric_inference(tmp_path):
    path = tmp_path / "split.index.csv"
    path.write_text("source_id_json,split\n12,train\n34,validation\n56,test\n", encoding="utf-8")
    assert _manifest_ids(path) == {"train": [12], "validation": [34], "test": [56]}
