"""Shared constants for Assignment 04 experiments."""

from pathlib import Path

RANDOM_STATE = 42
KERNEL_SIZE = 3
CONV_STRIDE = 1
POOL_SIZE = 2
POOL_STRIDE = 2
FIELD_DIM = 8
EMBEDDING_DIM = 16
VOCAB_SIZE = 20_000
SEQUENCE_LENGTH = 128

DIABETES_DATA = Path("data/diabetes/diabetes_binary_health_indicators_BRFSS2015.csv")
HOUSE_DATA = Path("data/house_price/realtor-data.zip.csv")
COMMENTS_TRAIN = Path("data/ecommerce/train.csv")
COMMENTS_TEST = Path("data/ecommerce/test.csv")
