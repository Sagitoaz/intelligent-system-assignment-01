"""Cấu hình chung cho các test trong tests/: mặc định chạy KB bằng file (kg.json) để test không phụ thuộc mạng/Neo4j.

Muốn thử với Neo4j thật: đặt KB_BACKEND=neo4j trước khi chạy pytest (cần file Neo4j-*.txt hoặc biến NEO4J_*).
"""
import os

os.environ.setdefault("KB_BACKEND", "file")
