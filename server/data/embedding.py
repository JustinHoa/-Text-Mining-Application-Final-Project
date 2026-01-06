from datasets import load_from_disk
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np
import pickle

ds = load_from_disk("seeding/")

# bỏ các records không có Evidence
ds = ds.filter(lambda x: x['Evidence'] is not None and len(x['Evidence']) > 0)
evidence_list = ds['Evidence']

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

print("Đang embed dữ liệu... vui lòng đợi.")
embeddings = model.encode(evidence_list, batch_size=12, return_dense=True)['dense_vecs']

# tạo FAISS Index
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension) 
index.add(embeddings.astype('float32'))

# lưu Index và Metadata để dùng sau
faiss.write_index(index, "vifactcheck.index")
with open("metadata.pkl", "wb") as f:
    pickle.dump(ds, f) # Lưu lại dataset để lấy thông tin chi tiết khi search trúng

print("Đã lưu Index thành công")