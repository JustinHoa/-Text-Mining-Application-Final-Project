from datasets import load_from_disk
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

def main():
    # 1. cấu hình
    # lưu dữ liệu vào folder 'vectordb'
    client = QdrantClient(path="vectordb") 
    collection_name = "vifactcheck"

    # 2. load data
    print("-> Đang load dataset...")
    ds = load_from_disk("seeding/")
    ds = ds.filter(lambda x: x['Evidence'] is not None and len(x['Evidence']) > 0)

    # 3. load Model
    print("-> Đang load model BGE-M3...")
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    # 4. tạo collection (xóa cũ tạo mới để tránh trùng lặp khi chạy lại)
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    print(f"Đã tạo mới collection '{collection_name}'")

    # 5. xử lý dữ liệu
    print(f"Đang embed {len(ds)} dòng dữ liệu...")
    
    texts = [item['Evidence'] for item in ds]
    # embedding
    embeddings = model.encode(texts, batch_size=12, return_dense=True)['dense_vecs']

    print("-> Đang chuẩn bị dữ liệu để lưu...")
    points = []
    for i, item in enumerate(ds):
        points.append(PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={
                "statement": item.get('Statement', ''),
                "evidence": item.get('Evidence', ''),
                "url": item.get('Url', ''),
                "topic": item.get('Topic', ''),
                "label": item.get('labels', '')
            }
        ))

    # 6. upsert vào Qdrant
    print("Đang ghi vào ổ đĩa...")
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    client.close()
    print("Dữ liệu đã được lưu vào folder 'vectordb'.")

if __name__ == "__main__":
    main()