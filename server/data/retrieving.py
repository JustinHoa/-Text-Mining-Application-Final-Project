from qdrant_client import QdrantClient, models
from FlagEmbedding import BGEM3FlagModel
import sys

class FactCheckSearcher:
    def __init__(self, db_path="vectordb", collection_name="vifactcheck"):
        print(f"Kết nối với DB tại: {db_path}")
        self.client = QdrantClient(path=db_path)
        self.collection_name = collection_name
        
        print("Load model BGE-M3...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

    def search(self, query, k=3, threshold=0.45):
        try:
            query_vec = self.model.encode(
                [query],
                return_dense=True
            )["dense_vecs"][0]

            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec.tolist(),
                limit=k,
                score_threshold=threshold,
                with_payload=True
            )

            results = []
            for hit in search_result.points:
                results.append({
                    "score": hit.score,
                    "evidence": hit.payload.get("evidence"),
                    "statement": hit.payload.get("statement"),
                    "url": hit.payload.get("url")
                })

            return results

        except Exception as e:
            print(f"Lỗi tìm kiếm: {e}")
            return []
    def close(self):
        self.client.close()

if __name__ == "__main__":
    searcher = None
    searcher = FactCheckSearcher()
    
    user_input = "Phó thủ tướng Trần Hồng Hà chúc mừng đài truyền hình"
    print(f"\nQuery: '{user_input}'")
    
    try:
        output = searcher.search(user_input, k=2, threshold=0.4)
        
        if output:
            print(f"Tìm thấy {len(output)} kết quả:")
            for res in output:
                print("-" * 50)
                print(f"Score: {res['score']:.4f}")
                print(f"Evidence: {res['evidence']}")
        else:
            print("Không tìm thấy kết quả nào đủ giống.")
    finally:
        if searcher:
            searcher.close()
        