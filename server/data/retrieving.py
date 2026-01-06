import faiss
import pickle
from FlagEmbedding import BGEM3FlagModel

class FactCheckSearcher:
    def __init__(self, index_path="vectordb/vifactcheck.index", metadata_path="vectordb/metadata.pkl"):
        # load model
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        # load index
        self.index = faiss.read_index(index_path)
        # load metadata
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
            
    def search(self, query, k=3, threshold=0.45):
        """
        query: Câu nhập vào
        k: Số lượng kết quả tối đa
        threshold: Ngưỡng độ tương đồng (0.0 đến 1.0). 
                   Nếu score thấp hơn ngưỡng này sẽ coi như không thấy.
        """
        # embed query
        query_vec = self.model.encode([query], return_dense=True)['dense_vecs']
        
        # search trong FAISS
        scores, indices = self.index.search(query_vec.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # kiểm tra nếu score threshold mới lấy
            if score >= threshold:
                item = self.metadata[int(idx)]
                results.append({
                    "score": float(score),
                    "statement": item.get('Statement'),
                    "evidence": item.get('Evidence'),
                    "url": item.get('Url'),
                    "topic": item.get('Topic')
                })
        
        return results if len(results) > 0 else None
    
if __name__ == "__main__":
    searcher = FactCheckSearcher()
    
    user_input = "bất động sản"
    output = searcher.search(user_input, k=2, threshold=0.5)
    
    if output:
        print(f"Tìm thấy {len(output)} kết quả:")
        for res in output:
            print(f"[{res['score']:.4f}] - {res['evidence']}")
    else:
        print("Không tìm thấy nội dung nào phù hợp.")