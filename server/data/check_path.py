from qdrant_client import QdrantClient
import qdrant_client

print("--- THÔNG TIN CHI TIẾT ---")
print(f"1. Version Python nhận diện: {qdrant_client.__version__}")
print(f"2. Đường dẫn file: {qdrant_client.__file__}")

client = QdrantClient(path="vectordb")
print(f"3. Loại object client: {type(client)}")

# Kiểm tra xem có hàm 'search' không
has_search = hasattr(client, 'search')
print(f"4. Có hàm 'search' không?: {'✅ CÓ' if has_search else '❌ KHÔNG'}")

# Nếu không có, liệt kê các hàm có chữ 'search' hoặc 'query' để xem nó dùng tên gì
if not has_search:
    print("\n--- CÁC HÀM HIỆN CÓ TRONG CLIENT ---")
    all_methods = dir(client)
    search_methods = [m for m in all_methods if 'search' in m or 'query' in m]
    print(f"Các hàm liên quan tìm kiếm: {search_methods}")
    
    print("\n5 hàm đầu tiên (bất kỳ):")
    print([m for m in all_methods if not m.startswith('_')][:5])
    
client.close()