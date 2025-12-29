import asyncio
import json
import time
import httpx
import os
from ollama import AsyncClient
from dotenv import load_dotenv 

load_dotenv()

OLLAMA_HOST = "http://localhost:11434"
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

DECOMPOSER_MODEL = "claim-splitter"
NORMALIZER_MODEL = "evidence-normalizer"
VERIFIER_MODEL = "fact-checker"

def safe_json_load(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
        raise


# SEARCH
async def search_serper(query):
    url = "https://google.serper.dev/news"
    payload = {
        "q": query,
        "gl": "vn",
        "hl": "vi",
        "num": 5
    }
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=5.0) as client:
        res = await client.post(url, json=payload, headers=headers)
        data = res.json()

    blocks = []
    for item in data.get("news", [])[:3]:
        blocks.append(
            f"LINK: {item.get('link','')}\n"
            f"DATE: {item.get('date','')}\n"
            f"TITLE: {item.get('title','')}\n"
            f"CONTENT: {item.get('snippet','')}"
        )

    return "\n\n".join(blocks) or ""

# NORMALIZE
async def normalize_evidence(client, raw_evidence):
    if not raw_evidence:
        return {"summary": "", "numbers_found": []}
    raw_evidence = raw_evidence[:1200]
    print(raw_evidence)

    res = await client.generate(
        model=NORMALIZER_MODEL,
        prompt=f"EVIDENCE:\n{raw_evidence}",
        format="json",
        stream=False
    )
    print("Response sau khi được normalize:", res["response"])
    return safe_json_load(res["response"])


# FACT CHECK
async def verify_claim(client, claim):
    raw_evidence = await search_serper(claim["query"])
    # raw_evidence = """
    # LINK: https://vneconomy.vn/chenh-lech-gia-vang-trong-nuoc-va-the-gioi-giam-gan-28-tu-phien-dinh-411.htm
    # DATE: 10 giờ trước TITLE: Chênh lệch giá vàng trong nước và thế giới giảm gần 28% từ phiên đỉnh 4/11 
    # CONTENT: Giá vàng miếng SJC giảm 6 triệu đồng/lượng trong phiên 22/12, chênh lệch với giá thế giới giảm còn 15,99 triệu đồng/lượng. 

    # LINK: https://nhandan.vn/gia-vang-ngay-2212-gia-vang-the-gioi-lap-dinh-moi-trong-nuoc-tang-vuot-muc-157-trieu-dongluong-post932118.html 
    # DATE: 10 giờ trước TITLE: Giá vàng ngày 22/12: Giá vàng thế giới lập đỉnh mới, trong nước tăng vượt mức 157 triệu đồng/lượng 
    # CONTENT: Giá vàng thế giới lập đỉnh mới 4.400 USD/ounce. Trong nước, giá vàng miếng SJC giao dịch quanh mức 157 triệu đồng/lượng, vàng nhẫn 154,1...

    # LINK: https://nld.com.vn/chieu-22-12-gia-vang-the-gioi-va-trong-nuoc-cung-lap-dinh-moi-196251222135734583.htm
    # DATE: 9 giờ trước TITLE: Chiều 22-12, giá vàng thế giới và trong nước cùng lập đỉnh mới 
    # CONTENT: (NLĐO) – Giá vàng thế giới lập đỉnh mới khi vượt 4.400 USD/ounce trong khi giá vàng miếng SJC cũng cao nhất từ trước tới nay.
    # """
    normalized = await normalize_evidence(client, raw_evidence)
    prompt = f"""
    CLAIM:
    {claim['text']}

    EVIDENCE:
    {normalized['summary']}

    Chỉ trả về format JSON, không thêm chữ gì khác ngoài format json.:

    {{
        "verdict": "TRUE" | "FALSE" | "UNCERTAIN",
        "reasoning": "Tối đa 15 từ.",
        "evidence_quote": "Trích dẫn ngắn nhất từ EVIDENCE (≤20 từ) hoặc rỗng."
    }}
    """

    res = await client.generate(
        model=VERIFIER_MODEL,
        prompt=prompt,
        format="json",
        stream=False
    )
    print("Kết quả cuối cùng:", res["response"])
    return safe_json_load(res["response"])


# MAIN PIPELINE
async def process_fact_check(user_input):
    start = time.time()
    client = AsyncClient(host=OLLAMA_HOST)  

    # Decompose
    res = await client.generate(
        model=DECOMPOSER_MODEL,
        prompt=user_input,
        format="json",
        stream=False
    )
    claims = json.loads(res["response"]).get("claims", [])
    print(f"[*] {len(claims)} claims detected\n")

    # Verify in parallel
    results = await asyncio.gather(
        *[verify_claim(client, c) for c in claims]
    )

    print("=" * 60)
    print("FACT CHECK RESULT")
    print("=" * 60)

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['verdict']}] {r['reasoning']}")
        if r["evidence_quote"]:
            print(f"Evidence: {r['evidence_quote']}")

    print(f"\nDone in {time.time() - start:.2f}s")


if __name__ == "__main__":
    query = "Giá vàng SJC hôm nay tại Việt Nam vượt mốc 90 triệu đồng trong hôm nay, vào ngày 23 tháng 12 năm 2025"
    asyncio.run(process_fact_check(query))
