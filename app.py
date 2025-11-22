import os
import re
import requests
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)

# --- تنظیمات ---
GENERATION_MODEL = "gemini-2.5-flash" 
EMBEDDING_MODEL = "models/text-embedding-004" 

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

DB_NAME = "my_rag_db"
COLLECTION_NAME = "perplex_context"
INDEX_NAME = "vector_index"

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# اتصال به دیتابیس
try:
    if not MONGO_URI:
        print("❌ FATAL: MONGO_URI is missing.")
        mongo_client = None
        collection = None
    else:
        mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]
        # تست اتصال سریع
        mongo_client.admin.command('ping')
        print(f"✅ Connected to MongoDB Atlas. DB: {DB_NAME}, Coll: {COLLECTION_NAME}")
except Exception as e:
    print(f"❌ DB Connection Error: {e}")
    mongo_client = None
    collection = None

# --- توابع کمکی ---

def recursive_chunk_text(text, chunk_size=1000, overlap=100):
    if not text: return []
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break
        
        block = text[start:end]
        split_point = -1
        match = re.search(r'[.!?]\s+', block[::-1])
        if match: split_point = len(block) - match.start()
        
        if split_point == -1:
            last_space = block.rfind(' ')
            if last_space != -1: split_point = last_space
        
        if split_point == -1: split_point = chunk_size
            
        chunks.append(text[start : start + split_point])
        start += split_point - overlap
    
    # DEBUG: چاپ تعداد چانک‌ها
    print(f"✂️ Chunked text into {len(chunks)} parts.")
    return chunks

def get_embedding(text, task_type="retrieval_document"):
    if not text or not text.strip(): return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        emb = result['embedding']
        # DEBUG: چاپ طول وکتور (فقط یکبار برای چک کردن)
        # print(f"📏 Vector Dimension Generated: {len(emb)}") 
        return emb
    except Exception as e:
        print(f"⚠️ Embedding Error: {e}")
        return None

def generate_search_queries(prompt):
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        sys_prompt = (
            f"User prompt: '{prompt}'\n"
            "Generate 3 specific search queries. Return ONLY the queries separated by newlines."
        )
        resp = model.generate_content(sys_prompt)
        return [q.strip() for q in resp.text.split('\n') if q.strip()]
    except Exception as e:
        print(f"Query Gen Error: {e}")
        return [prompt]

def tavily_search(queries):
    if not TAVILY_API_KEY: 
        print("⚠️ Tavily Key Missing")
        return []
    combined_results = []
    for q in queries[:2]: 
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": q,
                    "search_depth": "basic",
                    "max_results": 3
                }
            )
            data = resp.json()
            if 'results' in data:
                combined_results.extend(data['results'])
        except Exception as e:
            print(f"Tavily Error: {e}")
    return combined_results

# --- مسیرها ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_agent', methods=['POST'])
def run_agent():
    if collection is None:
        return jsonify({"error": "خطا در اتصال به دیتابیس"}), 500

    # 1. پاکسازی دیتای قبلی
    try: 
        collection.delete_many({}) 
        print("🧹 Database Cleared for new session.")
    except Exception as e:
        print(f"Delete Error: {e}")

    prompt = request.form.get('prompt')
    file = request.files.get('file')
    
    if not prompt: return jsonify({"error": "سوال وارد نشده است"}), 400

    response_data = {"generated_queries": [], "all_sources": [], "retrieved_chunks": [], "answer": "", "logs": []}
    
    # متغیر برای چک کردن سایز وکتور
    debug_vec_dim = 0

    # 2. پردازش فایل
    if file and file.filename != '':
        response_data["logs"].append("📂 پردازش فایل آپلود شده...")
        try:
            text = file.read().decode('utf-8')
            chunks = recursive_chunk_text(text)
            docs = []
            for ch in chunks:
                emb = get_embedding(ch, task_type="retrieval_document")
                if emb:
                    if debug_vec_dim == 0: debug_vec_dim = len(emb)
                    docs.append({"text": ch, "embedding": emb, "source": "File: " + file.filename})
            if docs:
                collection.insert_many(docs)
                msg = f"✅ {len(docs)} بخش از فایل ذخیره شد. (Vector Dim: {debug_vec_dim})"
                print(msg)
                response_data["logs"].append(msg)
                response_data["all_sources"].append({"title": file.filename, "url": "#", "type": "file"})
        except Exception as e:
            response_data["logs"].append(f"❌ خطا در فایل: {str(e)}")

    # 3. جستجو در وب
    response_data["logs"].append("🌎 تولید کوئری و جستجو در وب...")
    queries = generate_search_queries(prompt)
    response_data["generated_queries"] = queries
    
    search_results = tavily_search(queries)
    
    if search_results:
        web_docs = []
        seen_urls = set()
        for res in search_results:
            if res['url'] not in seen_urls:
                response_data["all_sources"].append({"title": res['title'], "url": res['url'], "type": "web"})
                seen_urls.add(res['url'])
            
            content = res.get('content', '')
            if len(content) < 50: continue # نادیده گرفتن محتوای خیلی کوتاه

            web_chunks = recursive_chunk_text(content, chunk_size=800)
            for ch in web_chunks:
                emb = get_embedding(ch, task_type="retrieval_document")
                if emb:
                    if debug_vec_dim == 0: debug_vec_dim = len(emb)
                    web_docs.append({
                        "text": ch, 
                        "embedding": emb, 
                        "source": res.get('url'),
                        "title": res.get('title')
                    })
        if web_docs:
            collection.insert_many(web_docs)
            msg = f"🌐 {len(web_docs)} تکه دانش از وب ذخیره شد. (Vector Dim: {debug_vec_dim})"
            print(msg)
            response_data["logs"].append(msg)
    
    # --- 4. بازیابی (بخش حیاتی دیباگ) ---
    response_data["logs"].append("🤔 بازیابی اطلاعات مرتبط...")
    
    # چک کردن تعداد داکیومنت‌ها قبل از سرچ
    total_docs = collection.count_documents({})
    print(f"📊 DB STATUS: Total Documents in DB: {total_docs}")
    response_data["logs"].append(f"📊 تعداد کل داده‌ها در دیتابیس: {total_docs}")

    query_emb = get_embedding(prompt, task_type="retrieval_query")
    
    retrieved = []
    if query_emb:
        print(f"❓ Query Vector Dimension: {len(query_emb)}")
        
        # هشدار عدم تطابق ابعاد
        if debug_vec_dim > 0 and len(query_emb) != debug_vec_dim:
             print("🚨 CRITICAL ERROR: Dimension Mismatch!")
             response_data["logs"].append(f"🚨 خطای ابعاد: دیتابیس={debug_vec_dim} ولی کوئری={len(query_emb)}")

        pipeline = [
            {
                "$vectorSearch": {
                    "index": INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_emb,
                    "numCandidates": 100, # افزایش دامنه
                    "limit": 10
                }
            },
            {"$project": {"_id": 0, "text": 1, "source": 1, "title": 1, "score": {"$meta": "vectorSearchScore"}}}
        ]
        
        try:
            retrieved = list(collection.aggregate(pipeline))
            print(f"🎯 Vector Search Results: {len(retrieved)}")
            response_data["retrieved_chunks"] = retrieved
        except Exception as e:
            print(f"❌ Aggregation Error: {e}")
            response_data["logs"].append(f"❌ خطا در جستجوی وکتور: {e}")

    # 5. تولید پاسخ
    response_data["logs"].append("✍️ نگارش پاسخ نهایی...")
    context_text = ""
    
    # اگر وکتور سرچ کار نکرد، حداقل متن خام وب را بدهیم (Fail-safe)
    if not retrieved:
        print("⚠️ Vector search returned 0 results. Using raw fallback.")
        response_data["logs"].append("⚠️ جستجوی وکتوری نتیجه نداشت. استفاده از متن خام.")
        if search_results:
            for res in search_results:
                context_text += f"Source ({res['title']}): {res.get('content', '')[:600]}...\n\n"
        else:
            context_text = "No context found."
    else:
        for doc in retrieved:
            src = doc.get('title', doc.get('source'))
            context_text += f"Source ({src}): {doc['text']}\n\n"

    try:
        final_model = genai.GenerativeModel(GENERATION_MODEL)
        final_prompt = (
            f"User Question: {prompt}\n\n"
            f"Based ONLY on the following context, write a comprehensive answer in Persian (Farsi).\n"
            f"Cite sources inline like [Source Name].\n"
            f"CONTEXT:\n{context_text}"
        )
        answer_resp = final_model.generate_content(final_prompt)
        response_data["answer"] = answer_resp.text
    except Exception as e:
        response_data["answer"] = f"خطا در تولید پاسخ: {str(e)}"

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
