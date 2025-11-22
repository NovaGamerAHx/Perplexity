import os
import re
import requests
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)

# --- تنظیمات مدل‌ها (اینجا را دقت کنید) ---
# جدیدترین مدل گوگل در حال حاضر gemini-2.0-flash-exp است
# اگر به ارور خوردید، به gemini-1.5-flash برگردانید
GENERATION_MODEL = "gemini-2.5-flash" 
EMBEDDING_MODEL = "models/text-embedding-004" 

# --- کلیدها و دیتابیس ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
MONGO_URI = os.environ.get("MONGO_URI")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

DB_NAME = "my_rag_db"
COLLECTION_NAME = "perplex_context"
INDEX_NAME = "vector_index"

# تنظیم هوش مصنوعی
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# اتصال به دیتابیس
try:
    if not MONGO_URI:
        print("❌ Error: MONGO_URI is missing.")
        mongo_client = None
        collection = None
    else:
        mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print("✅ Connected to MongoDB")
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
    print(chunks)
    return chunks

def get_embedding(text, task_type="retrieval_document"):
    if not text or not text.strip(): return None
    try:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"⚠️ Embedding Error: {e}")
        return None

def generate_search_queries(prompt):
    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        sys_prompt = (
            f"User prompt: '{prompt}'\n"
            "Generate 3 specific search queries to find information about this prompt. "
            "Return ONLY the queries separated by newlines."
        )
        resp = model.generate_content(sys_prompt)
        return [q.strip() for q in resp.text.split('\n') if q.strip()]
    except Exception as e:
        print(f"Query Gen Error: {e}")
        return [prompt]

def tavily_search(queries):
    if not TAVILY_API_KEY: return []
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

    # پاکسازی دیتای قبلی برای شروع تمیز
    try: collection.delete_many({}) 
    except: pass

    prompt = request.form.get('prompt')
    file = request.files.get('file')
    
    if not prompt: return jsonify({"error": "سوال وارد نشده است"}), 400

    # داده‌هایی که قرار است به فرانت‌اند بفرستیم
    response_data = {
        "generated_queries": [],
        "all_sources": [],
        "retrieved_chunks": [],
        "answer": "",
        "logs": []
    }
    
    # 1. پردازش فایل
    if file and file.filename != '':
        response_data["logs"].append("📂 پردازش فایل آپلود شده...")
        try:
            text = file.read().decode('utf-8')
            chunks = recursive_chunk_text(text)
            docs = []
            for ch in chunks:
                emb = get_embedding(ch, task_type="retrieval_document")
                if emb:
                    docs.append({"text": ch, "embedding": emb, "source": "File: " + file.filename})
            if docs:
                collection.insert_many(docs)
                response_data["all_sources"].append({"title": file.filename, "url": "#", "type": "file"})
                response_data["logs"].append(f"✅ {len(docs)} بخش از فایل ذخیره شد.")
        except Exception as e:
            response_data["logs"].append(f"❌ خطا در فایل: {str(e)}")

    # 2. جستجو در وب
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
            web_chunks = recursive_chunk_text(content, chunk_size=800)
            for ch in web_chunks:
                emb = get_embedding(ch, task_type="retrieval_document")
                if emb:
                    web_docs.append({
                        "text": ch, 
                        "embedding": emb, 
                        "source": res.get('url'),
                        "title": res.get('title')
                    })
        if web_docs:
            collection.insert_many(web_docs)
            response_data["logs"].append(f"🌐 {len(web_docs)} تکه دانش از وب ذخیره شد.")

    # 3. بازیابی (Retrieval)
    response_data["logs"].append("🤔 بازیابی اطلاعات مرتبط...")
    query_emb = get_embedding(prompt, task_type="retrieval_query")
    
    if query_emb is None:
        return jsonify({"error": "خطا در تولید وکتور سوال", "logs": response_data["logs"]}), 500
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_emb,
                "numCandidates": 100,
                "limit": 8
            }
        },
        {"$project": {"_id": 0, "text": 1, "source": 1, "title": 1}}
    ]
    
    retrieved = list(collection.aggregate(pipeline))
    response_data["retrieved_chunks"] = retrieved

    # 4. تولید پاسخ
    response_data["logs"].append("✍️ نگارش پاسخ نهایی...")
    context_text = ""
    if not retrieved:
        context_text = "No specific context found."
    else:
        for doc in retrieved:
            src = doc.get('title', doc.get('source'))
            context_text += f"Source ({src}): {doc['text']}\n\n"

    try:
        final_model = genai.GenerativeModel(GENERATION_MODEL)
        final_prompt = (
            f"User Question: {prompt}\n\n"
            f"Based ONLY on the following context, write a comprehensive answer.\n"
            f"Cite sources inline like [Source Name].\n"
            f"CONTEXT:\n{context_text}"
        )
        answer_resp = final_model.generate_content(final_prompt)
        response_data["answer"] = answer_resp.text
    except Exception as e:
        response_data["answer"] = f"خطا در تولید پاسخ با مدل {GENERATION_MODEL}: {str(e)}"

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)


