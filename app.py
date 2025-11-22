import os
import time
import requests
import re
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from pymongo import MongoClient
from pymongo.server_api import ServerApi

app = Flask(__name__)

# --- تنظیمات ---
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

# --- 1. چانک‌بندی اصولی ---
def recursive_chunk_text(text, chunk_size=800, overlap=100):
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
        if match:
            split_point = len(block) - match.start()
        
        if split_point == -1:
            last_space = block.rfind(' ')
            if last_space != -1:
                split_point = last_space
        
        if split_point == -1:
            split_point = chunk_size
            
        final_chunk = text[start : start + split_point]
        chunks.append(final_chunk)
        start += split_point - overlap
        
    return chunks

# --- 2. ابزارهای کمکی (اصلاح شده) ---

def get_embedding(text, task_type="retrieval_document"):
    """
    تولید وکتور با قابلیت هندل کردن خطا
    task_type: میتونه 'retrieval_document' (برای متن‌ها) یا 'retrieval_query' (برای سوال) باشه
    """
    if not text or not text.strip():
        return None
        
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"⚠️ Embedding Error for text '{text[:30]}...': {e}")
        return None

def generate_search_queries(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        sys_prompt = (
            f"User prompt: '{prompt}'\n"
            "Generate 3 specific search queries to find information about this prompt. "
            "Return ONLY the queries separated by newlines."
        )
        resp = model.generate_content(sys_prompt)
        return [q.strip() for q in resp.text.split('\n') if q.strip()]
    except:
        return [prompt] # اگر مدل خطا داد، خود سوال را جستجو کن

def tavily_search(queries):
    if not TAVILY_API_KEY:
        print("⚠️ Tavily Key missing")
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

# --- مسیرهای سایت ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/run_agent', methods=['POST'])
def run_agent():
    if collection is None:
        return jsonify({"error": "خطا در اتصال به پایگاه داده"}), 500

    # پاکسازی دیتای قبلی
    try:
        collection.delete_many({}) 
    except Exception as e:
        print(f"Delete Error: {e}")

    prompt = request.form.get('prompt')
    file = request.files.get('file')
    
    if not prompt:
        return jsonify({"error": "سوال وارد نشده است"}), 400

    steps = []
    
    # --- مرحله ۱: پردازش فایل ---
    if file and file.filename != '':
        steps.append("📂 پردازش فایل آپلود شده...")
        try:
            text = file.read().decode('utf-8')
            chunks = recursive_chunk_text(text)
            docs = []
            for ch in chunks:
                # اینجا مهم: نوع تسک را داکیومنت می‌گذاریم
                emb = get_embedding(ch, task_type="retrieval_document")
                if emb:
                    docs.append({"text": ch, "embedding": emb, "source": "File: " + file.filename})
            if docs:
                collection.insert_many(docs)
                steps.append(f"✅ {len(docs)} بخش از فایل ذخیره شد.")
        except Exception as e:
            steps.append(f"❌ خطا در پردازش فایل: {str(e)}")

    # --- مرحله ۲: جستجو در وب ---
    steps.append("🌎 جستجو در وب...")
    queries = generate_search_queries(prompt)
    search_results = tavily_search(queries)
    
    if search_results:
        steps.append(f"🌐 {len(search_results)} منبع پیدا شد.")
        web_docs = []
        for res in search_results:
            content = res.get('content', '')
            web_chunks = recursive_chunk_text(content, chunk_size=800, overlap=100)
            for ch in web_chunks:
                # اینجا هم نوع تسک داکیومنت است
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

    # --- مرحله ۳: بازیابی (Retrieval) ---
    steps.append("🤔 تحلیل نهایی...")
    
    # اصلاح حیاتی: اینجا نوع تسک را 'query' می‌گذاریم و چک می‌کنیم None نباشد
    query_emb = get_embedding(prompt, task_type="retrieval_query")
    
    if query_emb is None:
        return jsonify({
            "error": "خطا در تولید وکتور برای سوال. لطفا دوباره تلاش کنید.",
            "steps": steps
        }), 500
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_emb, # این مقدار نباید None باشد
                "numCandidates": 100,
                "limit": 8
            }
        },
        {"$project": {"_id": 0, "text": 1, "source": 1, "title": 1}}
    ]
    
    try:
        retrieved = list(collection.aggregate(pipeline))
    except Exception as e:
        print(f"Aggregation Error: {e}")
        return jsonify({"error": f"خطای دیتابیس: {str(e)}", "steps": steps}), 500
    
    # --- مرحله ۴: تولید پاسخ ---
    context_text = ""
    sources_list = set()
    
    if not retrieved:
        context_text = "No specific context found within the allowed sources."
    else:
        for doc in retrieved:
            source_info = doc.get('title', doc.get('source'))
            context_text += f"Source ({source_info}): {doc['text']}\n\n"
            sources_list.add(doc.get('source'))

    try:
        final_model = genai.GenerativeModel('gemini-1.5-flash')
        final_prompt = (
            f"User Question: {prompt}\n\n"
            f"Based ONLY on the following context, write a detailed answer with citations.\n"
            f"CONTEXT:\n{context_text}"
        )
        answer_response = final_model.generate_content(final_prompt)
        answer_text = answer_response.text
    except Exception as e:
        answer_text = f"خطا در تولید پاسخ نهایی: {str(e)}"
    
    return jsonify({
        "steps": steps,
        "answer": answer_text,
        "sources": list(sources_list)
    })

if __name__ == '__main__':
    app.run(debug=True)
