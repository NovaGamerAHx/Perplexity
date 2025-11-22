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
COLLECTION_NAME = "perplex_context" # کالکشن جدید برای عدم تداخل با قبلی
INDEX_NAME = "vector_index"

genai.configure(api_key=GEMINI_API_KEY)

# اتصال به دیتابیس
try:
    mongo_client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = mongo_client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ DB Connection Error: {e}")

# --- 1. چانک‌بندی اصولی و حرفه‌ای (Recursive with Overlap) ---
def recursive_chunk_text(text, chunk_size=800, overlap=100):
    """
    متن را هوشمندانه تقسیم می‌کند:
    1. اول سعی می‌کند با دو اینتر (پاراگراف) جدا کند.
    2. اگر نشد، با یک اینتر.
    3. اگر نشد، با نقطه (پایان جمله).
    4. نهایتا با فاصله (کلمات).
    همچنین مقداری همپوشانی (Overlap) دارد تا معنی در مرز برش‌ها گم نشود.
    """
    if not text: return []
    
    # تمیز کردن اولیه متن
    text = re.sub(r'\s+', ' ', text).strip()
    
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        # پیدا کردن بهترین نقطه برش (به ترتیب اولویت)
        # تلاش برای برش در پایان جمله (نقطه یا علامت سوال/تعجب)
        block = text[start:end]
        # جستجو برای آخرین جداکننده جمله در نیمه دوم چانک (تا چانک خیلی کوچک نشود)
        split_point = -1
        
        # اولویت ۱: پایان پاراگراف یا جمله
        match = re.search(r'[.!?]\s+', block[::-1]) # جستجو از آخر به اول
        if match:
            split_point = len(block) - match.start()
        
        # اولویت ۲: اگر نقطه نبود، فاصله (Space)
        if split_point == -1:
            last_space = block.rfind(' ')
            if last_space != -1:
                split_point = last_space
        
        # اگر هیچکدام نبود (یک کلمه خیلی طولانی)، برش سخت
        if split_point == -1:
            split_point = chunk_size
            
        # اضافه کردن چانک
        final_chunk = text[start : start + split_point]
        chunks.append(final_chunk)
        
        # حرکت به جلو (با کسر همپوشانی برای حفظ کانتکست)
        start += split_point - overlap
        
    return chunks

# --- 2. ابزارهای کمکی ---
def get_embedding(text):
    try:
        result = genai.embed_content(
            model="models/text-embedding-005",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except:
        return None

def generate_search_queries(prompt):
    """تولید کوئری‌های جستجو با جمینای"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    sys_prompt = (
        f"User prompt: '{prompt}'\n"
        "Generate 3 specific, effective search queries to find information about this prompt on Google. "
        "Return ONLY the queries separated by newlines. Do not number them."
    )
    resp = model.generate_content(sys_prompt)
    return [q.strip() for q in resp.text.split('\n') if q.strip()]

def tavily_search(queries):
    """جستجو در وب با Tavily"""
    combined_results = []
    # فقط کوئری اول و دوم را جستجو میکنیم تا سرعت بالا بماند
    for q in queries[:2]: 
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": q,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
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
    # پاک کردن حافظه قبلی (برای اینکه هر بار یک سرچ جدید باشد)
    # در نسخه واقعی باید Session ID داشته باشیم، ولی برای استفاده شخصی پاک کردن اوکیه
    try:
        collection.delete_many({}) 
    except: pass

    prompt = request.form.get('prompt')
    file = request.files.get('file')
    
    if not prompt:
        return jsonify({"error": "لطفا سوال خود را وارد کنید"}), 400

    steps = [] # لاگ مراحل برای نمایش به کاربر
    
    # --- مرحله ۱: پردازش فایل (اگر باشد) ---
    if file and file.filename != '':
        steps.append("📂 در حال پردازش فایل آپلود شده...")
        text = file.read().decode('utf-8')
        chunks = recursive_chunk_text(text)
        docs = []
        for ch in chunks:
            emb = get_embedding(ch)
            if emb:
                docs.append({"text": ch, "embedding": emb, "source": "File: " + file.filename})
        if docs:
            collection.insert_many(docs)
            steps.append(f"✅ {len(docs)} بخش از فایل ذخیره شد.")

    # --- مرحله ۲: تولید کوئری و جستجو ---
    steps.append("🌎 در حال طراحی کوئری‌های جستجو...")
    queries = generate_search_queries(prompt)
    steps.append(f"🔍 جستجو در وب برای: {queries}")
    
    search_results = tavily_search(queries)
    
    # --- مرحله ۳: پردازش نتایج وب ---
    steps.append(f"🌐 {len(search_results)} صفحه وب پیدا شد. در حال مطالعه و چانک‌بندی...")
    web_docs = []
    for res in search_results:
        # محتوای وب را هم چانک میکنیم
        content = res.get('content', '')
        web_chunks = recursive_chunk_text(content, chunk_size=800, overlap=100)
        for ch in web_chunks:
            emb = get_embedding(ch)
            if emb:
                web_docs.append({
                    "text": ch, 
                    "embedding": emb, 
                    "source": res.get('url'),
                    "title": res.get('title')
                })
    
    if web_docs:
        collection.insert_many(web_docs)
        steps.append(f"🧠 {len(web_docs)} بخش دانش از وب به حافظه اضافه شد.")

    # --- مرحله ۴: بازیابی (Retrieval) ---
    steps.append("🤔 در حال فکر کردن و جمع‌بندی اطلاعات...")
    
    # وکتور سوال اصلی
    query_emb = get_embedding(prompt)
    
    pipeline = [
        {
            "$vectorSearch": {
                "index": INDEX_NAME,
                "path": "embedding",
                "queryVector": query_emb,
                "numCandidates": 100,
                "limit": 8 # 8 تکه مرتبط‌ترین را بردار
            }
        },
        {"$project": {"_id": 0, "text": 1, "source": 1, "title": 1}}
    ]
    retrieved = list(collection.aggregate(pipeline))
    
    # --- مرحله ۵: تولید پاسخ نهایی ---
    context_text = ""
    sources_list = set()
    
    for doc in retrieved:
        source_info = doc.get('title', doc.get('source'))
        context_text += f"Source ({source_info}): {doc['text']}\n\n"
        sources_list.add(doc.get('source'))

    final_model = genai.GenerativeModel('gemini-2.5-flash')
    final_prompt = (
        f"You are an AI research assistant (like Perplexity). \n"
        f"User Question: {prompt}\n\n"
        f"Based ONLY on the following context, write a comprehensive, well-structured answer. "
        f"Cite your sources using [1], [2] etc. if possible, or explicitly mention the source name.\n"
        f"If the context doesn't answer the question, admit it.\n\n"
        f"CONTEXT:\n{context_text}"
    )
    
    answer_response = final_model.generate_content(final_prompt)
    
    return jsonify({
        "steps": steps,
        "answer": answer_response.text,
        "sources": list(sources_list)
    })

if __name__ == '__main__':
    app.run(debug=True)
