import os
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ckip_transformers.nlp import CkipWordSegmenter
except ImportError:
    print("請安裝 ckip-transformers")
    exit()

def get_2025_articles(folder_path):
    articles = []
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    print(f"📂 開始利用「報告編號 (3416~3962)」精準篩選 2025 年資料...")
    for filename in file_list:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'報告編號[：:]\s*(\d+)', content)
            if match and (3416 <= int(match.group(1)) <= 3962):
                text_parts = content.split("--------------------------------------------------")
                main_text = text_parts[-1].strip() if len(text_parts) > 1 else content
                main_text = re.sub(r'(發佈|更新|報告編號|查核記者|責任編輯|記者|編輯)[：:].+', '', main_text)
                articles.append(main_text)
                
    return articles

def run_hybrid_trend_analysis_top5():
    print("\n--- 🚀 [高階] TF-IDF 提純 (Top 5) + DF 趨勢分析 ---")
    texts = get_2025_articles("tfc_reports")
    total_docs = len(texts)
    
    if total_docs == 0: return

    print("🤖 載入 CKIP 斷詞模型 (🚀 已啟用 GPU 硬體加速 device=0)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    print(f"✂️ 1. 開始對全部 {total_docs} 篇文章進行斷詞...")
    ws_results = ws_driver(texts, batch_size=16)
    
    stopwords = {"我們", "可以", "因為", "所以", "這個", "那個", "沒有", "什麼", 
                 "表示", "指出", "影片", "網傳", "照片", "內容", "訊息", "圖片",
                 "查核", "中心", "台灣", "發現", "結果", "進行", "出現", "相關",
                 "報導", "部分", "不是", "目前", "流傳", "已經", "可能", "看到"}
    
    corpus = []
    for word_list in ws_results:
        filtered = [w for w in word_list if len(w) > 1 and re.match(r'^[\w\u4e00-\u9fa5]+$', w) and w not in stopwords]
        corpus.append(" ".join(filtered))

    print("🧮 2. 計算 TF-IDF 矩陣...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 這裡的文字說明改為 Top 5
    print("🎯 3. 嚴格提取每篇文章的 Top 5 核心關鍵字，並進行大會師統計...")
    
    trend_counter = Counter()
    
    for i in range(total_docs):
        row = tfidf_matrix[i].toarray()[0]
        
        # 🌟 關鍵修改：這裡從 [-10:] 改成了 [-5:] 🌟
        # 代表我們只取分數最高的最後 5 個索引值
        top_indices = np.argsort(row)[-5:]
        
        top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        trend_counter.update(top_words)

    print("\n" + "="*70)
    print("🏆 2025 年度事實查核報告：最極致核心話題趨勢 Top 20")
    print("(指標：該詞彙成為單篇文章『Top 5 絕對核心關鍵字』的總篇數)")
    print("="*70)
    
    top_trends = trend_counter.most_common(20)
    
    print(f"{'排名':<4} | {'絕對核心話題':<12} | {'成為絕對核心的篇數':<14} | {'話題強度 (%)':<10}")
    print("-" * 70)
    
    for rank, (word, count) in enumerate(top_trends, start=1):
        power = (count / total_docs) * 100
        print(f"第{rank:2d}名 | 【{word:^8}】 | {count:^18} | {power:^10.1f}%")
        
    print("-" * 70)

if __name__ == "__main__":
    run_hybrid_trend_analysis_top5()