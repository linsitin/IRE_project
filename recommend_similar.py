import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from ckip_transformers.nlp import CkipWordSegmenter
except ImportError:
    print("請安裝 ckip-transformers")
    exit()

def get_2025_articles(folder_path):
    articles = []
    titles = []
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    
    print(f"📂 載入 2025 年事實查核資料庫...")
    for filename in file_list:
        filepath = os.path.join(folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'報告編號[：:]\s*(\d+)', content)
            if match and (3416 <= int(match.group(1)) <= 3962):
                title = content.split('\n')[0].replace('標題：', '').strip()
                text_parts = content.split("--------------------------------------------------")
                main_text = text_parts[-1].strip() if len(text_parts) > 1 else content
                main_text = re.sub(r'(發佈|更新|報告編號|查核記者|責任編輯|記者|編輯)[：:].+', '', main_text)
                
                articles.append(main_text)
                titles.append(title)
                
    return articles, titles

def run_recommendation_system():
    print("\n--- 🚀 [功能二] 相似文章推薦系統 ---")
    texts, titles = get_2025_articles("tfc_reports")
    total_docs = len(texts)
    
    if total_docs == 0: return

    print("🤖 載入 CKIP 斷詞模型 (🚀 已啟用 GPU 硬體加速 device=0)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    print(f"✂️ 1. 開始對資料庫 {total_docs} 篇文章進行斷詞...")
    ws_results = ws_driver(texts, batch_size=16)
    
    stopwords = {"我們", "可以", "因為", "所以", "這個", "那個", "沒有", "什麼", 
                 "表示", "指出", "影片", "網傳", "照片", "內容", "訊息", "圖片",
                 "查核", "中心", "台灣", "發現", "結果", "進行", "出現", "相關",
                 "報導", "部分", "不是", "目前", "流傳", "已經", "可能", "看到"}
    
    corpus = []
    for word_list in ws_results:
        filtered = [w for w in word_list if len(w) > 1 and re.match(r'^[\w\u4e00-\u9fa5]+$', w) and w not in stopwords]
        corpus.append(" ".join(filtered))

    print("🧮 2. 計算全體資料庫的 TF-IDF 矩陣...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # ==========================================
    # 🎯 核心功能：輸入一篇目標文章，找出最相似的 N 篇
    # ==========================================
    
    # 假設使用者現在正在看第 0 篇文章 (你可以隨便改數字測試，例如 target_idx = 5)
    target_idx = 5 
    target_title = titles[target_idx]
    
    print("\n" + "="*70)
    print(f"👀 使用者正在閱讀：【 {target_title[:30]}... 】")
    print("="*70)
    print("🔍 系統正在比對數千維度的語意向量，尋找推薦名單...\n")
    
    # 抓出目標文章的 TF-IDF 向量
    target_vector = tfidf_matrix[target_idx]
    
    # 🌟 魔法函數：計算目標文章與「資料庫裡所有文章」的餘弦相似度
    # 這會回傳一個陣列，裡面包含了目標文章與第0篇、第1篇...第N篇的相似度分數
    similarity_scores = cosine_similarity(target_vector, tfidf_matrix)[0]
    
    # 將分數由高到低排序 (抓出索引值)
    # argsort 會由小排到大，加上 [::-1] 把它反轉成由大排到小
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    print("✨ 【為您推薦以下相似查核報告】 ✨")
    print("-" * 70)
    
    recommend_count = 0
    for idx in ranked_indices:
        score = similarity_scores[idx]
        
        # 條件 1：不能推薦自己給自己 (相似度通常是 1.0)
        # 條件 2：只顯示相似度大於 0.1 (10%) 的文章，避免推薦完全無關的
        if idx != target_idx and score > 0.1:
            recommend_count += 1
            print(f"💡 推薦 {recommend_count} | 相似度：{score*100:5.1f}% | {titles[idx][:40]}...")
            
        # 我們只推薦前 3 篇
        if recommend_count >= 3:
            break
            
    if recommend_count == 0:
        print("🤷‍♂️ 抱歉，資料庫中目前沒有與此主題高度相關的其他報告。")
    print("-" * 70)

if __name__ == "__main__":
    run_recommendation_system()