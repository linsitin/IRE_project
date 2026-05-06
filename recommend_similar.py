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
    
    print(f" 載入 2025 年事實查核資料庫 ...")
    for filename in file_list:
        if "[2025" in filename:
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 抓取第一行的標題
                title = content.split('\n')[0].replace('標題：', '').strip()
                
                text_parts = content.split("==================================================")
                if len(text_parts) > 1:
                    main_text = text_parts[-1].strip()
                else:
                    main_text = content.strip()
                    
                # 簡單清理可能的殘留雜訊
                main_text = re.sub(r'(發佈|更新|查核記者|責任編輯|記者|編輯)[：:].+', '', main_text)
                
                if main_text:
                    articles.append(main_text)
                    titles.append(title)
                
    return articles, titles

def run_recommendation_system():
    print("\n---  [功能二] 相似文章推薦系統  ---")
    
    texts, titles = get_2025_articles("tfc_reports_api")
    total_docs = len(texts)
    
    if total_docs == 0: 
        print("❌ 找不到 2025 年的文章，請確認爬蟲是否有抓到資料！")
        return

    print(" 載入 CKIP 斷詞模型 ( 已啟用 GPU 硬體加速 device=0)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    print(f" 1. 開始對資料庫 {total_docs} 篇文章進行斷詞...")
    ws_results = ws_driver(texts, batch_size=16)
    
    stopwords = {"我們", "可以", "因為", "所以", "這個", "那個", "沒有", "什麼", 
                 "表示", "指出", "影片", "網傳", "照片", "內容", "訊息", "圖片",
                 "查核", "中心", "台灣", "發現", "結果", "進行", "出現", "相關",
                 "報導", "部分", "不是", "目前", "流傳", "已經", "可能", "看到"}
    
    corpus = []
    for word_list in ws_results:
        filtered = [w for w in word_list if len(w) > 1 and re.match(r'^[\w\u4e00-\u9fa5]+$', w) and w not in stopwords]
        corpus.append(" ".join(filtered))

    print(" 2. 計算全體資料庫的 TF-IDF 矩陣...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # 核心功能：輸入一篇目標文章，找出最相似的 N 篇
    
    # 假設使用者現在正在看第 幾 篇文章
    target_idx = 287 
    
    # 避免索引超出範圍的防呆機制
    if target_idx >= total_docs:
        print(f" 警告：資料庫只有 {total_docs} 篇文章，請將 target_idx 設定在 0 到 {total_docs-1} 之間。")
        return
        
    target_title = titles[target_idx]
    
    print("\n" + "="*70)
    print(f" 使用者正在閱讀：【 {target_title[:30]}... 】")
    print("="*70)
    print(" 系統正在比對所有文章，尋找推薦名單...\n")
    
    # 抓出目標文章的 TF-IDF 向量
    target_vector = tfidf_matrix[target_idx]
    
    #  計算餘弦相似度
    similarity_scores = cosine_similarity(target_vector, tfidf_matrix)[0]
    
    # 將分數由高到低排序 (抓出索引值)
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    print(" 【為您推薦以下相似查核報告】 ")
    print("-" * 70)
    
    recommend_count = 0
    for idx in ranked_indices:
        score = similarity_scores[idx]
        
        # 條件 1：不能推薦自己給自己
        # 條件 2：只顯示相似度大於 0.1 的文章
        if idx != target_idx and score > 0.1:
            recommend_count += 1
            print(f" 推薦 {recommend_count} | 相似度：{score:.3f} | {titles[idx][:40]}...")
            
        # 我們只推薦前 3 篇
        if recommend_count >= 3:
            break
            
    if recommend_count == 0:
        print(" 抱歉，資料庫中目前沒有與此主題高度相關的其他報告。")
    print("-" * 70)

if __name__ == "__main__":
    run_recommendation_system()