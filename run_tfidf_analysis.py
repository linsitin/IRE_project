import os
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

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

def run_tfidf_candidates_global_df():
    print("\n--- 🚀 [終極版] TF-IDF 候選提名 + 全域 DF 排名 ---")
    texts = get_2025_articles("tfc_reports")
    total_docs = len(texts)
    
    if total_docs == 0: return

    print("🤖 載入 CKIP 斷詞模型 (🚀 已啟用 GPU 硬體加速 device=0)...")
    ws_driver = CkipWordSegmenter(model="albert-base", device=0)
    
    print(f"✂️ 1. 開始對全部 {total_docs} 篇文章進行斷詞...")
    ws_results = ws_driver(texts, batch_size=16)
    
    # stopwords = {"我們", "可以", "因為", "所以", "這個", "那個", "沒有", "什麼", 
    #              "表示", "指出", "影片", "網傳", "照片", "內容", "訊息", "圖片",
    #              "查核", "中心", "台灣", "發現", "結果", "進行", "出現", "相關",
    #              "報導", "部分", "不是", "目前", "流傳", "已經", "可能", "看到"}
    stopwords = { }
    
    corpus = []
    for word_list in ws_results:
        filtered = [w for w in word_list if len(w) > 1 and re.match(r'^[\w\u4e00-\u9fa5]+$', w) and w not in stopwords]
        corpus.append(" ".join(filtered))

    print("🧮 2. 計算全域 TF、DF 與 TF-IDF 矩陣...")
    cv = CountVectorizer()
    tf_matrix = cv.fit_transform(corpus)
    feature_names = cv.get_feature_names_out()
    
    # 計算所有詞彙的「全域標準 DF」(只要有出現就算 1 票)
    global_dfs = np.array((tf_matrix > 0).sum(axis=0))[0]
    
    # 計算 TF-IDF 矩陣
    tfidf_trans = TfidfTransformer()
    tfidf_matrix = tfidf_trans.fit_transform(tf_matrix)

    print("🎯 3. 提取每篇文章的 Top 5 核心關鍵字，建立「菁英白名單」...")
    candidate_pool = set()
    
    for i in range(total_docs):
        row = tfidf_matrix[i].toarray()[0]
        # 抓取每篇文章 TF-IDF 分數最高的 5 個詞
        top_indices = np.argsort(row)[-10:]
        for idx in top_indices:
            if row[idx] > 0:
                candidate_pool.add(feature_names[idx])
                
    print(f"   ➤ 共選出 {len(candidate_pool)} 個不重複的菁英關鍵字！")

    print("🏆 4. 針對菁英白名單，查詢它們的「全域 DF」並進行排名...")
    final_ranking = []
    
    # 把 feature_names 轉成 list 方便找索引
    feature_names_list = list(feature_names)
    
    for word in candidate_pool:
        # 找出這個詞在陣列中的位置，並查出它的全域 DF
        idx = feature_names_list.index(word)
        df_count = global_dfs[idx]
        final_ranking.append((word, df_count))

    # 依照全域 DF 由高到低排序
    final_ranking.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*75)
    print("🌟 2025 年度事實查核報告：TF-IDF 菁英關鍵字之全域 DF 排行榜 Top 20")
    print("(指標：先確保該詞至少是一篇文章的主角，再計算它在『全部文章』的總觸及篇數)")
    print("="*75)
    
    print(f"{'排名':<4} | {'菁英話題關鍵字':<14} | {'全域 DF (總觸及篇數)':<18} | {'話題觸及率 (%)':<10}")
    print("-" * 75)
    
    for rank, (word, df_count) in enumerate(final_ranking[:20], start=1):
        coverage = (df_count / total_docs) * 100
        print(f"第{rank:2d}名 | 【{word:^10}】 | {df_count:^22} | {coverage:^12.1f}%")
        
    print("-" * 75)

if __name__ == "__main__":
    run_tfidf_candidates_global_df()