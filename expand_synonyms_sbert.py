import pickle
import numpy as np
import unicodedata
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

TFIDF_DB = "tfc_database_maxdf0.5.pkl" # 請確認你的檔名

def pad_str(s, total_width, align='left'):
    s = str(s)
    width = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)
    padding = total_width - width
    if padding <= 0: return s
    if align == 'center': return ' ' * (padding // 2) + s + ' ' * (padding - padding // 2)
    return s + ' ' * padding

def run_global_synonym_expansion(top_n=100, similarity_threshold=0.75):
    print("=" * 80)
    print(f"啟動 SBERT 全域同義詞擴充引擎 (種子關鍵字：Top {top_n})")
    print("=" * 80)

    # 1. 載入資料庫與詞彙表
    print("正在載入 TF-IDF 詞彙資料庫...")
    try:
        with open(TFIDF_DB, 'rb') as f:
            db_data = pickle.load(f)
    except FileNotFoundError:
        print(f"找不到 {TFIDF_DB}，請確認檔案路徑。")
        return

    tfidf_matrix = db_data['tfidf_matrix']
    vectorizer = db_data['vectorizer']
    dates = db_data['dates']
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 2. 建立候選字池 (Candidate Pool)
    print("正在建立全域候選字池...")
    candidate_words = [w for w in feature_names if len(w) >= 2]
    print(f"有效候選詞彙總數：{len(candidate_words)} 個")

    # 3. 統一標準：使用 2019-2025 的文章投票制找出 Top N 種子字
    print(f"正在使用「文章投票制」計算 2019-2025 權重最高的 {top_n} 個種子關鍵字...")
    
    # 步驟 3-A: 過濾日期
    global_indices = []
    for i, date_str in enumerate(dates):
        if date_str == "未知日期" or date_str < '2019' or date_str >= '2026': 
            continue
        global_indices.append(i)
        
    # 步驟 3-B: 執行每篇文章獨立投票
    trend_counter = Counter()
    for i in global_indices:
        row = tfidf_matrix[i].toarray()[0]
        top_indices = np.argsort(row)[-10:]
        top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        trend_counter.update(top_words)
        
    # 步驟 3-C: 結算出前 N 名
    seed_words = [word for word, count in trend_counter.most_common(top_n)]

    # (接下來的程式碼與原本完全相同)
    # 4. 載入 SBERT 模型
    print("\n正在載入 SBERT 模型 (這需要幾秒鐘)...")
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    
    # 5. 向量化運算
    print("正在將種子字轉換為語意向量...")
    seed_vectors = model.encode(seed_words)

    print(f"正在將 {len(candidate_words)} 個候選字轉換為語意向量 (這可能需要 1~3 分鐘，請耐心等候)...")
    candidate_vectors = model.encode(candidate_words, show_progress_bar=True, batch_size=256)

    # 6. 計算矩陣相似度
    print("\n正在進行跨維度同義詞比對...")
    sim_matrix = cosine_similarity(seed_vectors, candidate_vectors)

    # 7. 輸出結果
    print("\n" + "=" * 80)
    print(f"全域同義詞擴充結果 (相似度門檻：{similarity_threshold})")
    print("=" * 80)
    
    found_any = False
    
    for i, seed_word in enumerate(seed_words):
        scores = sim_matrix[i]
        ranked_indices = np.argsort(scores)[::-1]
        
        synonyms = []
        for idx in ranked_indices:
            candidate = candidate_words[idx]
            score = scores[idx]
            
            if score < similarity_threshold:
                break
                
            if candidate == seed_word:
                continue
                
            synonyms.append(f"{candidate}({score:.2f})")
            
        if synonyms:
            found_any = True
            print(f"第 {pad_str(i+1, 3)} 名種子: 【 {pad_str(seed_word, 10)} 】")
            print(f"  └─ 發現同義詞: {', '.join(synonyms)}")
            print("-" * 80)

    if not found_any:
        print("在目前的門檻下，沒有為這些種子字找到任何全域同義詞。")

if __name__ == "__main__":
    run_global_synonym_expansion(top_n=100, similarity_threshold=0.75)