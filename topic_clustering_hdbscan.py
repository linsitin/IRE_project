import pickle
import numpy as np
import umap
import hdbscan
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import time
import json
import os

# Initialize the new genai client
# Replace 'YOUR_API_KEY_HERE' with your actual key
client = genai.Client(api_key="YOUR_API_KEY_HERE")

# Define the cache file
CACHE_FILE = "llm_topic_names_cache.json"

def load_llm_cache():
    """讀取本地的命名快取檔案"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_llm_cache(cache_data):
    """將命名結果存入本地檔案"""
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=4)

def generate_topic_name_with_llm(keywords, representative_titles):
    """將特徵丟給 LLM 產生主題名稱，搭載指數退避重試機制"""
    prompt = f"""
    你是一位專業的台灣事實查核中心資深主編。
    我正在使用機器學習對歷年的查核報告進行主題分群。以下是其中一個群集的特徵：

    [高頻關鍵字]: {', '.join(keywords)}
    
    [核心代表文章標題]:
    1. {representative_titles[0]}
    2. {representative_titles[1]}
    3. {representative_titles[2] if len(representative_titles) > 2 else ''}

    請根據上述資訊，為這個群集命名。
    規則：
    1. 必須極度精準、客觀且具備專業感。
    2. 長度嚴格限制在 10 個中文字以內。
    3. 請直接輸出名稱字串，不要有任何引言、引號、句號或額外的解釋。
    """
    
    wait_time = 5  # 初始等待時間 (秒)
    
    while True: # 無限重試，直到成功為止
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=prompt,
            )
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                print(f"       [系統提示] 觸發 API 流量限制，退避等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
                wait_time *= 2  # 指數退避：下次等待時間加倍 (5 -> 10 -> 20 -> 40...)
                
                # 設定等待時間上限，避免等太久 (例如最長等 60 秒)
                if wait_time > 60:
                    wait_time = 60
            else:
                # 如果是其他類型的錯誤 (例如網路斷線)，印出錯誤並回傳失敗
                print(f"       [LLM 呼叫失敗]: {e}")
                return "自動命名失敗"

# 資料庫路徑
TFIDF_DB = "tfc_database_maxdf0.5.pkl" # 請確認你的資料庫檔名
SBERT_DB = "sbert_model_data.pkl"

def run_hdbscan_clustering_pro():
    print("\n" + "="*60)
    print(" [功能3] SBERT + UMAP + HDBSCAN 主題探勘與檢驗")
    print("="*60)
    
    # 1. 載入資料庫
    try:
        with open(TFIDF_DB, 'rb') as f: tfidf_data = pickle.load(f)
        with open(SBERT_DB, 'rb') as f: sbert_data = pickle.load(f)
    except FileNotFoundError:
        print("找不到資料庫，請確認 pkl 檔案是否存在！")
        return

    sbert_vectors = sbert_data['sbert_doc_vectors']
    corpus_tokens = tfidf_data['corpus'] # 已經斷好詞的內文
    titles = tfidf_data['titles']
    dates = tfidf_data['dates']

    # 2. 過濾資料 (只取 2019~2025)
    valid_indices = []
    for i, date_str in enumerate(dates):
        if date_str == "未知日期": continue
        year = date_str[:4]
        if '2019' <= year <= '2025':
            valid_indices.append(i)

    print(f"成功篩選 {len(valid_indices)} 篇有效報告。")
    sub_sbert = sbert_vectors[valid_indices]

    # 3. 核心演算法：降維與分群
    print("\n步驟一：UMAP 語意空間降維中...")
    umap_model = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=5, metric='cosine', random_state=42)
    reduced_embeddings = umap_model.fit_transform(sub_sbert)

    print("步驟二：HDBSCAN 密度分群與穩定度計算中...")
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=10, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = hdbscan_model.fit_predict(reduced_embeddings)
    
    # 取得群集穩定度 (Persistence)
    cluster_persistence = hdbscan_model.cluster_persistence_

    # 4. 分析所有群集 (顯示 Noise Ratio)
    unique_labels = set(cluster_labels)
    total_docs = len(cluster_labels)
    noise_count = list(cluster_labels).count(-1)
    noise_ratio = (noise_count / total_docs) * 100
    
    cluster_sizes = {label: list(cluster_labels).count(label) for label in unique_labels if label != -1}
    sorted_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)

    print("\n" + "-"*60)
    print(f"系統共探測到 {len(sorted_clusters)} 個主題聚落。")
    print(f"成功過濾雜訊：{noise_count} 篇 (佔總體 {noise_ratio:.2f}%)")
    print("-"*60 + "\n")

    # 讀取已經命名過的快取紀錄
    llm_topic_cache = load_llm_cache()
    # 記錄本次是否有新增命名，以便稍後存檔
    cache_updated = False

    # 手動命名區 (若有設定則優先顯示)
    manual_topic_names = {
        # 30: "選舉與投票爭議",
    }

    # 5. 輸出群集詳細資訊
    for rank, cluster_id in enumerate(sorted_clusters, 1):
        size = cluster_sizes[cluster_id]
        
        # 找出屬於這個群集的所有文章在 valid_indices 中的相對位置
        indices_in_cluster = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        # 取得該群集的穩定度
        stability = cluster_persistence[cluster_id] if cluster_id < len(cluster_persistence) else 0.0

        # 使用 Counter 統計群內詞頻
        cluster_words = []
        for i in indices_in_cluster:
            real_idx = valid_indices[i]
            # 過濾掉長度小於2的字
            words = [w for w in corpus_tokens[real_idx] if len(w) >= 2] 
            cluster_words.extend(words)
            
        counter = Counter(cluster_words)
        top_words = [word for word, count in counter.most_common(5)]
        auto_keywords = "、".join(top_words)


        # 尋找群集幾何中心，抽出「最核心代表文章」
        cluster_vectors = reduced_embeddings[indices_in_cluster]
        centroid = cluster_vectors.mean(axis=0)
        
        sims = cosine_similarity(cluster_vectors, centroid.reshape(1, -1)).flatten()
        top_idx = np.argsort(sims)[::-1][:3] 

        # 收集核心代表文章的標題，準備餵給 LLM
        rep_titles = []
        for idx in top_idx:
            real_idx = valid_indices[indices_in_cluster[idx]]
            rep_titles.append(titles[real_idx])

        # LLM 自動命名邏輯 (搭配快取機制)
        str_cluster_id = str(cluster_id)
        
        if str_cluster_id in llm_topic_cache:
            # 如果快取裡有，直接使用
            llm_topic_name = llm_topic_cache[str_cluster_id]
        else:
            # 如果沒有，呼叫 LLM
            print(f"正在請 LLM 分析 群集 {cluster_id} 的命名...")
            llm_topic_name = generate_topic_name_with_llm(top_words, rep_titles)
            
            # 加入防呆機制：只有當命名成功時，才存入快取
            if llm_topic_name != "自動命名失敗":
                llm_topic_cache[str_cluster_id] = llm_topic_name
                cache_updated = True
            

        # 組合顯示名稱
        display_name = f"[{llm_topic_name}] (AI關鍵字: {auto_keywords})"

        # 印出群集標頭
        print(f"No.{str(rank).ljust(2)} | 群集 {str(cluster_id).ljust(2)} | {str(size).rjust(3)} 篇 | 穩定度: {stability:.2f} | {display_name}")

        # 印出代表文章
        print("[核心代表文章]:")
        for title in rep_titles:
            title_preview = title[:50] + "..." if len(title) > 50 else title
            print(f"- {title_preview}")
        print("-" * 60)

    # 迴圈結束後，如果有呼叫過 LLM，就把新的結果存入硬碟
    if cache_updated:
        save_llm_cache(llm_topic_cache)
        print("已將新的 LLM 命名結果儲存至快取檔案。")

    # [人工驗證模式]：抽查群集內的真實文章
    import random
    
    print("\n" + "="*60)
    print(" [人工抽查模式] 隨機驗證分群結果的真實性")
    print("="*60)
    
    while True:
        target = input("\n請輸入你想抽查的『群集 ID』(輸入 q 離開): ").strip()
        if target.lower() == 'q':
            print("離開抽查模式。")
            break
            
        try:
            target_id = int(target)
            if target_id not in cluster_labels:
                print(f"找不到群集 {target_id}，請確認輸入了正確的群集 ID。")
                continue
                
            # 找出屬於這個群集的所有文章在 valid_indices 中的相對位置
            indices = [i for i, label in enumerate(cluster_labels) if label == target_id]
            
            print(f"\n[群集 {target_id}] 共有 {len(indices)} 篇文章。為您隨機抽查 10 篇：")
            print("-" * 60)
            
            # 隨機抽取最多 10 篇
            sample_size = min(10, len(indices))
            sampled_indices = random.sample(indices, sample_size)
            
            for rank, idx in enumerate(sampled_indices, 1):
                # 將相對位置轉換回原始資料庫的真實位置
                real_idx = valid_indices[idx]
                
                # 印出發布日期與標題
                title_preview = titles[real_idx][:60] + "..." if len(titles[real_idx]) > 60 else titles[real_idx]
                print(f"{rank}. [{dates[real_idx]}] {title_preview}")
                
            print("-" * 60)
            
        except ValueError:
            print("指令錯誤，請輸入整數數字 ID。")
if __name__ == "__main__":
    run_hdbscan_clustering_pro()