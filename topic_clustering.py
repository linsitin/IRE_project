import pickle
import numpy as np
import random
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# 設定區塊
TFIDF_DB = "tfc_database_maxdf0.5.pkl" # 請確認檔名
SBERT_DB = "sbert_model_data.pkl"
TARGET_CLUSTERS = 44                    # 強制設定為 44 群以對標 HDBSCAN

def get_cluster_table(cluster_labels, valid_indices, corpus_tokens):
    """印出清爽的群集列表"""
    unique_labels = set(cluster_labels)
    cluster_sizes = {label: list(cluster_labels).count(label) for label in unique_labels}
    sorted_clusters = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)

    print(f"{'排名':<4} | {'群集ID':<6} | {'篇數':<6} | 群集核心關鍵字 (Top 5)")
    print("-" * 70)

    for rank, cluster_id in enumerate(sorted_clusters, 1):
        size = cluster_sizes[cluster_id]
        
        # 找出屬於這個群集的所有文章位置
        indices_in_cluster = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
        
        # 統計群內詞頻
        cluster_words = []
        for i in indices_in_cluster:
            real_idx = valid_indices[i]
            words = [w for w in corpus_tokens[real_idx] if len(w) >= 2] 
            cluster_words.extend(words)
            
        counter = Counter(cluster_words)
        top_words = [word for word, count in counter.most_common(5)]
        auto_keywords = "、".join(top_words)

        print(f"No.{str(rank).ljust(2)} | 群集 {str(cluster_id).ljust(2)} | {str(size).ljust(4)} 篇 | {auto_keywords}")

def manual_check_loop(cluster_labels, valid_indices, titles, dates, method_name):
    """人工抽查迴圈"""
    print("\n" + "="*70)
    print(f" [{method_name} 人工抽查模式] 輸入群集 ID，隨機檢驗文章關聯性")
    print("="*70)
    
    while True:
        target = input(f"\n請輸入想抽查的『群集 ID』(輸入 q 結束 {method_name} 測試): ").strip()
        if target.lower() == 'q':
            print(f"結束 {method_name} 抽查模式。")
            break
            
        try:
            target_id = int(target)
            if target_id not in cluster_labels:
                print(f"找不到群集 {target_id}，請參考上方列表輸入正確的 ID (注意：是群集ID，不是排名)。")
                continue
                
            indices = [i for i, label in enumerate(cluster_labels) if label == target_id]
            
            print(f"\n[{method_name} - 群集 {target_id}] 共有 {len(indices)} 篇文章。隨機抽查最多 10 篇：")
            print("-" * 70)
            
            # 隨機抽取最多 10 篇
            sample_size = min(10, len(indices))
            sampled_indices = random.sample(indices, sample_size)
            
            for rank, idx in enumerate(sampled_indices, 1):
                real_idx = valid_indices[idx]
                title_preview = titles[real_idx][:65] + "..." if len(titles[real_idx]) > 65 else titles[real_idx]
                print(f"{rank:2d}. [{dates[real_idx]}] {title_preview}")
                
            print("-" * 70)
            
        except ValueError:
            print("指令錯誤，請輸入整數的『群集 ID』。")

def run_dual_kmeans_test():
    # 1. 載入資料庫
    try:
        with open(TFIDF_DB, 'rb') as f: tfidf_data = pickle.load(f)
        with open(SBERT_DB, 'rb') as f: sbert_data = pickle.load(f)
    except FileNotFoundError:
        print("找不到資料庫，請確認 pkl 檔案是否存在！")
        return

    sbert_vectors = sbert_data['sbert_doc_vectors']
    tfidf_matrix = tfidf_data['tfidf_matrix']
    corpus_tokens = tfidf_data['corpus']
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
    sub_tfidf = tfidf_matrix[valid_indices]

    # ==========================================
    # 測試一：TF-IDF + K-Means
    # ==========================================
    print("\n" + "="*80)
    print(f" [測試一] TF-IDF + K-Means (尋找 {TARGET_CLUSTERS} 群)")
    print("="*80)
    
    kmeans_tfidf = KMeans(n_clusters=TARGET_CLUSTERS, random_state=42, n_init=10)
    labels_tfidf = kmeans_tfidf.fit_predict(sub_tfidf)
    
    # 印出列表
    get_cluster_table(labels_tfidf, valid_indices, corpus_tokens)
    
    # 進入 TF-IDF 抽查模式
    manual_check_loop(labels_tfidf, valid_indices, titles, dates, "TF-IDF")

    # 暫停一下，準備進入 SBERT
    input("\n▶ 按 Enter 鍵繼續執行 測試二 (SBERT 版本)...")

    # ==========================================
    # 測試二：SBERT + K-Means
    # ==========================================
    print("\n" + "="*80)
    print(f" [測試二] SBERT + K-Means (尋找 {TARGET_CLUSTERS} 群)")
    print("="*80)
    
    kmeans_sbert = KMeans(n_clusters=TARGET_CLUSTERS, random_state=42, n_init=10)
    labels_sbert = kmeans_sbert.fit_predict(sub_sbert)
    
    # 印出列表
    get_cluster_table(labels_sbert, valid_indices, corpus_tokens)
    
    # 進入 SBERT 抽查模式
    manual_check_loop(labels_sbert, valid_indices, titles, dates, "SBERT")


if __name__ == "__main__":
    run_dual_kmeans_test()