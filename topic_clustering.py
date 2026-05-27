import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 全域字體放大設定 (你在 PPT 展示時最需要的功能)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14 

DB_FILE = "tfc_database_maxdf0.33.pkl" 

def run_topic_clustering(n_clusters=5):
    print(f"\n---  [功能三] AI 自動主題分群與趨勢追蹤 (K-Means) ---")
    
    # 1. 載入資料庫
    try:
        with open(DB_FILE, 'rb') as f:
            db_data = pickle.load(f)
    except FileNotFoundError:
        print(f" 找不到 {DB_FILE}！")
        return

    tfidf_matrix = db_data['tfidf_matrix']
    vectorizer = db_data['vectorizer']
    dates = db_data['dates']
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 2. 過濾資料與建立時間標籤 (只取 2019~2025)
    valid_indices = []
    q_labels = []
    
    for i, date_str in enumerate(dates):
        if date_str == "未知日期": continue
        year = date_str[:4]
        if year <= '2018' or year >= '2026': continue # 過濾頭尾雜訊
            
        month_str = date_str[5:7]
        if not month_str.isdigit(): continue
            
        # 計算季度
        month = int(month_str)
        q_label = f"{year}-Q{((month - 1) // 3) + 1}"
        
        valid_indices.append(i)
        q_labels.append(q_label)

    print(f" 成功載入 {len(valid_indices)} 篇有效報告 (2019-2025)。")
    
    # 取出有效文章的 TF-IDF 子矩陣
    sub_tfidf_matrix = tfidf_matrix[valid_indices]

    # 3. 執行 K-Means 機器學習分群
    print(f" 正在啟動 K-Means 演算法，將文章分為 {n_clusters} 大主題叢集 (這可能需要幾秒鐘)...")
    # random_state=42 確保每次跑出來的分群結果顏色跟標籤都一樣，方便你 Demo
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(sub_tfidf_matrix)

    # 4. AI 自動為這 5 大主題「命名」
    print("\n" + "="*70)
    print("  AI 自動提煉的 5 大主題標籤")
    print("="*70)
    
    # 找出每個分群的「重心 (Centroid)」中，權重最高的 5 個字
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    topic_names = {}
    
    for i in range(n_clusters):
        top_words = [feature_names[ind] for ind in order_centroids[i, :5]]
        # 用前 3 個字當作圖表的簡短標籤
        short_name = f"群 {i+1} [{top_words[0]}/{top_words[1]}]"
        topic_names[i] = short_name
        print(f"🔹 【群集 {i+1}】 核心關鍵字 ➔ " + "、".join(top_words))
    print("="*70)

    # 5. 計算時間趨勢 (每季各主題的佔比)
    # 初始化資料結構：quarter_counts['2020-Q1'][0] 代表 2020年Q1，屬於群集 0 的文章數
    quarter_counts = defaultdict(lambda: {c: 0 for c in range(n_clusters)})
    quarter_totals = defaultdict(int)

    for idx, c_label in enumerate(cluster_labels):
        q = q_labels[idx]
        quarter_counts[q][c_label] += 1
        quarter_totals[q] += 1

    sorted_quarters = sorted(quarter_counts.keys())
    
    # 準備畫圖資料
    trend_data = {c: [] for c in range(n_clusters)}
    for q in sorted_quarters:
        total = quarter_totals[q]
        for c in range(n_clusters):
            # 計算該主題在該季度的「市佔率」(%)
            percentage = (quarter_counts[q][c] / total) * 100 if total > 0 else 0
            trend_data[c].append(percentage)

    # 6. 繪製精美的折線圖
    print("\n 正在為您繪製主題趨勢折線圖...")
    plt.figure(figsize=(15, 8))
    
    # 為了讓 5 條線好分辨，我們加上不同標記
    markers = ['o', 's', '^', 'D', 'v'] 
    
    for c in range(n_clusters):
        plt.plot(sorted_quarters, trend_data[c], marker=markers[c], linewidth=2.5, label=topic_names[c])

    plt.title('2019-2025 台灣查核報告：五大潛在主題趨勢消長圖 (K-Means)', fontsize=22, pad=20)
    plt.xlabel('季度', fontsize=16)
    plt.ylabel('該主題佔當季總查核量比例 (%)', fontsize=16)
    
    # 旋轉 X 軸標籤
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=14)
    
    # 將圖例放在圖表外面右側，以免擋住折線
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14) 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_topic_clustering()