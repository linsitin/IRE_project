import pickle
import numpy as np
import unicodedata
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# 設定 Matplotlib 支援顯示繁體中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

#  加入這一行，直接把基礎字體大小調到 14 (原本預設大約是 10)
plt.rcParams['font.size'] = 14

DB_FILE = "tfc_database_maxdf0.5.pkl"  # 讀取你的 TF-IDF 資料庫

def get_display_width(s):
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in str(s))

def pad_str(s, total_width, align='center'):
    s = str(s)
    current_width = get_display_width(s)
    padding = total_width - current_width
    if padding <= 0: return s
    if align == 'center':
        left_pad = padding // 2
        right_pad = padding - left_pad
        return ' ' * left_pad + s + ' ' * right_pad
    return s + ' ' * padding

def calculate_top_keywords_for_indices(indices, tfidf_matrix, feature_names, top_n=10):
    """計算指定文章群中，TF-IDF 權重最高的 Top N 關鍵字 (僅回傳字詞清單)"""
    if not indices: return []
    trend_counter = Counter()
    for i in indices:
        row = tfidf_matrix[i].toarray()[0]
        top_indices = np.argsort(row)[-10:]
        top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        trend_counter.update(top_words)
    return [word for word, count in trend_counter.most_common(top_n)]

def print_detailed_table(title, group_dict, tfidf_matrix, feature_names):
    """輸出詳細的數據對齊表格 (結合你原本的排版風格)"""
    print("\n" + "="*70)
    print(f"  【{title}】核心話題趨勢 Top 10 詳細報表")
    print("="*70)
    
    for period in sorted(group_dict.keys()):
        indices = group_dict[period]
        total_in_group = len(indices)
        if total_in_group == 0: continue
        
        # 針對該區間計算詳細數據
        trend_counter = Counter()
        for i in indices:
            row = tfidf_matrix[i].toarray()[0]
            top_indices = np.argsort(row)[-10:]
            top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            trend_counter.update(top_words)
            
        print(f"\n▶ 區間：【 {period} 】 (本期共收錄 {total_in_group} 篇文章)")
        print(f"{pad_str('排名', 8)} | {pad_str('核心話題關鍵字', 16)} | {pad_str('篇數', 8)} | {pad_str('話題強度', 12)}")
        print("-" * 55)
        
        top_trends = trend_counter.most_common(10)
        for rank, (word, count) in enumerate(top_trends, start=1):
            power = count / total_in_group
            print(f"{pad_str(f'第 {rank} 名', 8)} | {pad_str(f'【 {word} 】', 16)} | {pad_str(count, 8)} | {pad_str(f'{power:.3f}', 12)}")
        print("-" * 55)

def plot_macro_trend(time_label, group_dict, target_words, tfidf_matrix, feature_names):
    """通用的折線圖繪製函數 (追蹤 8 年總體的 Top 10)"""
    print(f"\n  正在為【{time_label}】維度繪製趨勢折線圖...")
    
    periods = sorted(group_dict.keys())
    word_trends = {word: [] for word in target_words}
    
    for period in periods:
        indices = group_dict[period]
        total = len(indices)
        
        if total == 0:
            for w in target_words: word_trends[w].append(0)
            continue
            
        trend_counter = Counter()
        for i in indices:
            row = tfidf_matrix[i].toarray()[0]
            top_indices = np.argsort(row)[-10:]
            top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            trend_counter.update(top_words)
            
        for w in target_words:
            word_trends[w].append((trend_counter[w] / total) * 100)

    plt.figure(figsize=(14, 7))
    for word in target_words:
        plt.plot(periods, word_trends[word], marker='o', linewidth=2, label=word)
        
    plt.title(f'2019-2025 的核心詞彙 Top 10 趨勢演變' , fontsize=16, pad=15)
    plt.xlabel(time_label, fontsize=12)
    plt.ylabel('成為核心詞彙的佔比 (%)', fontsize=12)
    
    if len(periods) > 12:
        plt.xticks(rotation=45, fontsize=9)
        
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11) 
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def run_macro_trend_analysis():
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

    global_indices = [] 
    yearly_groups = defaultdict(list)
    half_yearly_groups = defaultdict(list)
    quarterly_groups = defaultdict(list)

    for i, date_str in enumerate(dates):
        if date_str == "未知日期" or date_str < '2018': 
            continue
            
        year = date_str[:4]
        # 把 2018 和 2026 (含以後) 的年份擋掉
        if year == '2018' or year >= '2026': 
            continue         
        month_str = date_str[5:7]   
        
        if not month_str.isdigit(): continue
        month = int(month_str)
        
        h_label = f"{year}-H1" if month <= 6 else f"{year}-H2"
        q_label = f"{year}-Q{((month - 1) // 3) + 1}"
        
        global_indices.append(i)
        yearly_groups[year].append(i)
        half_yearly_groups[h_label].append(i)
        quarterly_groups[q_label].append(i)

    print(" 正在計算 2018-2025 八年總體核心基準線...")
    global_top_words = calculate_top_keywords_for_indices(global_indices, tfidf_matrix, feature_names, top_n=20)

    print("\n" + "★"*60)
    print("  2018-2025 八年總體核心話題 Top 20 基準線")
    print(" " + "、".join([f"【{w}】" for w in global_top_words[:20]]))
    print("★"*60)

    while True:
        print("\n" + "▼"*55)
        print(" 請選擇微觀時間分析維度 (包含詳細數據表 + 趨勢折線圖)：")
        print("1. 查看【 1 年 】核心演變")
        print("2. 查看【 半年 (H1/H2) 】核心演變")
        print("3. 查看【 季度 (Q1~Q4) 】核心演變")
        print("0. 退出程式")
        print("▲"*55)
        
        choice = input(" 請選擇 (0~3): ").strip()
        
        if choice == '0':
            print(" 系統已優雅退出！")
            break
        elif choice == '1':
            print_detailed_table("1年維度", yearly_groups, tfidf_matrix, feature_names)
            plot_macro_trend("年度", yearly_groups, global_top_words[:10], tfidf_matrix, feature_names)
        elif choice == '2':
            print_detailed_table("半年維度", half_yearly_groups, tfidf_matrix, feature_names)
            plot_macro_trend("半年", half_yearly_groups, global_top_words[:10], tfidf_matrix, feature_names)
        elif choice == '3':
            print_detailed_table("季度維度", quarterly_groups, tfidf_matrix, feature_names)
            plot_macro_trend("季度", quarterly_groups, global_top_words[:10], tfidf_matrix, feature_names)
        else:
            print(" 輸入錯誤，請輸入 0, 1, 2 或 3。")

if __name__ == "__main__":
    run_macro_trend_analysis()