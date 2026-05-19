import pickle
import numpy as np
import unicodedata
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

# ==========================================
# 🎨 設定 Matplotlib 支援顯示繁體中文
# ==========================================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'PingFang HK', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

DB_FILE = "tfc_database_maxdf0.33.pkl"  # 請確認你的資料庫檔名

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

def get_top_keywords(indices, tfidf_matrix, feature_names, top_n=5):
    """小幫手函數：專門用來計算某群文章中的 Top N 關鍵字"""
    if not indices: return []
    trend_counter = Counter()
    for i in indices:
        row = tfidf_matrix[i].toarray()[0]
        top_indices = np.argsort(row)[-10:] 
        top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        trend_counter.update(top_words)
    return [word for word, count in trend_counter.most_common(top_n)]

def print_trend_for_group(group_name, indices, tfidf_matrix, feature_names):
    """純文字版的趨勢印出"""
    total_in_group = len(indices)
    if total_in_group == 0: return

    trend_counter = Counter()
    for i in indices:
        row = tfidf_matrix[i].toarray()[0]
        top_indices = np.argsort(row)[-10:] 
        top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        trend_counter.update(top_words)

    print("\n" + "="*70)
    print(f"📅 【 {group_name} 】 核心話題趨勢 Top 20 (本期共收錄 {total_in_group} 篇文章)")
    print("="*70)
    print(f"{pad_str('排名', 8)} | {pad_str('核心話題關鍵字', 16)} | {pad_str('成為核心的篇數', 16)} | {pad_str('話題強度 (比例)', 16)}")
    print("-" * 70)
    
    top_trends = trend_counter.most_common(20)
    for rank, (word, count) in enumerate(top_trends, start=1):
        power = count / total_in_group
        print(f"{pad_str(f'第 {rank} 名', 8)} | {pad_str(f'【 {word} 】', 16)} | {pad_str(count, 16)} | {pad_str(f'{power:.3f}', 16)}")
    print("-" * 70)

# ==========================================
# 📈 新增功能：畫圖函數
# ==========================================
def plot_yearly_trend(yearly_groups, tfidf_matrix, feature_names):
    """畫出全年度 Top 5 關鍵字的歷年走勢"""
    print("\n📊 正在計算全區間的 Top 5 關鍵字並繪製折線圖...")
    
    # 1. 抓出所有有資料的年份 (例如 2018 ~ 2025)
    years = sorted([y for y in yearly_groups.keys() if y >= '2018'])
    if not years:
        print("沒有足夠的年份資料！")
        return

    # 2. 抓出整個資料庫的 Top 5 關鍵字作為追蹤目標
    all_indices = [idx for y in years for idx in yearly_groups[y]]
    target_words = get_top_keywords(all_indices, tfidf_matrix, feature_names, top_n=5)
    
    # 3. 計算這 5 個字在每年的強度
    word_trends = {word: [] for word in target_words}
    for year in years:
        indices = yearly_groups[year]
        total = len(indices)
        if total == 0:
            for w in target_words: word_trends[w].append(0)
            continue
            
        trend_counter = Counter()
        for i in indices:
            row = tfidf_matrix[i].toarray()[0]
            # 嚴格濾網：只取前 10 名
            top_indices = np.argsort(row)[-10:]
            top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            trend_counter.update(top_words)
            
        for w in target_words:
            # 轉換為百分比 (%)
            word_trends[w].append((trend_counter[w] / total) * 100)

    # 4. 繪製折線圖
    plt.figure(figsize=(10, 6))
    for word in target_words:
        plt.plot(years, word_trends[word], marker='o', linewidth=2, label=word)
        
    plt.title('2018-2025 查核報告核心話題趨勢 (Top 5 關鍵字)', fontsize=16, pad=15)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('話題強度涵蓋率 (%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_monthly_trend(target_year, monthly_groups, yearly_groups, tfidf_matrix, feature_names):
    """畫出特定年份中，Top 10 關鍵字的各月走勢"""
    if target_year not in yearly_groups:
        print(f"❌ 找不到 {target_year} 年的資料！")
        return
        
    print(f"\n📊 正在計算 {target_year} 年的 Top 10 關鍵字並繪製月份折線圖...")
    
    # 1. 抓出該年份「專屬」的 Top 10 關鍵字
    target_words = get_top_keywords(yearly_groups[target_year], tfidf_matrix, feature_names, top_n=10)
    
    # 2. 抓出該年所有的月份 (例如 2022-01 ~ 2022-12)
    months = sorted([m for m in monthly_groups.keys() if m.startswith(target_year)])
    word_trends = {word: [] for word in target_words}
    
    # 3. 計算這 10 個字在每個月的強度
    for month in months:
        indices = monthly_groups[month]
        total = len(indices)
        if total == 0:
            for w in target_words: word_trends[w].append(0)
            continue
            
        trend_counter = Counter()
        for i in indices:
            row = tfidf_matrix[i].toarray()[0]
            # 嚴格濾網：只取前 10 名
            top_indices = np.argsort(row)[-10:]
            top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
            trend_counter.update(top_words)
            
        for w in target_words:
            word_trends[w].append((trend_counter[w] / total) * 100)

    # 4. 繪製折線圖
    plt.figure(figsize=(12, 6))
    for word in target_words:
        plt.plot(months, word_trends[word], marker='s', linewidth=2, label=word)
        
    plt.title(f'{target_year} 年度各月份核心話題趨勢 (Top 10 關鍵字)', fontsize=16, pad=15)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('話題強度涵蓋率 (%)', fontsize=12)
    plt.xticks(rotation=45) # 標籤稍微旋轉避免擠在一起
    # 將圖例放在圖表外面右側，以免擋住折線
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11) 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# ==========================================

def run_fast_trend_analysis():
    print(f"\n--- 🚀 [功能一] TF-IDF 時間序列趨勢分析 (當前使用資料庫: {DB_FILE}) ---")
    
    try:
        print(f"📂 正在讀取預載語料庫 {DB_FILE} ...")
        with open(DB_FILE, 'rb') as f:
            db_data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到 {DB_FILE}！請先執行 0_build_database.py")
        return

    tfidf_matrix = db_data['tfidf_matrix']
    vectorizer = db_data['vectorizer']
    dates = db_data['dates']  
    feature_names = np.array(vectorizer.get_feature_names_out())

    yearly_groups = defaultdict(list)
    monthly_groups = defaultdict(list)

    for i, date_str in enumerate(dates):
        if date_str == "未知日期":
            continue
        year = date_str[:4]         
        month = date_str[:7]        
        yearly_groups[year].append(i)
        monthly_groups[month].append(i)

    while True:
        print("\n" + "▼"*50)
        print("🔍 請選擇您要觀察的「時間趨勢維度」：")
        print("1. 📝 按【年度】文字報表 (顯示各年獨立排名)")
        print("2. 📝 按【月份】文字報表 (顯示歷史月份獨立排名)")
        print("3. 📈 繪製【年度趨勢折線圖】(2018-2025 整體 Top 5 趨勢)")
        print("4. 📈 繪製【單一年度的月份走勢圖】(該年份 Top 10 趨勢)")
        print("0. 退出程式")
        print("▲"*50)
        
        choice = input("👉 請選擇 (0~4): ").strip()
        
        if choice == '0':
            print("👋 感謝使用！")
            break
            
        elif choice == '1':
            sorted_years = sorted(yearly_groups.keys())
            for year in sorted_years:
                print_trend_for_group(f"{year} 年度", yearly_groups[year], tfidf_matrix, feature_names)
                
        elif choice == '2':
            sorted_months = sorted(monthly_groups.keys())
            for month in sorted_months:
                print_trend_for_group(f"{month} 月份", monthly_groups[month], tfidf_matrix, feature_names)
                
        elif choice == '3':
            plot_yearly_trend(yearly_groups, tfidf_matrix, feature_names)
            
        elif choice == '4':
            target_year = input("👉 請輸入要觀察的年份 (例如 2021): ").strip()
            plot_monthly_trend(target_year, monthly_groups, yearly_groups, tfidf_matrix, feature_names)
            
        else:
            print("❌ 輸入錯誤，請輸入 0, 1, 2, 3 或 4。")

if __name__ == "__main__":
    run_fast_trend_analysis()