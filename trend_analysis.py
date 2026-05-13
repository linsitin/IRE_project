import pickle
import numpy as np
import unicodedata
from collections import Counter, defaultdict

DB_FILE = "tfc_database_maxdf0.33.pkl"  # 想要測舊版就改成 "tfc_database.pkl"

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

def print_trend_for_group(group_name, indices, tfidf_matrix, feature_names):
    """專門用來計算並印出『特定時間區間』趨勢的函數"""
    total_in_group = len(indices)
    if total_in_group == 0: return

    trend_counter = Counter()
    for i in indices:
        row = tfidf_matrix[i].toarray()[0]
        # 依然保持每篇文章抓取 Top 10 核心字
        top_indices = np.argsort(row)[-10:] 
        top_words = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        trend_counter.update(top_words)

    print("\n" + "="*70)
    #  標示這個區間有幾篇文章
    print(f" 【 {group_name} 】 核心話題趨勢 Top 10 (本期共收錄 {total_in_group} 篇文章)")
    print("="*70)
    print(f"{pad_str('排名', 8)} | {pad_str('核心話題關鍵字', 16)} | {pad_str('成為核心的篇數', 16)} | {pad_str('話題強度 (比例)', 16)}")
    print("-" * 70)
    
    # 為了避免畫面太長，單一區間我們顯示 Top 10 即可
    top_trends = trend_counter.most_common(10)
    for rank, (word, count) in enumerate(top_trends, start=1):
        #  這裡的比例是除以「該區間」的文章總數，這樣才精準！
        power = count / total_in_group
        print(f"{pad_str(f'第 {rank} 名', 8)} | {pad_str(f'【 {word} 】', 16)} | {pad_str(count, 16)} | {pad_str(f'{power:.3f}', 16)}")
    print("-" * 70)

def run_fast_trend_analysis():
    print(f"\n---  [功能一] TF-IDF 時間序列趨勢分析 (當前使用資料庫: {DB_FILE}) ---")
    
    try:
        print(f" 正在讀取預載語料庫 {DB_FILE} ...")
        with open(DB_FILE, 'rb') as f:
            db_data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到 {DB_FILE}！請先執行 0_build_database.py")
        return

    tfidf_matrix = db_data['tfidf_matrix']
    vectorizer = db_data['vectorizer']
    dates = db_data['dates']  #  把日期資料讀出來
    feature_names = np.array(vectorizer.get_feature_names_out())

    # 將資料依照「年份」與「月份」進行分組 (Grouping)
    yearly_groups = defaultdict(list)
    monthly_groups = defaultdict(list)

    for i, date_str in enumerate(dates):
        if date_str == "未知日期":
            continue
        
        # date_str 格式為 "YYYY-MM-DD"
        year = date_str[:4]         # 取前 4 個字元 (例如: 2022)
        month = date_str[:7]        # 取前 7 個字元 (例如: 2022-03)
        
        yearly_groups[year].append(i)
        monthly_groups[month].append(i)

    # 互動式選單：讓使用者選擇切片維度
    while True:
        print("\n" + "▼"*50)
        print(" 請選擇您要觀察的「時間趨勢維度」：")
        print("1. 按【年度】分析 (顯示從最早年份至今，各年的獨立排名)")
        print("2. 按【月份】分析 (顯示所有歷史月份的獨立排名)")
        print("0. 退出程式")
        print("▲"*50)
        
        choice = input(" 請選擇 (1/2/0): ").strip()
        
        if choice == '0':
            print(" 感謝使用！")
            break
            
        elif choice == '1':
            # 將年份由小排到大 (例如 2018 -> 2019 -> ...)
            sorted_years = sorted(yearly_groups.keys())
            for year in sorted_years:
                print_trend_for_group(f"{year} 年度", yearly_groups[year], tfidf_matrix, feature_names)
                
        elif choice == '2':
            # 將月份由小排到大 (例如 2022-01 -> 2022-02 -> ...)
            sorted_months = sorted(monthly_groups.keys())
            for month in sorted_months:
                print_trend_for_group(f"{month} 月份", monthly_groups[month], tfidf_matrix, feature_names)
                
        else:
            print("❌ 輸入錯誤，請輸入 1, 2 或 0。")

if __name__ == "__main__":
    run_fast_trend_analysis()