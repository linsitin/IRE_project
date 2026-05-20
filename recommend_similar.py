import pickle
import numpy as np
import platform
import os
import subprocess
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity

# 兩個資料庫路徑
TFIDF_DB = "tfc_database_maxdf0.5.pkl"
FT_DB = "fasttext_model_data.pkl"

def pad_str(s, total_width, align='center'):
    """文字對齊工具，讓輸出像 Excel 表格"""
    s = str(s)
    width = sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)
    padding = total_width - width
    if padding <= 0: return s
    if align == 'center': return ' ' * (padding // 2) + s + ' ' * (padding - padding // 2)
    return s + ' ' * padding

def open_file_in_system(filepath):
    try:
        abs_path = os.path.abspath(filepath)
        if platform.system() == 'Windows': os.startfile(abs_path)
        elif platform.system() == 'Darwin': subprocess.call(('open', abs_path))
        else: subprocess.call(('xdg-open', abs_path))
    except Exception as e: print(f" 無法開啟: {e}")

def select_target_article(titles, dates):
    while True:
        print("\n" + "="*50)
        print(" [選擇查詢目標]")
        print("1. 輸入發布日期 (YYYY-MM-DD)")
        print("2. 輸入標題關鍵字")
        print("0. 離開系統")
        print("="*50)
        choice = input(" 請選擇 (1/2/0): ").strip()
        if choice == '0': return None
        
        matches = []
        if choice == '1':
            q = input(" 請輸入日期: ").strip()
            matches = [i for i, d in enumerate(dates) if q in d]
        elif choice == '2':
            q = input(" 輸入關鍵字: ").strip()
            matches = [i for i, t in enumerate(titles) if q in t]
            
        if not matches:
            print(" 找不到符合條件的文章！")
            continue
            
        print(f"\n 找到 {len(matches)} 篇文章：")
        for i in matches: print(f"  [ID: {i:4d}] {titles[i][:40]}...")
        try:
            idx = int(input(" 請輸入選擇的 ID: "))
            if idx in matches: return idx
        except: print(" 輸入錯誤。")

def display_results(method_name, scores, ranked_indices, titles, dates, filepaths):
    print(f"\n>>> 【{method_name} 推薦結果】")
    # 這裡我增加了一個「日期」欄位 (寬度 12)
    print(f"{pad_str('No.', 5)} | {pad_str('相似度', 8)} | {pad_str('發布日期', 12)} | {pad_str('標題', 40, 'left')}")
    print("-" * 75)
    
    recommended_list = []
    for i in range(5):
        idx = ranked_indices[i]
        recommended_list.append(idx)
        # 這裡也把 dates[idx] 加入輸出
        print(f"{pad_str(i+1, 5)} | {pad_str(f'{scores[idx]:.4f}', 8)} | {pad_str(dates[idx], 12)} | {pad_str(titles[idx][:40], 40, 'left')}")
    return recommended_list

def main():
    if not (os.path.exists(TFIDF_DB) and os.path.exists(FT_DB)):
        print(" 找不到資料庫檔案，請確保 tfc_database_maxdf0.5.pkl 和 fasttext_model_data.pkl 都在資料夾內。")
        return

    with open(TFIDF_DB, "rb") as f: tfidf_data = pickle.load(f)
    with open(FT_DB, "rb") as f: ft_data = pickle.load(f)
    
    titles = tfidf_data['titles']
    dates = tfidf_data['dates']
    filepaths = tfidf_data['filepaths']
    
    target_idx = select_target_article(titles, dates)
    if target_idx is None: return

    print(f"\n查詢目標: [{dates[target_idx]}] {titles[target_idx]}")
    print("\n請選擇模式：1. TF-IDF | 2. FastText | 3. 兩者比較")
    mode = input("> ")

    # 運算
    recommend_map = {}
    if mode in ['1', '3']:
        scores = cosine_similarity(tfidf_data['tfidf_matrix'][target_idx], tfidf_data['tfidf_matrix'])[0]
        # 過濾自己 (score < 1.0)
        ranked = np.argsort(scores)[::-1]
        recommend_map['TF-IDF'] = (scores, [i for i in ranked if i != target_idx])
        
    if mode in ['2', '3']:
        scores = cosine_similarity(ft_data['doc_vectors'][target_idx].reshape(1, -1), ft_data['doc_vectors'])[0]
        ranked = np.argsort(scores)[::-1]
        recommend_map['FastText'] = (scores, [i for i in ranked if i != target_idx])

    # 輸出
    all_recommendations = {}
    for method, (scores, sorted_indices) in recommend_map.items():
        all_recommendations[method] = display_results(method, scores, sorted_indices, titles, dates, filepaths)

    # 開啟檔案邏輯
    choice = input("\n輸入『方法編號_推薦編號』(例如 1_1 表示 TF-IDF 的第 1 篇) 或 0 離開: ")
    if choice != '0':
        try:
            m_idx, r_idx = map(int, choice.split('_'))
            method_name = 'TF-IDF' if m_idx == 1 else 'FastText'
            target_idx = all_recommendations[method_name][r_idx-1]
            open_file_in_system(filepaths[target_idx])
        except: print("指令錯誤，請輸入格式如 1_1")

if __name__ == "__main__":
    main()