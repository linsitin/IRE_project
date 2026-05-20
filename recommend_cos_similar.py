import pickle
import numpy as np
import platform
import os
import subprocess
from sklearn.metrics.pairwise import cosine_similarity

DB_FILE = "tfc_database_maxdf0.5.pkl"  # 想要測舊版就改成 "tfc_database.pkl"

def open_file_in_system(filepath):
    """最穩定的做法：直接呼叫作業系統打開這個絕對路徑的檔案"""
    try:
        # 確保路徑是絕對路徑
        abs_path = os.path.abspath(filepath)
        if platform.system() == 'Windows':
            os.startfile(abs_path)
        elif platform.system() == 'Darwin':
            subprocess.call(('open', abs_path))
        else:
            subprocess.call(('xdg-open', abs_path))
    except Exception as e:
        print(f" 無法開啟檔案：{e}")

def select_target_article(titles, dates):
    while True:
        print("\n" + "="*50)
        print(" 請選擇「查詢目標文章」的方式：")
        print("1. 輸入文章 ID (測試用)")
        print("2. 輸入發布日期 (例如: 2025-01-08)")
        print("3. 輸入標題關鍵字")
        print("="*50)
        
        choice = input(" 請選擇 (1/2/3): ").strip()

        if choice == '1':
            try:
                idx = int(input(f" 請輸入 ID (0 ~ {len(titles)-1}): "))
                if 0 <= idx < len(titles): return idx
                print(" ID 超出範圍！")
            except ValueError: print(" 錯誤輸入。")

        elif choice == '2':
            q_date = input(" 請輸入日期 (YYYY-MM-DD): ").strip()
            matches = [i for i, d in enumerate(dates) if q_date in d]
            if not matches:
                print(" 找不到該日期的文章！")
                continue
            print(f"\n 找到 {len(matches)} 篇文章：")
            for i in matches: print(f"  [ID: {i:4d}] {titles[i][:40]}...")
            try:
                idx = int(input(" 選擇 ID: "))
                if idx in matches: return idx
            except ValueError: pass

        elif choice == '3':
            q_word = input(" 輸入關鍵字: ").strip()
            matches = [i for i, t in enumerate(titles) if q_word in t]
            if not matches:
                print(" 找不到包含該關鍵字的文章！")
                continue
            print(f"\n 找到 {len(matches)} 篇文章：")
            for i in matches: print(f"  [ID: {i:4d}] [{dates[i]}] {titles[i][:40]}...")
            try:
                idx = int(input(" 選擇 ID: "))
                if idx in matches: return idx
            except ValueError: pass

def run_fast_recommendation():
    print(f"\n---  [功能二] 相似文章推薦系統 (當前使用資料庫: {DB_FILE}) ---")
    
    try:
        with open(DB_FILE, 'rb') as f:
            db_data = pickle.load(f)
    except FileNotFoundError:
        print(f" 找不到 {DB_FILE}！請先執行 0_build_database.py")
        return

    titles = db_data['titles']
    dates = db_data['dates']
    filepaths = db_data['filepaths']
    tfidf_matrix = db_data['tfidf_matrix']

    target_idx = select_target_article(titles, dates)
    target_title = titles[target_idx]
    
    print("\n" + "="*75)
    print(f" 正在閱讀：[{dates[target_idx]}] {target_title[:35]}... ")
    print("="*75)
    
    target_vector = tfidf_matrix[target_idx]
    similarity_scores = cosine_similarity(target_vector, tfidf_matrix)[0]
    ranked_indices = np.argsort(similarity_scores)[::-1]
    
    print(" 【推薦以下相似查核報告】")
    print("-" * 75)
    
    recommend_count = 0
    recommended_indices = [] 
    
    for idx in ranked_indices:
        score = similarity_scores[idx]
        if idx != target_idx and score > 0.1:
            recommend_count += 1
            recommended_indices.append(idx)
            # 乾乾淨淨的印出純文字，不再搞終端機超連結
            print(f" 推薦 {recommend_count} | 相似度：{score:.3f} | [{dates[idx]}] {titles[idx][:35]}...")
        if recommend_count >= 3: break
            
    if recommend_count == 0: print(" 沒有高度相關的報告。")
    print("-" * 75)
    
    # 核心：靠輸入數字，呼叫 Windows 底層打開檔案
    while recommend_count > 0:
        cmd = input(f"\n 輸入編號 (1~{recommend_count}) 自動開啟原文，或輸入 0 結束: ").strip()
        
        if cmd == '0': 
            break
            
        if cmd.isdigit() and 1 <= int(cmd) <= recommend_count:
            selected_idx = recommended_indices[int(cmd)-1]
            target_file_path = filepaths[selected_idx]
            
            print(f" 正在為您彈出檔案...")
            # 直接把路徑丟給作業系統，它會用你預設的記事本打開
            open_file_in_system(target_file_path)
        else: 
            print(" 錯誤的指令，請輸入有效的數字！")

if __name__ == "__main__":
    run_fast_recommendation()