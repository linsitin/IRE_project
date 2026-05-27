import pickle
import numpy as np
import platform
import os
import subprocess
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity

# 三個資料庫路徑 (請確認檔名與你實際產出的一致)
TFIDF_DB = "tfc_database_maxdf0.33.pkl" # 若你的檔名是 0.5 請自行修改
FT_DB = "fasttext_model_data.pkl"
SBERT_DB = "sbert_model_data.pkl"

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
    print(f"{pad_str('No.', 5)} | {pad_str('相似度', 8)} | {pad_str('發布日期', 12)} | {pad_str('標題', 40, 'left')}")
    print("-" * 75)
    
    recommended_list = []
    for i in range(5):
        idx = ranked_indices[i]
        recommended_list.append(idx)
        print(f"{pad_str(i+1, 5)} | {pad_str(f'{scores[idx]:.4f}', 8)} | {pad_str(dates[idx], 12)} | {pad_str(titles[idx][:40], 40, 'left')}")
    return recommended_list

def main():
    if not os.path.exists(TFIDF_DB):
        print(f" 找不到核心資料庫 {TFIDF_DB}，請先執行 build_database.py")
        return

    # 1. 讀取資料庫 (加入防呆機制，即使沒訓練新模型也能跑舊的)
    with open(TFIDF_DB, "rb") as f: tfidf_data = pickle.load(f)
    
    ft_data = None
    if os.path.exists(FT_DB):
        with open(FT_DB, "rb") as f: ft_data = pickle.load(f)
        
    sbert_data = None
    if os.path.exists(SBERT_DB):
        with open(SBERT_DB, "rb") as f: sbert_data = pickle.load(f)
    
    titles = tfidf_data['titles']
    dates = tfidf_data['dates']
    filepaths = tfidf_data['filepaths']
    
    target_idx = select_target_article(titles, dates)
    if target_idx is None: return

    print(f"\n 查詢目標: [{dates[target_idx]}] {titles[target_idx]}")
    print("\n請選擇演算法：")
    print("1. TF-IDF ")
    print("2. FastText ")
    print("3. FastText ")
    print("4. SentenceTransformer ")
    print("5. 上述所有方法")
    mode = input("> ").strip()

    # 2. 運算推薦
    recommend_map = {}
    
    # [1] TF-IDF
    if mode in ['1', '5']:
        scores = cosine_similarity(tfidf_data['tfidf_matrix'][target_idx], tfidf_data['tfidf_matrix'])[0]
        ranked = np.argsort(scores)[::-1]
        recommend_map['1. TF-IDF (全文)'] = (scores, [i for i in ranked if i != target_idx])
        
    # [2] FastText (全文)
    if mode in ['2', '5'] and ft_data:
        scores = cosine_similarity(ft_data['doc_vectors'][target_idx].reshape(1, -1), ft_data['doc_vectors'])[0]
        ranked = np.argsort(scores)[::-1]
        recommend_map['2. FastText (全文)'] = (scores, [i for i in ranked if i != target_idx])

    # [3] FastText (標題)
    if mode in ['3', '5'] and ft_data and 'title_vectors' in ft_data:
        scores = cosine_similarity(ft_data['title_vectors'][target_idx].reshape(1, -1), ft_data['title_vectors'])[0]
        ranked = np.argsort(scores)[::-1]
        recommend_map['3. FastText (標題)'] = (scores, [i for i in ranked if i != target_idx])

    # [4] SBERT (全文)
    if mode in ['4', '5'] and sbert_data:
        scores = cosine_similarity(sbert_data['sbert_doc_vectors'][target_idx].reshape(1, -1), sbert_data['sbert_doc_vectors'])[0]
        ranked = np.argsort(scores)[::-1]
        recommend_map['4. SBERT (全文)'] = (scores, [i for i in ranked if i != target_idx])

    if not recommend_map:
        print(" 無法執行，請確認對應的資料庫檔案 (.pkl) 是否已建立！")
        return

    # 3. 輸出結果
    all_recommendations = {}
    for method, (scores, sorted_indices) in recommend_map.items():
        all_recommendations[method] = display_results(method, scores, sorted_indices, titles, dates, filepaths)

    # 4. 智慧開啟檔案邏輯 (支援動態對應)
    choice = input("\n輸入『方法編號_推薦編號』(例如 1_1 表示 TF-IDF 的第 1 篇，3_2 表示 FastText標題 第 2 篇) 或 0 離開: ").strip()
    if choice != '0' and '_' in choice:
        try:
            m_idx, r_idx = map(int, choice.split('_'))
            
            # 將輸入的方法數字對應到字典的 Key
            method_lookup = {
                1: '1. TF-IDF (全文)',
                2: '2. FastText (全文)',
                3: '3. FastText (標題)',
                4: '4. SBERT (全文)'
            }
            
            if m_idx in method_lookup and method_lookup[m_idx] in all_recommendations:
                method_name = method_lookup[m_idx]
                target_idx = all_recommendations[method_name][r_idx-1]
                open_file_in_system(filepaths[target_idx])
            else:
                print(" 找不到對應的方法結果，請確認該演算法有確實跑出結果。")
        except Exception as e:
            print(f" 指令錯誤，請輸入正確格式 (例如 1_1)！詳細錯誤: {e}")

if __name__ == "__main__":
    main()