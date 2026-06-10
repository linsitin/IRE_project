import pickle
import numpy as np
import platform
import os
import subprocess
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity

TFIDF_DB = "tfc_database_maxdf0.33.pkl" 
FT_DB = "fasttext_model_data.pkl"
SBERT_DB = "sbert_model_data.pkl"

def pad_str(s, total_width, align='center'):
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
    except Exception as e: 
        print(f"無法開啟: {e}")

def select_target_article(titles, dates):
    while True:
        print("\n" + "="*50)
        print("[選擇查詢目標]")
        print("1. 輸入發布日期 (YYYY-MM-DD)")
        print("2. 輸入標題關鍵字")
        print("0. 離開系統")
        print("="*50)
        choice = input("請選擇 (1/2/0): ").strip()
        if choice == '0': return None
        
        matches = []
        if choice == '1':
            q = input("請輸入日期: ").strip()
            matches = [i for i, d in enumerate(dates) if q in d]
        elif choice == '2':
            q = input("輸入關鍵字: ").strip()
            matches = [i for i, t in enumerate(titles) if q in t]
            
        if not matches:
            print("找不到符合條件的文章！")
            continue
            
        print(f"\n找到 {len(matches)} 篇文章：")
        for i in matches: print(f"  [ID: {i:4d}] {titles[i][:40]}...")
        try:
            idx = int(input("請輸入選擇的 ID: "))
            if idx in matches: return idx
        except: 
            print("輸入錯誤。")

def display_results(method_name, scores, ranked_indices, titles, dates, filepaths, k_limit, abs_threshold=0.0):
    """
    顯示結果：完全受控於 k_limit，只用 abs_threshold 作為最後防線
    """
    print(f"\n>>> 【{method_name} 推薦結果】( 以TF-IDF為基準，推薦 Top {k_limit})")
    print(f"{pad_str('No.', 5)} | {pad_str('相似度', 8)} | {pad_str('發布日期', 12)} | {pad_str('標題', 40, 'left')}")
    print("-" * 75)
    
    recommended_list = []
    display_count = 0
    
    if k_limit == 0:
        print("  (無推薦結果：TF-IDF 基準判定本篇無相關延伸閱讀)")
        return recommended_list
        
    for i in range(k_limit):
        if i >= len(ranked_indices): break 
        
        idx = ranked_indices[i]
        score = scores[idx]
        
        # 這是唯一的防線：如果強制要你出 3 篇，但你第 3 篇分數已經低於演算法的絕對底線，就攔截
        if score < abs_threshold:
            break
            
        recommended_list.append(idx)
        display_count += 1
        print(f"{pad_str(display_count, 5)} | {pad_str(f'{score:.4f}', 8)} | {pad_str(dates[idx], 12)} | {pad_str(titles[idx][:40], 40, 'left')}")
        
    if display_count < k_limit and display_count > 0:
        print(f"  (系統提示：原定推薦 {k_limit} 篇，但後續文章已低於該演算法的極限底線而遭攔截)")
        
    return recommended_list

def main():
    if not os.path.exists(TFIDF_DB):
        print(f"找不到核心資料庫 {TFIDF_DB}，請先執行 build_database.py")
        return

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
    tfidf_matrix = tfidf_data['tfidf_matrix']
    
    target_idx = select_target_article(titles, dates)
    if target_idx is None: return

    print(f"\n查詢目標: [{dates[target_idx]}] {titles[target_idx]}")
    
    # 背景計算 TF-IDF 決定全域推薦數量 (Global K)
    print("\n系統正在使用 TF-IDF 分析基準文章的關聯邊界...")
    
    tfidf_scores = cosine_similarity(tfidf_matrix[target_idx], tfidf_matrix)[0]
    tfidf_ranked = np.argsort(tfidf_scores)[::-1]
    tfidf_ranked = [i for i in tfidf_ranked if i != target_idx]
    
    global_k = 0
    
    if len(tfidf_ranked) > 0:
        base_score = tfidf_scores[tfidf_ranked[0]]
        if base_score > 0.95 and len(tfidf_ranked) > 1:
            base_score = tfidf_scores[tfidf_ranked[1]]
            
        # [動態比例門檻] 絕對及格線 0.22，且至少保留第一名 70% 的分數
        dynamic_cutoff = max(0.22, base_score * 0.70)
        
        print(f"-> TF-IDF 第一名基準分: {base_score:.3f} | 最終切斷線: {dynamic_cutoff:.3f}")
        
        for i in range(min(5, len(tfidf_ranked))):
            if tfidf_scores[tfidf_ranked[i]] >= dynamic_cutoff:
                global_k += 1
            else:
                break

    print(f"基準分析完成：判定所有演算法本次最多推薦 {global_k} 篇。")
    # =========================================================

    print("\n請選擇要檢視的演算法：")
    print("1. TF-IDF (內文精準比對)")                  
    print("2. FastText (內文語意比對)")                
    print("3. FastText (標題語意比對 - 無標籤)")         
    print("4. SentenceTransformer (SBERT 深度語意比對)")
    print("5. 上述所有方法")
    mode = input("> ").strip()

    recommend_map = {}
    
    if mode in ['1', '5']:
        recommend_map['1. TF-IDF (內文)'] = (tfidf_scores, tfidf_ranked) 
        
    if mode in ['2', '5'] and ft_data:
        scores = cosine_similarity(ft_data['doc_vectors'][target_idx].reshape(1, -1), ft_data['doc_vectors'])[0]
        ranked = np.argsort(scores)[::-1]
        recommend_map['2. FastText (內文)'] = (scores, [i for i in ranked if i != target_idx]) 

    if mode in ['3', '5'] and ft_data and 'title_vectors' in ft_data:
        scores = cosine_similarity(ft_data['title_vectors'][target_idx].reshape(1, -1), ft_data['title_vectors'])[0]
        ranked = np.argsort(scores)[::-1]
        recommend_map['3. FastText (標題)'] = (scores, [i for i in ranked if i != target_idx])

    if mode in ['4', '5'] and sbert_data:
        scores = cosine_similarity(sbert_data['sbert_doc_vectors'][target_idx].reshape(1, -1), sbert_data['sbert_doc_vectors'])[0]
        ranked = np.argsort(scores)[::-1]
        recommend_map['4. SBERT (內文)'] = (scores, [i for i in ranked if i != target_idx]) 

    if not recommend_map:
        print("無法執行，請確認對應的資料庫檔案 (.pkl) 是否已建立！")
        return

    # 3. 輸出結果 (所有演算法強制綁定 global_k)
    all_recommendations = {}
    for method, (scores, sorted_indices) in recommend_map.items():
        
        # 這裡的門檻非常低，只是為了防止演算法在被強制要求輸出 N 篇時，硬湊出完全無關的文章
        if 'TF-IDF' in method:
            abs_min = 0.0 # 已經由 global_k 控制，不需再設底線
        elif 'FastText' in method:
            abs_min = 0.60 
        elif 'SBERT' in method:
            abs_min = 0.70 
        else:
            abs_min = 0.0
            
        all_recommendations[method] = display_results(
            method, scores, sorted_indices, titles, dates, filepaths, 
            k_limit=global_k, abs_threshold=abs_min
        )

    # 4. 智慧開啟檔案邏輯
    while True:
        choice = input("\n輸入『方法編號_推薦編號』(例如 1_1) 或 0 離開: ").strip()
        
        if choice == '0':
            break
            
        if '_' in choice:
            try:
                m_idx, r_idx = map(int, choice.split('_'))
                
                method_lookup = {
                    1: '1. TF-IDF (內文)',
                    2: '2. FastText (內文)',
                    3: '3. FastText (標題)',
                    4: '4. SBERT (內文)'
                }
                
                if m_idx in method_lookup and method_lookup[m_idx] in all_recommendations:
                    method_name = method_lookup[m_idx]
                    
                    if 1 <= r_idx <= len(all_recommendations[method_name]):
                        target_idx = all_recommendations[method_name][r_idx-1]
                        print(f"正在為您開啟: {titles[target_idx][:50]}...")
                        open_file_in_system(filepaths[target_idx])
                    else:
                        print(f"找不到該篇推薦！請確認該演算法有輸出第 {r_idx} 篇文章。")
                else:
                    print("找不到對應的方法。")
                    
            except ValueError:
                print("指令格式錯誤！請確認輸入格式如 1_1 或 4_2。")
            except Exception as e:
                print(f"發生錯誤: {e}")
        else:
            print("輸入無效，請輸入如 2_3 或 0 離開。")

if __name__ == "__main__":
    main()