import pickle
import os

# 讀取你的 TF-IDF 資料庫
DB_FILE = "tfc_database_maxdf0.5.pkl"  

def main():
    if not os.path.exists(DB_FILE):
        print(f"找不到核心資料庫 {DB_FILE}，請確認檔案是否存在。")
        return

    print("正在載入資料庫，請稍候...")
    with open(DB_FILE, 'rb') as f:
        db_data = pickle.load(f)

    titles = db_data['titles']
    dates = db_data['dates']
    corpus = db_data['corpus'] # 這裡面存的就是 CKIP 的斷詞結果

    while True:
        print("\n==================================================")
        print("[CKIP 斷詞結果微觀檢查工具]")
        print("1. 輸入標題關鍵字進行搜尋")
        print("0. 離開系統")
        print("==================================================")
        choice = input("請選擇 (1/0): ").strip()

        if choice == '0':
            print("已關閉檢查工具。")
            break
            
        elif choice == '1':
            keyword = input("請輸入要檢查的標題關鍵字: ").strip()
            if not keyword:
                continue

            # 搜尋包含關鍵字的文章索引
            matches = [i for i, t in enumerate(titles) if keyword in t]

            if not matches:
                print("資料庫中找不到包含該關鍵字的文章。")
                continue

            print(f"\n找到 {len(matches)} 篇相符的文章：")
            for i in matches:
                print(f"  [ID: {i:4d}] [{dates[i]}] {titles[i]}")

            try:
                target_id = int(input("\n請輸入想要查看斷詞細節的文章 ID: "))
                if target_id in matches:
                    print("\n" + "-"*70)
                    print(f"文章 ID: {target_id}")
                    print(f"發布日期: {dates[target_id]}")
                    print(f"原始標題: {titles[target_id]}")
                    print("-"*70)
                    print("CKIP 斷詞序列 (詞與詞之間以 / 區隔)：\n")

                    tokens = corpus[target_id]
                    
                    # 確保資料格式正確並印出
                    if isinstance(tokens, list):
                        print(" / ".join(tokens))
                    else:
                        print(tokens)
                    print("-"*70)
                    
                    # 計算這篇文章總共被切成幾個詞
                    print(f"總計詞數：{len(tokens)} 個詞")
                    print("-"*70)
                else:
                    print("輸入的 ID 不在搜尋結果中，請重新輸入。")
            except ValueError:
                print("格式錯誤，請輸入正確的數字 ID。")
        else:
            print("輸入無效，請輸入 1 或 0。")

if __name__ == "__main__":
    main()