import json
import nltk
from collections import defaultdict

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def main():
    # 1. 讀取文件集 
    file_path = 'ReutersCorn-train.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"找不到檔案 {file_path}，請確認檔案路徑是否正確。")
        return

    # 2. 建立資料結構儲存各詞彙的 df 與 tf
    # 結構設計: term_stats[word][docID] = count (tf)
    term_stats = defaultdict(lambda: defaultdict(int))
    
    # 3. 處理每一篇文件
    for doc in documents:
        doc_id = doc.get('docID', '')
        text = doc.get('text', '')
        
        if not text:
            continue
            
        # 使用 NLTK 的 word_tokenize 將文章內容切詞 (tokenization)
        tokens = nltk.word_tokenize(text)
        
        # 將切出的詞彙全數轉成小寫，並統計在該文件中的出現次數 (tf)
        for token in tokens:
            token_lower = token.lower()
            term_stats[token_lower][doc_id] += 1

    # 4. 定義測驗一回答的詞彙清單
    df_only_terms = ['some', 'into', 'here', 'control', 'service', 'what', 'sea']
    tf_df_terms = ['weekend', 'heart', 'turn', 'field', 'design', 'check', 'broken']
    
    # 5. 輸出測驗一的第一部分 (各詞的 df 值)
    print("=== 以下各詞的 df 值 ===")
    for word in df_only_terms:
        df = len(term_stats[word]) if word in term_stats else 0
        print(f"{word}, df={df}")
        
    # 輸出測驗一的第二部分 (各詞的 df 值及 tf 列表)
    print("\n=== 以下各詞的 df 值以及出現在各文件中的 tf 值 ===")
    for word in tf_df_terms:
        if word in term_stats:
            doc_freq = term_stats[word]
            df = len(doc_freq)
            
            # 依照 DocID 進行字母順序排序
            sorted_docs = sorted(doc_freq.items(), key=lambda x: x[0])
            
            # 組合 inv_list 格式: (DocID,tf)(DocID,tf)...
            inv_list = "".join([f"({doc_id},{tf})" for doc_id, tf in sorted_docs])
            print(f"{word}, df={df}, inv_list={inv_list}")
        else:
            print(f"{word}, df=0, inv_list=")

    # 6. 輸出範例對照區 (Debug)
    print("\n=== 範例答案檢查 (Reference data for debug) ===")
    test_df_terms = ['this', 'support', 'city']
    test_tf_df_terms = ['ships', 'hotel', 'hotels']
    
    for word in test_df_terms:
        df = len(term_stats[word]) if word in term_stats else 0
        print(f"{word}, df={df}")
        
    for word in test_tf_df_terms:
        if word in term_stats:
            doc_freq = term_stats[word]
            df = len(doc_freq)
            sorted_docs = sorted(doc_freq.items(), key=lambda x: x[0])
            inv_list = "".join([f"({doc_id},{tf})" for doc_id, tf in sorted_docs])
            print(f"{word}, df={df}, inv_list={inv_list}")
        else:
            print(f"{word}, df=0, inv_list=")

if __name__ == "__main__":
    main()