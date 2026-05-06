import json
import math
import nltk
from collections import defaultdict

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def compute_cosine_similarity(vec1, vec2):
    """
    計算兩個向量的餘弦相似度。
    由於傳入的 vec1 和 vec2 都已經過 Cosine Normalization (長度為 1 的單位向量)，
    因此兩者的 Cosine 夾角值即為純粹的內積 (Dot Product)。
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    # 直接計算交集詞彙的權重乘積並加總
    return sum(vec1[w] * vec2[w] for w in intersection)

def main():
    file_path = 'ReutersCorn-train.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"找不到檔案 {file_path}，請確認檔案路徑是否正確。")
        return

    # 斷詞與統計
    doc_tf = defaultdict(lambda: defaultdict(int))
    df_counts = defaultdict(int)
    doc_ids = []
    
    for doc in documents:
        doc_id = doc.get('docID', '')
        text = doc.get('text', '')
        
        if not text:
            continue
            
        doc_ids.append(doc_id)
        tokens = nltk.word_tokenize(text)
        
        # 統計 TF
        for token in tokens:
            doc_tf[doc_id][token.lower()] += 1
            
        # 統計 DF (利用 set 確保同一文件中的詞彙只算一次)
        for unique_token in set(token.lower() for token in tokens):
            df_counts[unique_token] += 1

    N = len(doc_ids)
    doc_tfidf = defaultdict(dict)
    
    
    # 計算 ltc 權重 (Log-TF * IDF) 並且進行 Cosine Normalization
    for doc_id, tfs in doc_tf.items():
        # 先計算尚未正規化的 lt 權重
        unnormalized_vec = {}
        for word, tf in tfs.items():
            if tf > 0:
                smooth_tf = 1 + math.log10(tf)
                idf = math.log10(N / df_counts[word])
                unnormalized_vec[word] = smooth_tf * idf
                
        # 進行 Cosine Normalization (c)
        # 計算該文件向量的總長度 (L2 Norm)
        length = math.sqrt(sum(val ** 2 for val in unnormalized_vec.values()))
        
        # 將每個維度除以總長度，轉為單位向量並存入 doc_tfidf
        if length > 0:
            for word, weight in unnormalized_vec.items():
                doc_tfidf[doc_id][word] = weight / length

    # 目標文件清單
    target_docs = ['RTC_TR0159', 'RTC_TR0197', 'RTC_TR0346', 'RTC_TR0371', 'RTC_TR0781']
    
    
    for target_id in target_docs:
        if target_id not in doc_tfidf:
            print(f"找不到目標文件: {target_id}，請確認資料庫。")
            continue
            
        target_vec = doc_tfidf[target_id]
        similarities = []
        
        # 與其他所有文件進行比對
        for other_id in doc_ids:
            if other_id == target_id:
                continue
                
            other_vec = doc_tfidf[other_id]
            sim = compute_cosine_similarity(target_vec, other_vec)
            similarities.append({'docid': other_id, 'cosine': sim})
            
        # 排序：Cosine 分數高到低 (降冪)；同分時依 docid 字母順序 (升冪) 
        similarities.sort(key=lambda x: (-round(x['cosine'], 6), x['docid']))
        
        top_5 = similarities[:5]
        
        # 輸出結果 
        top_5_docids = "/".join([item['docid'] for item in top_5])
        print(f"{target_id} 相似前 5 名是：{top_5_docids}")
        print("[")
        for i, item in enumerate(top_5):
            end_char = "}," if i < 4 else "}"
            print(f" {{'docid': '{item['docid']}', 'cosine': {item['cosine']}}}{end_char}")
        print("]\n")

if __name__ == "__main__":
    main()