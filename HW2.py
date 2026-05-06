import json
import math
import nltk
from collections import defaultdict

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def compute_cosine_similarity(vec1, vec2):
    """計算兩個向量的餘弦相似度 (Cosine Similarity)"""
    intersection = set(vec1.keys()) & set(vec2.keys())
    
    # 計算內積
    dot_product = sum(vec1[w] * vec2[w] for w in intersection)
    
    # 計算向量長度
    mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
        
    return dot_product / (mag1 * mag2)

def main():
    file_path = 'ReutersCorn-train.json'
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(f"找不到檔案 {file_path}，請確認檔案路徑是否正確。")
        return

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
            
        # 統計 DF
        for unique_token in set(token.lower() for token in tokens):
            df_counts[unique_token] += 1

    N = len(doc_ids)
    doc_tfidf = defaultdict(dict)
    
    for doc_id, tfs in doc_tf.items():
        for word, tf in tfs.items():
            if tf > 0:
                # 使用 Scheme 2 (ltc) 的平滑化公式
                smooth_tf = 1 + math.log10(tf)
                idf = math.log10(N / df_counts[word])
                doc_tfidf[doc_id][word] = smooth_tf * idf

    target_docs = ['RTC_TR0159', 'RTC_TR0197', 'RTC_TR0346', 'RTC_TR0371', 'RTC_TR0781']
    
    
    for target_id in target_docs:
        if target_id not in doc_tfidf:
            print(f"找不到目標文件: {target_id}")
            continue
            
        target_vec = doc_tfidf[target_id]
        similarities = []
        
        for other_id in doc_ids:
            if other_id == target_id:
                continue
                
            other_vec = doc_tfidf[other_id]
            sim = compute_cosine_similarity(target_vec, other_vec)
            similarities.append({'docid': other_id, 'cosine': sim})
            
        # 依照 cosine 分數降冪排序，若分數相同則依 docid 字母順序升冪排序
        similarities.sort(key=lambda x: (-round(x['cosine'], 6), x['docid']))
        
        top_5 = similarities[:5]
        
        # 輸出結果
        top_5_docids = "/".join([item['docid'] for item in top_5])
        print(f"{target_id} 相似前 5 名是：{top_5_docids}")
        # 輸出格式
        print("[")
        for i, item in enumerate(top_5):
            end_char = "}," if i < 4 else "}"
            print(f" {{'docid': '{item['docid']}', 'cosine': {item['cosine']}}}{end_char}")
        print("]\n")

if __name__ == "__main__":
    main()