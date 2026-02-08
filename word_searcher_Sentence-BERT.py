# -*- coding: utf-8 -*-

import os
import json
import pickle
from sentence_transformers import SentenceTransformer, util
import torch
import sys
from itertools import combinations


# system ディレクトリ
SYSTEM_DIR = os.path.dirname(os.path.abspath(__file__))
CROSSWORD_DIR = os.path.dirname(SYSTEM_DIR)

DATASET_PATH = os.path.join(
    CROSSWORD_DIR,
    "word_list",
    "wordnet_data_100+.json"
)


OUTPUT_SCORE_FILE = os.path.join(
    CROSSWORD_DIR,
    "pkl",
    "word_scores.pkl"
)

OUTPUT_SEARCHER_FILE = os.path.join(
    CROSSWORD_DIR,
    "pkl",
    "searcher.pkl"
)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"



# ================
# Sentence-BERT関連
# ================
class SemanticSearcher:
    def __init__(self, dataset_path, model_name):
        print("Sentence-BERTモデル読み込み中...")
        self.model = SentenceTransformer(model_name)
        
        print("データセット読み込み中...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.words = [d["word"] for d in data]
        self.originals = [d["original"] for d in data]
        self.data = data
        
        print("単語をベクトル化中...")
        self.word_embeddings = self.model.encode(self.words, convert_to_tensor=True)
        
        # original → word のマッピング
        self.word_dict = {}
        for item in data:
            if item.get("original"):
                self.word_dict[item["original"]] = item["word"]
        
        print(f"初期化完了: 単語数: {len(self.words)}")
    
    def get_similarity_scores(self, image_words, answer_word):
        """
        修正版: 単独文字 + 2文字ペアの類似度を測定
        
        Args:
            image_words: イメージ語のリスト（例：["食べ物", "お菓子"]）
            answer_word: 答えの単語（例：「イチゴ」）
        """
        print(f"イメージ語: {image_words}")
        print(f"答えの単語: {answer_word}")
        
        # 各イメージ語の埋め込みを計算
        query_embeddings = [self.model.encode(w, convert_to_tensor=True) for w in image_words]
        
        # 答えの文字を抽出（重複削除）
        answer_chars = list(set(answer_word))
        print(f"答えの文字: {answer_chars}")
        
        # 2文字ペアを生成
        char_pairs = list(combinations(answer_chars, 2))
        print(f"2文字ペア: {[''.join(pair) for pair in char_pairs]}")
        
        # 各単語についてスコアを計算
        print("単語の類似度を計算中...")
        score_dict = {}
        char_pair_dict = {}  # 各文字/ペア → 単語リスト
        
        for i, orig in enumerate(self.originals):
            if not orig:
                continue
            
            # 進捗表示（1000件ごと）
            if i % 1000 == 0:
                print(f"  処理中: {i}/{len(self.originals)}")
            
            # 単語全体のベクトルとイメージ語との類似度を計算
            word_emb = self.word_embeddings[i]
            similarities = [util.cos_sim(query_emb, word_emb)[0][0] for query_emb in query_embeddings]
            avg_score = torch.mean(torch.stack(similarities))
            final_score = float(avg_score)
            
            score_dict[orig] = final_score
            
            # この単語がどの文字/ペアを含むか記録
            for char in answer_chars:
                if char in orig:
                    key = f"char_{char}"
                    if key not in char_pair_dict:
                        char_pair_dict[key] = []
                    char_pair_dict[key].append((orig, final_score))
            
            for pair in char_pairs:
                char1, char2 = pair
                if char1 in orig and char2 in orig:
                    key = f"pair_{''.join(pair)}"
                    if key not in char_pair_dict:
                        char_pair_dict[key] = []
                    char_pair_dict[key].append((orig, final_score))
        
        print("類似度計算完了")
        
        # 各文字/ペアごとの上位単語を表示
        print("\n" + "="*60)
        print("単独文字とペアの候補単語（上位10件）")
        print("="*60)
        
        for char in answer_chars:
            key = f"char_{char}"
            if key in char_pair_dict:
                words = sorted(char_pair_dict[key], key=lambda x: x[1], reverse=True)[:10]
                print(f"\n'{char}'を含む単語:")
                for w, s in words:
                    word_meaning = self.word_dict.get(w, "")
                    if word_meaning and word_meaning != w:
                        print(f"  {w} ({word_meaning}): {s:.4f}")
                    else:
                        print(f"  {w}: {s:.4f}")
        
        for pair in char_pairs:
            key = f"pair_{''.join(pair)}"
            if key in char_pair_dict:
                words = sorted(char_pair_dict[key], key=lambda x: x[1], reverse=True)[:10]
                print(f"\n'{''.join(pair)}'を両方含む単語:")
                for w, s in words:
                    word_meaning = self.word_dict.get(w, "")
                    if word_meaning and word_meaning != w:
                        print(f"  {w} ({word_meaning}): {s:.4f}")
                    else:
                        print(f"  {w}: {s:.4f}")
        
        return score_dict, char_pair_dict
    
    def get_words_containing_char(self, char):
        """指定文字を含むoriginal単語のリストを返す"""
        result = []
        for orig in self.originals:
            if orig and char in orig:
                if orig not in result:
                    result.append(orig)
        return result
    
    def get_words_containing_chars(self, chars):
        """複数文字を全て含むoriginal単語のリストを返す"""
        result = []
        for orig in self.originals:
            if orig and all(char in orig for char in chars):
                if orig not in result:
                    result.append(orig)
        return result

# ================
# メイン処理
# ================
def main():
    print("="*60)
    print("単語検索プログラム（2文字ペア対応版）")
    print("="*60)
    
    # データセットの存在確認
    if not os.path.exists(DATASET_PATH):
        print(f"エラー: {DATASET_PATH} が見つかりません。")
        sys.exit(1)
    
    # ユーザー入力
    print("\n以下の情報を入力してください:")
    
    # イメージ語の入力
    image_words_input = input("イメージ語（カンマ区切り）: ")
    image_words = [w.strip() for w in image_words_input.replace("、", ",").split(",")]
    
    # 答えの単語の入力
    answer_word = input("答えの単語（カタカナ）: ").strip()
    
    print(f"\nイメージ語: {', '.join(image_words)}")
    print(f"答えの単語: {answer_word}")
    
    # Sentence-BERT初期化
    searcher = SemanticSearcher(DATASET_PATH, MODEL_NAME)
    
    # 類似度スコアを計算（2文字ペア対応）
    print(f"\n イメージ語との類似度を計算中（2文字ペア対応）...")
    score_dict, char_pair_dict = searcher.get_similarity_scores(image_words, answer_word)
    
    # データを保存
    print(f"\n データを保存中...")
    
    # スコア辞書を保存
    save_data = {
        'score_dict': score_dict,
        'char_pair_dict': char_pair_dict,
        'image_words': image_words,
        'answer_word': answer_word,
        'word_dict': searcher.word_dict
    }
    
    with open(OUTPUT_SCORE_FILE, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"スコアデータを保存しました: {OUTPUT_SCORE_FILE}")
    
    # SemanticSearcherを保存
    with open(OUTPUT_SEARCHER_FILE, 'wb') as f:
        pickle.dump(searcher, f)
    print(f"Searcherオブジェクトを保存しました: {OUTPUT_SEARCHER_FILE}")
    
    print("\n" + "="*60)
    print("単語検索完了")
    print("  - 単独文字を含む単語の類似度を測定")
    print("\n次に crossword_builder.py を実行してください。")
    print("="*60)

if __name__ == "__main__":
    main()