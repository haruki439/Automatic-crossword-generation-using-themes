# -*- coding: utf-8 -*-

import os
import json
import pickle
import numpy as np
from gensim.models import KeyedVectors
import sys
from itertools import combinations



# ================
# 設定
# ================
SYSTEM_DIR = os.path.dirname(os.path.abspath(__file__))
CROSSWORD_DIR = os.path.dirname(SYSTEM_DIR)

DATASET_PATH = os.path.join(
    CROSSWORD_DIR,
    "word_list",
    "wordnet_data_100+.json"
)

JAWIKI_MODEL_PATH = os.path.join(
    SYSTEM_DIR,
    "entity_vector",
    "entity_vector.model.txt"
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


# ================
# 日本語Wikipediaベクトルを使った検索クラス
# ================
class SemanticSearcher:
    def __init__(self, dataset_path, model_path):
        print("日本語Wikipediaベクトルモデル読み込み中...")
     
        try:
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)
            print(f"モデル読み込み完了: 語彙数 {len(self.model.key_to_index)}")
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            print("ヒント: モデルファイルをダウンロードして正しいパスを指定してください")
            sys.exit(1)
        
        print("データセット読み込み中...")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.words = [d["word"] for d in data]
        self.originals = [d["original"] for d in data]
        self.data = data
        
        print("単語をベクトル化中...")
        self.word_embeddings = []
        self.valid_indices = []  # ベクトルが取得できた単語のインデックス
        
        for i, word in enumerate(self.words):
            try:
                # モデルに単語が存在するか確認
                if word in self.model:
                    vec = self.model[word]
                    self.word_embeddings.append(vec)
                    self.valid_indices.append(i)
                else:
                
                    pass
            except:
                pass
        
        self.word_embeddings = np.array(self.word_embeddings)
        print(f"ベクトル化完了: {len(self.valid_indices)}/{len(self.words)} 件")
        
        # original → word のマッピング
        self.word_dict = {}
        for item in data:
            if item.get("original"):
                self.word_dict[item["original"]] = item["word"]
        
        print(f"初期化完了: 単語数: {len(self.valid_indices)}")
    
    def get_word_vector(self, word):
        """単語のベクトルを取得"""
        if word in self.model:
            return self.model[word]
        return None
    
    def cosine_similarity(self, vec1, vec2):
        """コサイン類似度を計算"""
        if vec1 is None or vec2 is None:
            return 0.0
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    def get_similarity_scores(self, image_words, answer_word):
     
        print(f"イメージ語: {image_words}")
        print(f"答えの単語: {answer_word}")
        
        # 各イメージ語のベクトルを取得
        query_vectors = []
        for word in image_words:
            vec = self.get_word_vector(word)
            if vec is not None:
                query_vectors.append(vec)
            else:
                print(f"警告: '{word}' のベクトルが見つかりません")
        
        if not query_vectors:
            print("エラー: イメージ語のベクトルが1つも取得できませんでした")
            return {}, {}
        
        # イメージ語の平均ベクトルを計算
        avg_query_vector = np.mean(query_vectors, axis=0)
        
        # 答えの文字を抽出(重複削除)
        answer_chars = list(set(answer_word))
        print(f"答えの文字: {answer_chars}")
        
        # 2文字ペアを生成
        char_pairs = list(combinations(answer_chars, 2))
        print(f"2文字ペア: {[''.join(pair) for pair in char_pairs]}")
        
        # 各単語についてスコアを計算
        print("単語の類似度を計算中...")
        score_dict = {}
        char_pair_dict = {}  # 各文字/ペア → 単語リスト
        
        for idx in self.valid_indices:
            orig = self.originals[idx]
            if not orig:
                continue
            
            # 進捗表示(1000件ごと)
            if idx % 1000 == 0:
                print(f"  処理中: {idx}/{len(self.originals)}")
            
            # 単語のベクトルとイメージ語との類似度を計算
            word_vec = self.word_embeddings[self.valid_indices.index(idx)]
            similarity = self.cosine_similarity(avg_query_vector, word_vec)
            
            score_dict[orig] = float(similarity)
            
            # この単語がどの文字/ペアを含むか記録
            for char in answer_chars:
                if char in orig:
                    key = f"char_{char}"
                    if key not in char_pair_dict:
                        char_pair_dict[key] = []
                    char_pair_dict[key].append((orig, float(similarity)))
            
            for pair in char_pairs:
                char1, char2 = pair
                if char1 in orig and char2 in orig:
                    key = f"pair_{''.join(pair)}"
                    if key not in char_pair_dict:
                        char_pair_dict[key] = []
                    char_pair_dict[key].append((orig, float(similarity)))
        
        print("類似度計算完了")
        
        # 各文字/ペアごとの上位単語を表示
        print("\n" + "="*60)
        print("単独文字とペアの候補単語(上位10件)")
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
    print("単語検索プログラム(日本語Wikipediaベクトル版)")
    print("="*60)
    
    # データセットの存在確認
    if not os.path.exists(DATASET_PATH):
        print(f"エラー: {DATASET_PATH} が見つかりません。")
        sys.exit(1)
    
    # モデルファイルの存在確認
    if not os.path.exists(JAWIKI_MODEL_PATH):
        print(f"エラー: {JAWIKI_MODEL_PATH} が見つかりません。")
        print("以下のURLからモデルをダウンロードしてください:")
        print("https://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/")
        print("推奨: jawiki.word_vectors.200d.txt (200次元)")
        sys.exit(1)
    
    # ユーザー入力
    print("\n以下の情報を入力してください:")
    
    # イメージ語の入力
    image_words_input = input("イメージ語(カンマ区切り): ")
    image_words = [w.strip() for w in image_words_input.replace("、", ",").split(",")]
    
    # 答えの単語の入力
    answer_word = input("答えの単語(カタカナ): ").strip()
    
    print(f"\nイメージ語: {', '.join(image_words)}")
    print(f"答えの単語: {answer_word}")
    
    # Word2Vecモデル初期化
    searcher = SemanticSearcher(DATASET_PATH, JAWIKI_MODEL_PATH)
    
    # 類似度スコアを計算(2文字ペア対応)
    print(f"\nイメージ語との類似度を計算中(日本語Wikipediaベクトル使用)...")
    score_dict, char_pair_dict = searcher.get_similarity_scores(image_words, answer_word)
    
    # データを保存
    print(f"\nデータを保存中...")
    
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
    print("  - 日本語Wikipediaベクトルを使用した類似度測定")
    print("\n次に crossword_builder.py を実行してください。")
    print("="*60)

if __name__ == "__main__":
    main()