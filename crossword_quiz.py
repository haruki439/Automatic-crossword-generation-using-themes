# -*- coding: utf-8 -*-
from openai import OpenAI
import json
import time
import pickle
import sys
import os

# =====================================
# 🔧 設定
# =====================================
API_KEY = "your-api-key-here"
MODEL = "gpt-4.1"

# ★★ ここにpklファイルのパスを設定 ★★
SYSTEM_DIR = os.path.dirname(os.path.abspath(__file__))
CROSSWORD_DIR = os.path.dirname(SYSTEM_DIR)

SCORE_FILE = os.path.join(
    CROSSWORD_DIR,
    "pkl",
    "word_scores.pkl"
)

CROSSWORD_DATA_FILE = os.path.join(
    CROSSWORD_DIR,
    "pkl",
    "crossword_data.pkl"
)

QUIZ_OUTPUT_FILE = os.path.join(
    CROSSWORD_DIR,
    "pkl",
    "quiz_data.json"
)

BATCH_SIZE = 5
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0

client = OpenAI(api_key=API_KEY)


# =====================================
# データ読み込み関数
# =====================================
def load_crossword_data():
    """
    crossword_data.pkl から実際に使用された単語を読み込む
    """
    print("="*60)
    print("pklファイルからデータを読み込み中...")
    print("="*60)
    
    if not os.path.exists(CROSSWORD_DATA_FILE):
        print(f"エラー: {CROSSWORD_DATA_FILE} が見つかりません。")
        print("先に crossword_builder.py を実行してください。")
        sys.exit(1)
    
    # クロスワードデータを読み込み
    with open(CROSSWORD_DATA_FILE, 'rb') as f:
        crossword_data = pickle.load(f)
    
    # データ抽出
    words_data = crossword_data.get('words_data', [])
    image_words = crossword_data.get('image_words', [])
    answer_word = crossword_data.get('answer_word', '')
    word_dict = crossword_data.get('word_dict', {})
    
    if not words_data:
        print("エラー: クロスワードに使用された単語が見つかりません。")
        sys.exit(1)
    
    print(f"\nデータ読み込み完了!")
    print(f"・イメージ語: {', '.join(image_words)}")
    print(f"・答えの単語: {answer_word}")
    print(f"・使用単語数: {len(words_data)}")
    print("\n使用単語一覧:")
    for word_info in words_data:
        original = word_dict.get(word_info['word'], '')
        if original and original != word_info['word']:
            print(f"  {word_info['clue_label']}: {word_info['word']} ({original})")
        else:
            print(f"  {word_info['clue_label']}: {word_info['word']}")
    
    return {
        'words_data': words_data,
        'image_words': image_words,
        'answer_word': answer_word,
        'word_dict': word_dict
    }


# =====================================
# クイズ生成関数
# =====================================
def build_prompt(words_data, image_text, word_dict):
    """
    GPTへのプロンプトを構築
    
    Args:
        words_data: 単語データリスト（clue_labelとword含む）
        image_text: イメージ文章
        word_dict: 単語の意味辞書（original → word）
    """
    # 単語の意味情報を追加
    word_meanings = []
    for word_info in words_data:
        w = word_info['word']
        label = word_info['clue_label']
        meaning = word_dict.get(w, "")
        if meaning and meaning != w:
            word_meanings.append(f"{label} {w}（{meaning}）")
        else:
            word_meanings.append(f"{label} {w}")
    
    # 単語リストを文字列化
    words_list = [word_info['word'] for word_info in words_data]
    
    return (
        "あなたは日本語のクロスワードクイズ作成AIです。\n"
        "ユーザーが入力した『イメージ文章』に合わせて、"
        "各単語に対して穴埋めクイズを1問ずつ生成してください。\n\n"

        "【イメージ文章（情景・状況）】\n"
        f"{image_text}\n\n"

        "【出力形式】\n"
        '[{"word":"単語","clue_label":"→1","hint":"15〜40文字のヒント",'
        '"fill":"穴埋め文","choices":["選択肢1","選択肢2","選択肢3","選択肢4"],'
        '"answer_index":0,"difficulty":"easy"}]\n\n'

        "【生成ルール】\n"
        "- fill の文は **イメージ文章の状況と自然に関連した文** にすること。\n"
        "  例：イメージが「カフェで友達と勉強」なら、卵 →「店の人気ケーキには新鮮な＿＿が使われている」\n"
        "- fill の空欄には、答えの文字数と同じ数の **全角アンダーバー 『＿』** を使う。\n"
        "- choices は自然な4択を作り、answer_index は正しい選択肢の番号。\n"
        "- clue_label はクロスワードの番号（→1, ↓2 など）で、そのまま出力すること。\n"
        "- JSON以外の文章は出力しない。\n\n"
        f"【単語リスト】\n{chr(10).join(word_meanings)}\n\n"
        f"対象単語: {', '.join(words_list)}"
    )


def call_openai(prompt):
    """OpenAI APIを呼び出す（リトライ機能付き）"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=2000,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"エラー (試行 {attempt}): {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_BACKOFF ** attempt)


def parse_json(text):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("JSONを抽出できません")
    return json.loads(text[start:end + 1])


def generate_quizzes(words_data, image_text, word_dict):
    """
    クイズを生成
    """
    results = []
    for i in range(0, len(words_data), BATCH_SIZE):
        batch = words_data[i:i + BATCH_SIZE]
        prompt = build_prompt(batch, image_text, word_dict)
        
        batch_labels = [w['clue_label'] for w in batch]
        print(f"\n単語 {', '.join(batch_labels)} を処理中...\n")
        
        text = call_openai(prompt)
        try:
            quizzes = parse_json(text)
            results.extend(quizzes)
        except Exception as e:
            print(f"JSON解析に失敗しました: {e}\n", text)
    return results


# =====================================
# メイン処理
# =====================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("クロスワードクイズ自動生成プログラム")
    print("="*60 + "\n")
    
    # pklからデータ読み込み
    data = load_crossword_data()
    
    # イメージ文章を生成（イメージ語を繋げる）
    if data['image_words']:
        image_text = f"{', '.join(data['image_words'])}に関連する状況"
    else:
        image_text = "一般的な状況"
    
    print(f"\n生成するイメージ文章: {image_text}\n")
    
    # ユーザーに確認
    print("このイメージ文章でクイズを生成しますか？")
    print("変更する場合は新しい文章を入力してください（Enterでそのまま使用）:")
    user_input = input("> ").strip()
    if user_input:
        image_text = user_input
    
    print(f"\n使用するイメージ文章: {image_text}")
    
    # クイズ生成
    print("\n" + "="*60)
    print("GPTでクイズを生成中...")
    print("="*60)
    
    quizzes = generate_quizzes(data['words_data'], image_text, data['word_dict'])
    
    # 結果表示
    print("\n" + "="*60)
    print("生成結果")
    print("="*60 + "\n")
    
    for q in quizzes:
        print(f"{q.get('clue_label', '?')}: {q['word']}")
        print(f"ヒント: {q.get('hint', '')}")
        print(f"穴埋め: {q.get('fill', '')}")
        print("-" * 60)
    
    # クイズデータを保存
    with open(QUIZ_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(quizzes, f, ensure_ascii=False, indent=2)
    
    print(f"\nクイズデータを保存しました: {QUIZ_OUTPUT_FILE}")
    print(f"生成されたクイズ数: {len(quizzes)}")
    print("\n" + "="*60)
    print("完了！")
    print("="*60)