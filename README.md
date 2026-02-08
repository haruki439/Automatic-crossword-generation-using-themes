# テーマ情報を用いた日本語クロスワードの自動生成

## 概要
本リポジトリは、**テーマ情報を用いて日本語クロスワードを自動生成するシステム**です。  
テーマに基づいた単語選択とクイズ生成を行うことで、統一感のあるクロスワード目的とします。

---

## 環境設定

### 1. データセットの準備
本システムでは、日本語Wikipediaから構築された単語ベクトルデータを使用します。  
以下のURLからデータセット（txt形式）をダウンロードしてください。

https://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/

ダウンロード後、単語探索コードに当てはめてください。

---

## 手順

-- `word_searcher_Wikipedia.py` 
- `word_searcher_Sentence-BERT.py`  
　テーマと答えを単語探索コードで行う

- `crossword_builder.py`  
  単語探索を元にパズル生成

- `crossword_quiz.py`  
  パズルで使用した単語でクイズを生成


