# Automatic Crossword Generation Using Themes

## 概要
本リポジトリは、**テーマ情報を用いて日本語クロスワードを自動生成するシステム**です。  
テーマに基づいた単語選択とクイズ生成を行うことで、一貫性のあるクロスワードを作成することを目的としています。

本研究は、日本語語彙データおよび意味的類似度を活用し、  
クロスワード生成の自動化と品質向上を目指しています。

---

## 環境設定

### 1. データセットの準備
本システムでは、日本語Wikipediaから構築された単語ベクトルデータを使用します。  
以下のURLからデータセット（txt形式）をダウンロードしてください。

https://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/

ダウンロード後、必要に応じてプログラム内で参照パスを設定してください。

---

## ファイル構成

- `crossword_builder.py`  
  クロスワード盤面を生成するプログラム

- `crossword_quiz.py`  
  クロスワード用のクイズ（ヒント）を生成するプログラム

- `word_searcher_Wikipedia.py`  
  Wikipediaデータを用いた単語検索モジュール

- `word_searcher_Sentence-BERT.py`  
  Sentence-BERTを用いた意味的類似度に基づく単語検索モジュール

---

## 使用言語
- Python

---

## 備考
本リポジトリは研究・学習目的で作成されています。
