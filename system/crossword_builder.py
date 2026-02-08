# -*- coding: utf-8 -*-
import os
import json
import random
import platform
from PIL import Image, ImageDraw, ImageFont
import copy
import sys
import pickle

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

SCORE_FILE = os.path.join(
    CROSSWORD_DIR,
    "pkl",
    "word_scores.pkl"
)

SEARCHER_FILE = os.path.join(
    CROSSWORD_DIR,
    "pkl",
    "searcher.pkl"
)

CROSSWORD_DATA_FILE = os.path.join(
    CROSSWORD_DIR,
    "pkl",
    "crossword_data.pkl"
)

OUTPUT_FILE = os.path.join(
    CROSSWORD_DIR,
    "output",
    "dynamic_crossword_5x5.png"
)

GRID_SIZE = 5
CELL_SIZE = 70
MAX_FILL_ATTEMPTS = 100
ANSWER_WORD_MIN_SCORE = 0.2  # 答えの文字を含む単語の最低類似度

# ================
# SemanticSearcherクラス
# ================
class SemanticSearcher:
    def __init__(self, dataset_path=None, model_name=None):
        pass
    
    def get_words_containing_char(self, char):
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
# グリッド操作
# ================
def load_wordlist(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    word_dict = {}
    katakana_only = []
    
    for item in data:
        word = item.get("word", "")
        original = item.get("original", "")
        
        if not original:
            continue
        
        if not all('\u30A0' <= char <= '\u30FF' for char in original):
            continue
        if original[0] in ['ン', 'ー']:
            continue
        
        katakana_only.append(original)
        word_dict[original] = word
    
    return katakana_only, word_dict

def create_empty_grid(size):
    grid = []
    for y in range(size):
        row = [None for _ in range(size)]
        grid.append(row)
    return grid

def place_word(grid, word, x, y, direction):
    if direction == "horizontal":
        for i, ch in enumerate(word):
            grid[y][x+i] = ch
    else:
        for i, ch in enumerate(word):
            grid[y+i][x] = ch

def is_grid_empty(grid):
    for row in grid:
        for cell in row:
            if isinstance(cell, str):
                return False
    return True

def has_crossing(grid, word, x, y, direction):
    L = len(word)
    crossing_count = 0
    
    if direction == 'horizontal':
        for i in range(L):
            if isinstance(grid[y][x+i], str):
                crossing_count += 1
    else:
        for i in range(L):
            if isinstance(grid[y+i][x], str):
                crossing_count += 1
    
    return crossing_count > 0

def can_place_word(grid, word, x, y, direction, require_crossing=True):
    L = len(word)
    
    if direction == 'horizontal':
        if x + L > GRID_SIZE:
            return False
        
        for i in range(L):
            if grid[y][x+i] == 1:
                return False
        
        if x > 0 and grid[y][x-1] not in [1, None]:
            return False
        if x + L < GRID_SIZE and grid[y][x+L] not in [1, None]:
            return False
        
        for i in range(L):
            cell = grid[y][x+i]
            if isinstance(cell, str) and cell != word[i]:
                return False
        
        is_start = (x == 0 or grid[y][x-1] == 1 or grid[y][x-1] is None)
        if is_start and word[0] in ['ン', 'ー']:
            return False
            
    else:
        if y + L > GRID_SIZE:
            return False
        
        for i in range(L):
            if grid[y+i][x] == 1:
                return False
        
        if y > 0 and grid[y-1][x] not in [1, None]:
            return False
        if y + L < GRID_SIZE and grid[y+L][x] not in [1, None]:
            return False
        
        for i in range(L):
            cell = grid[y+i][x]
            if isinstance(cell, str) and cell != word[i]:
                return False
        
        is_start = (y == 0 or grid[y-1][x] == 1 or grid[y-1][x] is None)
        if is_start and word[0] in ['ン', 'ー']:
            return False
    
    if require_crossing and not is_grid_empty(grid):
        if not has_crossing(grid, word, x, y, direction):
            return False
    
    return True

def check_crossing_words_valid(grid, word, x, y, direction, wordlist_set):
    L = len(word)
    temp_grid = copy.deepcopy(grid)
    place_word(temp_grid, word, x, y, direction)
    
    if direction == 'horizontal':
        for i in range(L):
            cx, cy = x + i, y
            
            start_y = cy
            while start_y > 0 and isinstance(temp_grid[start_y-1][cx], str):
                start_y -= 1
            
            end_y = cy
            while end_y < GRID_SIZE - 1 and isinstance(temp_grid[end_y+1][cx], str):
                end_y += 1
            
            if end_y - start_y + 1 >= 2:
                cross_word = ''.join([temp_grid[yy][cx] for yy in range(start_y, end_y + 1)])
                
                if cross_word[0] in ['ン', 'ー']:
                    return False
                
                if cross_word not in wordlist_set:
                    return False
    
    else:
        for i in range(L):
            cx, cy = x, y + i
            
            start_x = cx
            while start_x > 0 and isinstance(temp_grid[cy][start_x-1], str):
                start_x -= 1
            
            end_x = cx
            while end_x < GRID_SIZE - 1 and isinstance(temp_grid[cy][end_x+1], str):
                end_x += 1
            
            if end_x - start_x + 1 >= 2:
                cross_word = ''.join([temp_grid[cy][xx] for xx in range(start_x, end_x + 1)])
                
                if cross_word[0] in ['ン', 'ー']:
                    return False
                
                if cross_word not in wordlist_set:
                    return False
    
    return True

def would_create_black_line(grid, x, y):
    left_black = 0
    for i in range(x-1, -1, -1):
        if grid[y][i] == 1:
            left_black += 1
        else:
            break
    
    right_black = 0
    for i in range(x+1, GRID_SIZE):
        if grid[y][i] == 1:
            right_black += 1
        else:
            break
    
    if left_black + right_black + 1 >= GRID_SIZE:
        return True
    
    top_black = 0
    for i in range(y-1, -1, -1):
        if grid[i][x] == 1:
            top_black += 1
        else:
            break
    
    bottom_black = 0
    for i in range(y+1, GRID_SIZE):
        if grid[i][x] == 1:
            bottom_black += 1
        else:
            break
    
    if top_black + bottom_black + 1 >= GRID_SIZE:
        return True
    
    return False

def check_all_words_connected(grid):
    char_cells = []
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if isinstance(grid[y][x], str):
                char_cells.append((y, x))
    
    if not char_cells:
        return True
    
    visited = set()
    stack = [char_cells[0]]
    visited.add(char_cells[0])
    
    while stack:
        y, x = stack.pop()
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                if (ny, nx) not in visited and isinstance(grid[ny][nx], str):
                    visited.add((ny, nx))
                    stack.append((ny, nx))
    
    return len(visited) == len(char_cells)

def try_place_word_anywhere(grid, word, wordlist_set, score_dict, used_words, require_crossing=True):
    # ★重要：既に使用済みの単語（original）はスキップ
    if word in used_words:
        return None, None, None
    
    positions = []
    
    if is_grid_empty(grid):
        require_crossing = False
    
    for direction in ['horizontal', 'vertical']:
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if can_place_word(grid, word, x, y, direction, require_crossing):
                    if check_crossing_words_valid(grid, word, x, y, direction, wordlist_set):
                        temp_grid = copy.deepcopy(grid)
                        place_word(temp_grid, word, x, y, direction)
                        if check_all_words_connected(temp_grid):
                            positions.append((x, y, direction))
    
    if positions:
        x, y, direction = random.choice(positions)
        return x, y, direction
    
    return None, None, None

def place_answer_words(grid, answer_word, wordlist_set, word_dict, score_dict, searcher, char_pair_dict):
    target_chars = list(answer_word)
    char_occurrence_tracker = {}
    answer_positions_list = []
    used_words = set()  # original（カタカナ）で重複管理
    
    for idx, char in enumerate(target_chars):
        occurrence_index = char_occurrence_tracker.get(char, 0)
        char_occurrence_tracker[char] = occurrence_index + 1
        
        # 単独文字とペアの候補を取得
        candidates = []
        
        # 単独文字の候補
        char_key = f"char_{char}"
        if char_key in char_pair_dict:
            candidates.extend(char_pair_dict[char_key])
        
        # 2文字ペアの候補（まだ配置していない文字とのペア）
        remaining_chars = target_chars[idx+1:]
        for other_char in remaining_chars:
            pair_key1 = f"pair_{char}{other_char}"
            pair_key2 = f"pair_{other_char}{char}"
            
            if pair_key1 in char_pair_dict:
                candidates.extend(char_pair_dict[pair_key1])
            if pair_key2 in char_pair_dict:
                candidates.extend(char_pair_dict[pair_key2])
        
        # 重複削除してスコアでソート
        unique_candidates = {}
        for word, score in candidates:
            if word not in unique_candidates or score > unique_candidates[word]:
                unique_candidates[word] = score
        
        word_list_scored = [(w, s) for w, s in unique_candidates.items()]
        word_list_scored.sort(key=lambda x: x[1], reverse=True)
        
        placed = False
        for word, score in word_list_scored:
            # 類似度チェック
            if score < ANSWER_WORD_MIN_SCORE:
                continue
            
            if char not in word or word[0] in ['ン', 'ー']:
                continue
            
            # ★重要：既に使用した単語（original）はスキップ
            if word in used_words:
                continue
            
            char_indices_in_word = [i for i, c in enumerate(word) if c == char]
            if len(char_indices_in_word) <= occurrence_index:
                continue
            
            x, y, direction = try_place_word_anywhere(grid, word, wordlist_set, score_dict, used_words, require_crossing=False)
            
            if x is not None:
                place_word(grid, word, x, y, direction)
                used_words.add(word)  # original（カタカナ）を追加
                
                if direction == 'horizontal':
                    char_pos = (y, x + char_indices_in_word[occurrence_index])
                else:
                    char_pos = (y + char_indices_in_word[occurrence_index], x)
                
                answer_positions_list.append(char_pos)
                
                original = word_dict.get(word, "")
                if original and original != word:
                    print(f" '{char}'({idx + 1}) → {word} ({original}) score={score:.4f} at {char_pos} ({direction})")
                else:
                    print(f" '{char}'({idx + 1}) → {word} score={score:.4f} at {char_pos} ({direction})")
                
                placed = True
                break
        
        if not placed:
            print(f"✖︎ '{char}'({idx + 1}) - 類似度{ANSWER_WORD_MIN_SCORE}以上の単語が見つかりませんでした")
            return False, answer_positions_list, used_words
    
    return True, answer_positions_list, used_words

def find_empty_slots(grid):
    slots = []
    
    for y in range(GRID_SIZE):
        x = 0
        while x < GRID_SIZE:
            if grid[y][x] == 1:
                x += 1
                continue
            
            start_x = x
            length = 0
            while x < GRID_SIZE and grid[y][x] != 1:
                length += 1
                x += 1
            
            if length >= 2:
                pattern = ''
                for i in range(length):
                    cell = grid[y][start_x + i]
                    if isinstance(cell, str):
                        pattern += cell
                    else:
                        pattern += '?'
                
                if '?' in pattern:
                    slots.append({
                        'direction': 'horizontal',
                        'y': y, 'x': start_x,
                        'length': length,
                        'pattern': pattern,
                        'empty_count': pattern.count('?')
                    })
    
    for x in range(GRID_SIZE):
        y = 0
        while y < GRID_SIZE:
            if grid[y][x] == 1:
                y += 1
                continue
            
            start_y = y
            length = 0
            while y < GRID_SIZE and grid[y][x] != 1:
                length += 1
                y += 1
            
            if length >= 2:
                pattern = ''
                for i in range(length):
                    cell = grid[start_y + i][x]
                    if isinstance(cell, str):
                        pattern += cell
                    else:
                        pattern += '?'
                
                if '?' in pattern:
                    slots.append({
                        'direction': 'vertical',
                        'y': start_y, 'x': x,
                        'length': length,
                        'pattern': pattern,
                        'empty_count': pattern.count('?')
                    })
    
    return slots

def matches_pattern(word, pattern):
    if len(word) != len(pattern):
        return False
    for w_char, p_char in zip(word, pattern):
        if p_char != '?' and p_char != w_char:
            return False
    return True

def fill_remaining_slots(grid, wordlist, wordlist_set, word_dict, score_dict, used_words):
    attempts = 0
    
    while attempts < MAX_FILL_ATTEMPTS:
        slots = find_empty_slots(grid)
        
        if not slots:
            print("全てのスロットを埋めました！")
            black_filled = fill_remaining_empty_cells(grid)
            if black_filled > 0:
                print(f"残りの空セル {black_filled} 個を黒マスで埋めました")
            return True
        
        slots.sort(key=lambda s: s['empty_count'])
        
        filled = False
        for slot in slots:
            candidates = []
            for word in wordlist:
                # ★重要：既に使用した単語（original）はスキップ
                if word in used_words:
                    continue
                    
                if matches_pattern(word, slot['pattern']):
                    if word[0] not in ['ン', 'ー']:
                        score = score_dict.get(word, 0.0)
                        candidates.append((word, score))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for word, score in candidates:
                if can_place_word(grid, word, slot['x'], slot['y'], slot['direction'], require_crossing=False):
                    if check_crossing_words_valid(grid, word, slot['x'], slot['y'], slot['direction'], wordlist_set):
                        temp_grid = copy.deepcopy(grid)
                        place_word(temp_grid, word, slot['x'], slot['y'], slot['direction'])
                        
                        if check_all_words_connected(temp_grid):
                            place_word(grid, word, slot['x'], slot['y'], slot['direction'])
                            used_words.add(word)  # original（カタカナ）を追加
                            
                            original = word_dict.get(word, "")
                            if original and original != word:
                                print(f"  配置: {word} ({original}) score={score:.4f} ({slot['direction']}, y={slot['y']}, x={slot['x']})")
                            else:
                                print(f"  配置: {word} score={score:.4f} ({slot['direction']}, y={slot['y']}, x={slot['x']})")
                            
                            filled = True
                            break
            
            if filled:
                break
        
        if not filled:
            print("  黒マスを追加中...")
            if add_black_cell_strategically(grid, slots):
                print("  黒マスを追加しました")
            else:
                print("  これ以上埋められません")
                fill_remaining_empty_cells(grid)
                return False
        
        attempts += 1
    
    print(f"最大試行回数に達しました")
    fill_remaining_empty_cells(grid)
    return False

def fill_remaining_empty_cells(grid):
    count = 0
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] is None:
                if not would_create_black_line(grid, x, y):
                    grid[y][x] = 1
                    count += 1
    return count

def add_black_cell_strategically(grid, slots):
    if not slots:
        return False
    
    difficult_slots = sorted(slots, key=lambda s: (s['empty_count'], s['length']), reverse=True)
    
    for slot in difficult_slots[:3]:
        mid_pos = slot['length'] // 2
        
        if slot['direction'] == 'horizontal':
            target_x = slot['x'] + mid_pos
            target_y = slot['y']
            
            if grid[target_y][target_x] is None:
                if not would_create_black_line(grid, target_x, target_y):
                    temp_grid = copy.deepcopy(grid)
                    temp_grid[target_y][target_x] = 1
                    
                    if check_all_words_connected(temp_grid):
                        grid[target_y][target_x] = 1
                        return True
        else:
            target_x = slot['x']
            target_y = slot['y'] + mid_pos
            
            if grid[target_y][target_x] is None:
                if not would_create_black_line(grid, target_x, target_y):
                    temp_grid = copy.deepcopy(grid)
                    temp_grid[target_y][target_x] = 1
                    
                    if check_all_words_connected(temp_grid):
                        grid[target_y][target_x] = 1
                        return True
    
    return False

def draw_grid_with_numbers(grid, output_path, answer_positions, answer_word):
    img = Image.new("RGB", (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE), "white")
    draw = ImageDraw.Draw(img)

    system = platform.system()
    font = None
    small_font = None
    answer_num_font = None
    try_paths = []
    if system == "Darwin":
        try_paths = ["/System/Library/Fonts/ヒラギノ角ゴシック W5.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"]
    elif system == "Windows":
        try_paths = ["C:\\Windows\\Fonts\\msgothic.ttc", "C:\\Windows\\Fonts\\meiryo.ttc"]
    else:
        try_paths = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]

    for p in try_paths:
        try:
            font = ImageFont.truetype(p, 32)
            small_font = ImageFont.truetype(p, 14)
            answer_num_font = ImageFont.truetype(p, 36)  # 答えの番号用（大きく）
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
        answer_num_font = ImageFont.load_default()

    # 横と縦で別々の番号システム
    clue_grid_h = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    clue_grid_v = [[None for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    h_num = 1
    v_num = 1
    word_start_positions = set()

    # 横方向の番号割り当て
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            if grid[y][x] == 1:
                continue
            is_h_start = (x == 0 or grid[y][x-1] == 1) and x < GRID_SIZE - 1 and grid[y][x+1] not in [1, None]
            if is_h_start:
                clue_grid_h[y][x] = h_num
                word_start_positions.add((y, x))
                h_num += 1
    
    # 縦方向の番号割り当て
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if grid[y][x] == 1:
                continue
            is_v_start = (y == 0 or grid[y-1][x] == 1) and y < GRID_SIZE - 1 and grid[y+1][x] not in [1, None]
            if is_v_start:
                clue_grid_v[y][x] = v_num
                word_start_positions.add((y, x))
                v_num += 1

    # 答えの位置マップ（位置 → 答えのインデックス）
    answer_position_map = {}
    for idx, pos in enumerate(answer_positions):
        answer_position_map[pos] = idx + 1

    # 半透明レイヤー用の画像を作成
    overlay = Image.new("RGBA", (GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            x0, y0 = x * CELL_SIZE, y * CELL_SIZE
            cell = grid[y][x]
            
            if cell == 1:
                draw.rectangle([x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE], fill="black")
                continue

            if (y, x) in word_start_positions:
                draw.rectangle([x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE], fill="white")
            else:
                draw.rectangle([x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE], fill="white")

            draw.rectangle([x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE], outline="black", width=2)

            # 番号表示（横と縦で分離）
            h_n = clue_grid_h[y][x]
            v_n = clue_grid_v[y][x]
            if h_n is not None:
                draw.text((x0 + 3, y0 + 1), f"→{h_n}", font=small_font, fill="blue")
            if v_n is not None:
                draw.text((x0 + CELL_SIZE - 24, y0 + 1), f"↓{v_n}", font=small_font, fill="green")

            # 文字表示
            if isinstance(cell, str):
                bbox = draw.textbbox((0,0), cell, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                draw.text((x0 + (CELL_SIZE - w)/2, y0 + (CELL_SIZE - h)/2 - 3), cell, font=font, fill="black")
                
                # 答えの文字の場合、赤丸と番号を表示
                if (y, x) in answer_position_map:
                    answer_index = answer_position_map[(y, x)]
                    
                    # 赤丸
                    circle_radius = 28
                    circle_center = (x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2)
                    draw.ellipse([circle_center[0] - circle_radius, circle_center[1] - circle_radius,
                                 circle_center[0] + circle_radius, circle_center[1] + circle_radius],
                                outline="red", width=3)
                    
                    # 半透明レイヤーに番号を描画（RGBA形式で半透明の赤色）
                    num_str = str(answer_index)
                    num_bbox = overlay_draw.textbbox((0,0), num_str, font=answer_num_font)
                    num_w = num_bbox[2] - num_bbox[0]
                    num_h = num_bbox[3] - num_bbox[1]
                    
                    # セルの中央に配置
                    num_x = x0 + (CELL_SIZE - num_w) / 2
                    num_y = y0 + (CELL_SIZE - num_h) / 2 - 3
                    
                    # 半透明の赤色で描画（アルファ値80で約30%の透明度）
                    overlay_draw.text((num_x, num_y), num_str, font=answer_num_font, fill=(255, 100, 100, 170))

    # RGBモードに変換してから半透明レイヤーを合成
    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")
    
    img.save(output_path)
    print(f"画像を保存しました: {output_path}")

def main():
    print("="*60)
    print("クロスワード生成プログラム")
    print("="*60)
    
    if not os.path.exists(SCORE_FILE):
        print(f"エラー: {SCORE_FILE} が見つかりません。")
        print("先に word_searcher.py を実行してください。")
        sys.exit(1)
    
    if not os.path.exists(SEARCHER_FILE):
        print(f"エラー: {SEARCHER_FILE} が見つかりません。")
        print("先に word_searcher.py を実行してください。")
        sys.exit(1)
    
    print("\n📂 保存データを読み込み中...")
    with open(SCORE_FILE, 'rb') as f:
        save_data = pickle.load(f)
    
    score_dict = save_data['score_dict']
    char_pair_dict = save_data.get('char_pair_dict', {})
    image_words = save_data['image_words']
    answer_word = save_data['answer_word']
    word_dict = save_data['word_dict']
    
    with open(SEARCHER_FILE, 'rb') as f:
        searcher = pickle.load(f)
    
    print(f"データ読み込み完了!")
    print(f"イメージ語: {', '.join(image_words)}")
    print(f"答えの単語: {answer_word}")
    print(f"最低類似度: {ANSWER_WORD_MIN_SCORE}")
    
    word_list, _ = load_wordlist(DATASET_PATH)
    wordlist_set = set(word_list)
    print(f"\n総単語数: {len(word_list)}")
    
    print(f"\n{GRID_SIZE}x{GRID_SIZE} グリッドを作成中...")
    grid = create_empty_grid(GRID_SIZE)
    
    target_chars = list(answer_word)
    
    print("\n" + "="*60)
    print("ステップ1: 答えの文字を含む単語を配置（交差不要、類似度0.5以上）")
    print("="*60)
    
    success, answer_positions_list, used_words = place_answer_words(
        grid, answer_word, wordlist_set, word_dict, score_dict, searcher, char_pair_dict
    )
    
    if not success:
        print("\n全ての答えの文字を配置できませんでした。")
        print("ヒント: イメージ語を変更するか、答えの単語を変更してみてください。")
        sys.exit(1)
    
    print(f"\n使用済み単語数: {len(used_words)}")
    
    print("\n" + "="*60)
    print("ステップ2: 残りのスロットを埋める")
    print("="*60)
    
    success = fill_remaining_slots(grid, word_list, wordlist_set, word_dict, score_dict, used_words)
    
    if success:
        print("\n✨ クロスワード完成！")
    else:
        print("\n一部のスロットを埋められませんでした。")
    
    if check_all_words_connected(grid):
        print("全ての単語が連結しています！")
    else:
        print("警告: 孤立した単語があります")
    
    all_answer_positions = set(answer_positions_list)
    
    draw_grid_with_numbers(grid, OUTPUT_FILE, answer_positions_list, answer_word)
    
    print("\n" + "="*60)
    print("最終グリッド:")
    print("="*60)
    for row in grid:
        print(' '.join(['##' if cell == 1 else (cell if isinstance(cell, str) else '??') for cell in row]))
    
    print("\n○答えの文字位置:")
    for idx, (char, pos) in enumerate(zip(target_chars, answer_positions_list)):
        print(f"  {idx + 1}. '{char}': {pos}")
    
    black_count = sum(1 for row in grid for cell in row if cell == 1)
    print(f"\n・黒マス数: {black_count}")
    print(f"・使用単語数: {len(used_words)}")
    
    # ★★ ここから追加・変更 ★★
    
    # グリッドから実際の単語を抽出
    def extract_words_from_grid(grid):
        """グリッドから横・縦の単語を抽出"""
        words_data = []
        
        # 横方向の単語を抽出
        h_num = 1
        for y in range(GRID_SIZE):
            x = 0
            while x < GRID_SIZE:
                if grid[y][x] == 1 or grid[y][x] is None:
                    x += 1
                    continue
                
                # 単語の開始位置
                start_x = x
                word_chars = []
                
                # 単語を構築
                while x < GRID_SIZE and isinstance(grid[y][x], str):
                    word_chars.append(grid[y][x])
                    x += 1
                
                # 2文字以上の単語のみ
                if len(word_chars) >= 2:
                    word = ''.join(word_chars)
                    words_data.append({
                        'word': word,
                        'direction': 'horizontal',
                        'number': h_num,
                        'position': (y, start_x),
                        'clue_label': f'→{h_num}'
                    })
                    h_num += 1
        
        # 縦方向の単語を抽出
        v_num = 1
        for x in range(GRID_SIZE):
            y = 0
            while y < GRID_SIZE:
                if grid[y][x] == 1 or grid[y][x] is None:
                    y += 1
                    continue
                
                # 単語の開始位置
                start_y = y
                word_chars = []
                
                # 単語を構築
                while y < GRID_SIZE and isinstance(grid[y][x], str):
                    word_chars.append(grid[y][x])
                    y += 1
                
                # 2文字以上の単語のみ
                if len(word_chars) >= 2:
                    word = ''.join(word_chars)
                    words_data.append({
                        'word': word,
                        'direction': 'vertical',
                        'number': v_num,
                        'position': (start_y, x),
                        'clue_label': f'↓{v_num}'
                    })
                    v_num += 1
        
        return words_data
    
    # グリッドから実際の単語を抽出
    extracted_words = extract_words_from_grid(grid)
    
    # クイズ生成用にデータを保存
    crossword_save_data = {
        'words_data': extracted_words,  # 抽出された単語データ
        'image_words': image_words,
        'answer_word': answer_word,
        'word_dict': word_dict,
        'grid': grid,
        'answer_positions': answer_positions_list
    }
    
    with open(CROSSWORD_DATA_FILE, 'wb') as f:
        pickle.dump(crossword_save_data, f)
    
    print(f"\nクロスワードデータを保存しました: {CROSSWORD_DATA_FILE}")
    print(f"保存された単語数: {len(extracted_words)}")
    print("\n保存された単語一覧:")
    for word_info in extracted_words:
        original = word_dict.get(word_info['word'], '')
        if original and original != word_info['word']:
            print(f"  {word_info['clue_label']}: {word_info['word']} ({original})")
        else:
            print(f"  {word_info['clue_label']}: {word_info['word']}")
    
    print("\n" + "="*60)
    print("完了！")
    print(f"イメージ語: {', '.join(image_words)}")
    print(f"答え: {answer_word}")
    print("横（→1, →2...）と縦（↓1, ↓2...）で番号を分離")
    print("答えの文字に順番番号を表示（赤丸内）")
    print("="*60)


if __name__ == "__main__":
    main()
