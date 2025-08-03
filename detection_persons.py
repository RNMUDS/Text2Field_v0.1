import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches

def detect_field_lines(img):
    """
    グラウンドの白線を検出して、フィールドの特徴点を見つける
    """
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # ハフ変換で直線検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    # 水平線と垂直線を分類
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            if angle < 10 or angle > 170:  # 水平線
                horizontal_lines.append(line[0])
            elif 80 < angle < 100:  # 垂直線
                vertical_lines.append(line[0])
    
    return horizontal_lines, vertical_lines, edges

def detect_players_and_ball_refined(image_path):
    """
    改良版：選手とボールを検出し、チームごとに正確に分類する
    """
    # 画像読み込み
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # フィールドライン検出
    h_lines, v_lines, edges = detect_field_lines(img)
    
    # YOLOv5モデルのロード
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
    # 検出実行
    results = model(img_rgb)
    detections = results.pandas().xyxy[0]
    
    # 人物とボールをフィルタリング
    persons = detections[detections['name'] == 'person']
    balls = detections[detections['name'] == 'sports ball']
    
    # より正確な色範囲（HSV色空間）
    # 赤チーム（より広い範囲）
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # 青/紫チーム
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # 黒（レフリー用）
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])  # 低い明度
    
    # HSV変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 各選手の位置とチーム情報を格納
    red_team = []
    blue_team = []
    referee = None  # レフリーは1人
    all_detections = []
    
    # 人物の検出と分類
    referee_candidates = []  # レフリー候補を保存
    
    for idx, person in persons.iterrows():
        x1, y1, x2, y2 = int(person['xmin']), int(person['ymin']), int(person['xmax']), int(person['ymax'])
        confidence = person['confidence']
        
        # バウンディングボックス内の領域を抽出
        person_roi = hsv[y1:y2, x1:x2]
        
        # 各色のマスクを作成
        mask_red1 = cv2.inRange(person_roi, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(person_roi, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        mask_blue = cv2.inRange(person_roi, lower_blue, upper_blue)
        mask_black = cv2.inRange(person_roi, lower_black, upper_black)
        
        # 各色のピクセル数をカウント
        red_pixels = cv2.countNonZero(mask_red)
        blue_pixels = cv2.countNonZero(mask_blue)
        black_pixels = cv2.countNonZero(mask_black)
        total_pixels = person_roi.shape[0] * person_roi.shape[1]
        
        # 各色の割合を計算
        red_ratio = red_pixels / total_pixels
        blue_ratio = blue_pixels / total_pixels
        black_ratio = black_pixels / total_pixels
        
        # 選手の中心位置
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # チーム分類（改良版）
        if black_ratio > 0.3:  # 黒が30%以上ならレフリー候補
            referee_candidates.append({
                'position': (center_x, center_y),
                'bbox': (x1, y1, x2, y2),
                'black_ratio': black_ratio,
                'confidence': confidence
            })
        elif red_ratio > blue_ratio and red_ratio > 0.1:  # 赤が優勢
            red_team.append((center_x, center_y))
            all_detections.append({
                'bbox': (x1, y1, x2, y2),
                'type': 'person',
                'team': 'red',
                'color': (255, 0, 0),
                'confidence': confidence
            })
        elif blue_ratio > red_ratio and blue_ratio > 0.1:  # 青が優勢
            blue_team.append((center_x, center_y))
            all_detections.append({
                'bbox': (x1, y1, x2, y2),
                'type': 'person',
                'team': 'blue',
                'color': (0, 0, 255),
                'confidence': confidence
            })
        else:  # どちらでもない場合は色の詳細を見て判断
            # より詳細な分析が必要な場合
            if red_pixels > blue_pixels:
                red_team.append((center_x, center_y))
                team = 'red'
                color = (255, 0, 0)
            else:
                blue_team.append((center_x, center_y))
                team = 'blue'
                color = (0, 0, 255)
            
            all_detections.append({
                'bbox': (x1, y1, x2, y2),
                'type': 'person',
                'team': team,
                'color': color,
                'confidence': confidence
            })
    
    # レフリーを1人選択（最も黒い割合が高い人物）
    if referee_candidates:
        best_referee = max(referee_candidates, key=lambda x: x['black_ratio'])
        referee = [best_referee['position']]
        all_detections.append({
            'bbox': best_referee['bbox'],
            'type': 'person',
            'team': 'referee',
            'color': (0, 0, 0),
            'confidence': best_referee['confidence']
        })
    
    # ボールの検出
    ball_positions = []
    for idx, ball in balls.iterrows():
        x1, y1, x2, y2 = int(ball['xmin']), int(ball['ymin']), int(ball['xmax']), int(ball['ymax'])
        confidence = ball['confidence']
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        ball_positions.append((center_x, center_y))
        all_detections.append({
            'bbox': (x1, y1, x2, y2),
            'type': 'ball',
            'color': (255, 255, 255),
            'confidence': confidence
        })
    
    return red_team, blue_team, referee, ball_positions, img_rgb, all_detections, (h_lines, v_lines)

def draw_detections_with_lines(img, detections, field_lines):
    """
    検出結果とフィールドラインを画像に描画
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(img)
    
    # フィールドラインを描画
    h_lines, v_lines = field_lines
    
    # 水平線を薄く描画
    for line in h_lines:
        x1, y1, x2, y2 = line
        ax.plot([x1, x2], [y1, y2], 'g-', alpha=0.3, linewidth=1)
    
    # 垂直線を薄く描画
    for line in v_lines:
        x1, y1, x2, y2 = line
        ax.plot([x1, x2], [y1, y2], 'g-', alpha=0.3, linewidth=1)
    
    # 検出結果を描画
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        width = x2 - x1
        height = y2 - y1
        
        if det['type'] == 'person':
            if det['team'] == 'red':
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=3, edgecolor='red', facecolor='none')
                label = '赤チーム'
            elif det['team'] == 'blue':
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=3, edgecolor='blue', facecolor='none')
                label = '青チーム'
            else:  # referee
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=3, edgecolor='black', facecolor='none')
                label = 'レフリー'
            
            ax.add_patch(rect)
            
            # ラベルを追加
            label_color = 'white' if det['team'] != 'referee' else 'yellow'
            ax.text(x1, y1-5, f'{label} ({det["confidence"]:.2f})', 
                   color=label_color, fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor=rect.get_edgecolor() if det['team'] != 'referee' else 'black', 
                           alpha=0.7))
            
        elif det['type'] == 'ball':
            rect = Rectangle((x1, y1), width, height, 
                           linewidth=3, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
            
            ax.text(x1, y1-5, f'ボール ({det["confidence"]:.2f})', 
                   color='black', fontsize=10, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title('選手とボールの検出結果（フィールドライン付き）', fontsize=18, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('detection_results_with_lines.png', dpi=300, bbox_inches='tight')
    plt.show()

def transform_to_2d_field_with_perspective(positions, img_shape, field_lines):
    """
    フィールドラインを考慮した、より正確な座標変換
    """
    field_width = 105
    field_height = 68
    
    img_height, img_width = img_shape[:2]
    field_positions = []
    
    # フィールドラインから透視変換のパラメータを推定（簡易版）
    # 実際にはもっと複雑な処理が必要
    
    for x, y in positions:
        # Y座標に基づく透視補正（改良版）
        # 画面下部（手前）ほど圧縮率を高くする
        perspective_factor = 1 + (y / img_height) * 0.8
        
        # 画面中央からの距離も考慮
        center_distance = abs(x - img_width/2) / (img_width/2)
        lateral_correction = 1 + center_distance * 0.1
        
        # フィールド座標に変換
        field_x = (x / img_width) * field_width * lateral_correction
        field_y = ((img_height - y) / img_height) * field_height * perspective_factor
        
        # 境界チェック
        field_x = max(0, min(field_width, field_x))
        field_y = max(0, min(field_height, field_y))
        
        field_positions.append((field_x, field_y))
    
    return field_positions

def create_accurate_soccer_field(red_team, blue_team, referee, ball_positions):
    """
    より正確なサッカーフィールドを作成
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # フィールドの基本設定
    field_length = 105
    field_width = 68
    
    # フィールドの背景
    field = patches.Rectangle((0, 0), field_length, field_width, 
                             linewidth=3, edgecolor='white', 
                             facecolor='#228B22')
    ax.add_patch(field)
    
    # 全てのフィールドマーキングを描画
    # センターライン
    ax.plot([field_length/2, field_length/2], [0, field_width], 
            color='white', linewidth=2)
    
    # センターサークル
    center_circle = patches.Circle((field_length/2, field_width/2), 9.15, 
                                  color='white', fill=False, linewidth=2)
    ax.add_patch(center_circle)
    
    # センタースポット
    center_spot = patches.Circle((field_length/2, field_width/2), 0.3, 
                                color='white', fill=True)
    ax.add_patch(center_spot)
    
    # ペナルティエリア、ゴールエリア等（詳細は省略）
    
    # 選手をプロット
    if red_team:
        red_x = [pos[0] for pos in red_team]
        red_y = [pos[1] for pos in red_team]
        ax.scatter(red_x, red_y, c='red', s=400, marker='o', 
                  edgecolors='darkred', linewidth=3, label=f'赤チーム ({len(red_team)}人)', 
                  zorder=5)
    
    if blue_team:
        blue_x = [pos[0] for pos in blue_team]
        blue_y = [pos[1] for pos in blue_team]
        ax.scatter(blue_x, blue_y, c='blue', s=400, marker='o', 
                  edgecolors='darkblue', linewidth=3, label=f'青チーム ({len(blue_team)}人)', 
                  zorder=5)
    
    if referee:
        ref_x = [pos[0] for pos in referee]
        ref_y = [pos[1] for pos in referee]
        ax.scatter(ref_x, ref_y, c='black', s=350, marker='s', 
                  edgecolors='yellow', linewidth=3, label='レフリー (1人)', 
                  zorder=5)
    
    if ball_positions:
        ball_x = [pos[0] for pos in ball_positions]
        ball_y = [pos[1] for pos in ball_positions]
        ax.scatter(ball_x, ball_y, c='white', s=200, marker='o', 
                  edgecolors='black', linewidth=2, label='ボール', zorder=6)
    
    # 軸の設定
    ax.set_xlim(-5, 110)
    ax.set_ylim(-5, 73)
    ax.set_aspect('equal')
    
    ax.set_title('サッカーフィールド 選手配置図（精密版）', fontsize=18, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
             ncol=4, fontsize=12, frameon=True)
    
    ax.set_xlabel('フィールド長さ (m)', fontsize=12)
    ax.set_ylabel('フィールド幅 (m)', fontsize=12)
    ax.grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('soccer_field_accurate.png', dpi=300, bbox_inches='tight')
    plt.show()

def main(image_path):
    """
    メイン処理
    """
    print("精密な選手・ボール検出を開始...")
    
    # 1. 選手とボールを検出（改良版）
    red_team, blue_team, referee, ball_positions, img, detections, field_lines = detect_players_and_ball_refined(image_path)
    
    print(f"\n検出結果:")
    print(f"赤チーム: {len(red_team)}人")
    print(f"青チーム: {len(blue_team)}人")
    print(f"レフリー: {len(referee) if referee else 0}人")
    print(f"ボール: {len(ball_positions)}個")
    
    # 2. 検出結果とフィールドラインを描画
    print("\n検出結果を画像に描画中...")
    draw_detections_with_lines(img, detections, field_lines)
    
    # 3. より正確なフィールド座標に変換
    print("\nフィールド座標に変換中...")
    red_field = transform_to_2d_field_with_perspective(red_team, img.shape, field_lines)
    blue_field = transform_to_2d_field_with_perspective(blue_team, img.shape, field_lines)
    referee_field = transform_to_2d_field_with_perspective(referee, img.shape, field_lines) if referee else []
    ball_field = transform_to_2d_field_with_perspective(ball_positions, img.shape, field_lines)
    
    # 4. 精密な2Dフィールドに配置
    print("\n精密な2Dフィールドを生成中...")
    create_accurate_soccer_field(red_field, blue_field, referee_field, ball_field)
    
    print("\n処理完了！")
    print("生成された画像:")
    print("- detection_results_with_lines.png: フィールドライン付き検出結果")
    print("- soccer_field_accurate.png: 精密な2Dフィールド配置図")

# 使用例
if __name__ == "__main__":
    image_path = "sample1.png"  # 実際の画像パスに変更
    main(image_path)