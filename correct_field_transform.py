import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_field_transform():
    """image.pngの正確なフィールド変換"""
    
    # 画像を読み込み
    img = cv2.imread('image.png')
    h, w = img.shape[:2]
    
    # この画像の特徴：
    # - スタジアムから撮影
    # - フィールドの手前側（タッチライン）が見える
    # - センターラインが中央に見える
    # - ペナルティエリアも部分的に見える
    
    # より正確な四隅の座標（画像を詳細に観察）
    src_points = np.float32([
        [100, 380],    # 左上（左側タッチラインとフィールド上端の交点）
        [700, 380],    # 右上（右側タッチラインとフィールド上端の交点）
        [800, 520],    # 右下（右側タッチラインとフィールド下端の交点）
        [0, 520]       # 左下（左側タッチラインとフィールド下端の交点）
    ])
    
    # 出力サイズ（見える範囲は約半分のフィールド）
    # 幅は68m（フィールドの幅）、高さは約50m（見える範囲）
    out_width = 680   # 68m × 10px/m
    out_height = 500  # 50m × 10px/m
    
    dst_points = np.float32([
        [0, 0],
        [out_width, 0],
        [out_width, out_height],
        [0, out_height]
    ])
    
    # 射影変換
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (out_width, out_height))
    
    # 可視化
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 元画像と四隅
    img_corners = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    labels = ['左上', '右上', '右下', '左下']
    
    for i, (pt, color, label) in enumerate(zip(src_points, colors, labels)):
        cv2.circle(img_corners, tuple(pt.astype(int)), 10, color, -1)
        cv2.putText(img_corners, label, 
                   (int(pt[0]+10), int(pt[1]-5)), 
                   cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
    
    # 四隅を線で結ぶ
    pts = src_points.astype(int)
    cv2.polylines(img_corners, [pts], True, (255, 255, 255), 2)
    
    ax1.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    ax1.set_title('Original with Field Corners')
    ax1.axis('off')
    
    # トップビュー
    ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'Top View ({out_width}×{out_height}px)')
    ax2.axis('off')
    
    # グリッドを追加したトップビュー
    warped_grid = warped.copy()
    
    # 10m間隔のグリッド
    for x in range(0, out_width, 100):  # 10m間隔
        cv2.line(warped_grid, (x, 0), (x, out_height), (255, 255, 0), 1)
    for y in range(0, out_height, 100):  # 10m間隔
        cv2.line(warped_grid, (0, y), (out_width, y), (255, 255, 0), 1)
    
    # フィールドの特徴を描画
    # センターライン（推定位置）
    center_x = out_width // 2
    cv2.line(warped_grid, (center_x, 0), (center_x, out_height), (0, 0, 255), 3)
    
    ax3.imshow(cv2.cvtColor(warped_grid, cv2.COLOR_BGR2RGB))
    ax3.set_title('Top View with 10m Grid')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('correct_field_transform.png', dpi=150)
    plt.close()
    
    # 保存
    cv2.imwrite('field_topview_correct.png', warped)
    np.save('transform_matrix_correct.npy', M)
    
    # 選手位置のテスト
    print("=== 選手位置の変換テスト ===")
    
    # 画像から目視で特定した選手位置
    players = [
        (250, 450, "白1"),
        (350, 430, "白2"),
        (450, 440, "青1"),
        (550, 450, "青2"),
    ]
    
    for x, y, name in players:
        # 変換
        pt = cv2.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float32), M
        )[0][0]
        
        # メートル換算（10px = 1m）
        x_m = pt[0] / 10
        y_m = pt[1] / 10
        
        print(f"{name}: 画像({x},{y}) → フィールド({x_m:.1f}m, {y_m:.1f}m)")
    
    return warped, M

def create_field_map():
    """2Dフィールドマップを作成"""
    # 変換行列を読み込み
    M = np.load('transform_matrix_correct.npy')
    
    # フィールドのサイズ（メートル）
    field_width = 68
    field_height = 50  # 見える範囲
    
    # 図を作成
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#2a7f2a')
    
    # フィールドの枠
    field = plt.Rectangle((0, 0), field_width, field_height, 
                         fill=False, edgecolor='white', linewidth=3)
    ax.add_patch(field)
    
    # センターライン
    ax.plot([field_width/2, field_width/2], [0, field_height], 'white', linewidth=2)
    
    # グリッド（10m間隔）
    for x in range(0, int(field_width)+1, 10):
        ax.axvline(x, color='white', alpha=0.3, linestyle='--')
    for y in range(0, int(field_height)+1, 10):
        ax.axhline(y, color='white', alpha=0.3, linestyle='--')
    
    # 選手位置（例）
    players_img = [
        (250, 450, "白1"),
        (350, 430, "白2"),
        (450, 440, "青1"),
        (550, 450, "青2"),
    ]
    
    for x_img, y_img, name in players_img:
        # 画像座標をフィールド座標に変換
        pt = cv2.perspectiveTransform(
            np.array([[[x_img, y_img]]], dtype=np.float32), M
        )[0][0]
        
        # メートル換算
        x_m = pt[0] / 10
        y_m = pt[1] / 10
        
        # 色を決定
        color = 'white' if '白' in name else 'blue'
        ax.plot(x_m, y_m, 'o', color=color, markersize=12, 
                markeredgecolor='black', markeredgewidth=2)
        ax.text(x_m+1, y_m+1, name, color='white', fontsize=10)
    
    # 設定
    ax.set_xlim(-5, field_width+5)
    ax.set_ylim(-5, field_height+5)
    ax.set_aspect('equal')
    ax.set_title('2D Field Position Map', fontsize=16, color='white', pad=20)
    ax.set_xlabel('Width (meters)', color='white')
    ax.set_ylabel('Length (meters)', color='white')
    
    # 軸の設定
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    
    plt.tight_layout()
    plt.savefig('field_2d_map_correct.png', dpi=150, facecolor='#1a5a1a')
    plt.close()
    
    print("\n✓ 2Dフィールドマップを生成 → field_2d_map_correct.png")

if __name__ == "__main__":
    print("=== 正確なフィールド変換 ===\n")
    
    # フィールド変換
    warped, M = correct_field_transform()
    
    print("\n✓ 完了:")
    print("  - field_topview_correct.png (トップビュー)")
    print("  - transform_matrix_correct.npy (変換行列)")
    print("  - correct_field_transform.png (処理過程)")
    
    # 2Dマップ作成
    create_field_map()