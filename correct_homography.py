import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_correct_points():
    """
    画像から正確な特徴点を見つける
    問題：カメラがピッチの横から撮影されているため、
    遠近法の歪みが大きい
    """
    img = cv2.imread('sample.jpg')
    h, w = img.shape[:2]
    print(f"画像サイズ: {w}x{h}")
    
    # 画像を詳細に分析
    # 1. センターラインは画像の中央を縦に走っている（x=450付近）
    # 2. センターサークルが見える
    # 3. カメラは左側から撮影されている（左側が近く、右側が遠い）
    
    # より多くの点を使用（6点以上）
    image_points = []
    pitch_points = []
    
    # センターライン上の点を複数取る
    # センターマーク
    image_points.append([450, 270])
    pitch_points.append([52.5, 34.0])
    
    # センターサークルと中央線の交点（上）
    image_points.append([450, 220])
    pitch_points.append([52.5, 24.85])
    
    # センターサークルと中央線の交点（下）
    image_points.append([450, 320])
    pitch_points.append([52.5, 43.15])
    
    # ペナルティエリアのコーナー（推定）
    # 左側ペナルティエリア右上
    image_points.append([150, 200])
    pitch_points.append([16.5, 13.85])
    
    # 左側ペナルティエリア右下
    image_points.append([150, 340])
    pitch_points.append([16.5, 54.15])
    
    # タッチライン上の点（推定）
    image_points.append([50, 170])
    pitch_points.append([0, 0])  # 左上コーナー
    
    image_points = np.array(image_points, dtype=np.float32)
    pitch_points = np.array(pitch_points, dtype=np.float32)
    
    # ホモグラフィ計算（RANSAC使用）
    H, mask = cv2.findHomography(image_points, pitch_points, cv2.RANSAC, 5.0)
    
    # 使用された点を確認
    print(f"\n使用された点: {np.sum(mask)} / {len(mask)}")
    
    return H, image_points, pitch_points, mask

def test_homography_quality(H, img):
    """ホモグラフィの品質をテスト"""
    h, w = img.shape[:2]
    H_inv = np.linalg.inv(H)
    
    # テストポイント：ピッチ上の既知の点
    test_points = [
        # ピッチの四隅
        ([0, 0], "左上コーナー"),
        ([0, 68], "左下コーナー"),
        ([105, 0], "右上コーナー"),
        ([105, 68], "右下コーナー"),
        # センターライン上
        ([52.5, 0], "センターライン上端"),
        ([52.5, 68], "センターライン下端"),
        # ペナルティエリア
        ([16.5, 34], "左ペナルティエリア右中央"),
        ([88.5, 34], "右ペナルティエリア左中央"),
    ]
    
    visible_count = 0
    for pitch_pt, desc in test_points:
        img_pt = cv2.perspectiveTransform(
            np.array([[pitch_pt]], dtype=np.float32), H_inv
        )[0][0]
        
        if 0 <= img_pt[0] < w and 0 <= img_pt[1] < h:
            visible_count += 1
            print(f"{desc}: 画像内 ({img_pt[0]:.0f}, {img_pt[1]:.0f})")
        else:
            print(f"{desc}: 画像外")
    
    print(f"\n可視範囲: {visible_count}/{len(test_points)} 点")
    
def visualize_correct_homography(H, image_points, pitch_points, mask):
    """正しいホモグラフィを可視化"""
    img = cv2.imread('sample.jpg')
    H_inv = np.linalg.inv(H)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. キャリブレーション点
    ax1 = axes[0, 0]
    img_points = img.copy()
    for i, (img_pt, is_used) in enumerate(zip(image_points, mask)):
        color = (0, 255, 0) if is_used else (0, 0, 255)
        cv2.circle(img_points, tuple(img_pt.astype(int)), 8, color, -1)
        cv2.putText(img_points, str(i+1), 
                   (int(img_pt[0]+10), int(img_pt[1]-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    ax1.imshow(cv2.cvtColor(img_points, cv2.COLOR_BGR2RGB))
    ax1.set_title('Calibration Points (Green=Used, Red=Outlier)')
    ax1.axis('off')
    
    # 2. 細かいグリッド（2m間隔）
    ax2 = axes[0, 1]
    img_fine_grid = img.copy()
    
    # 見える範囲のみ描画
    for x in range(0, 70, 2):
        pts = []
        for y in np.linspace(0, 68, 100):
            pitch_pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
            if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
                pts.append(img_pt)
        
        if len(pts) > 10:
            pts = np.array(pts, dtype=np.int32)
            color = (0, 255, 255) if x == 52 else (200, 200, 0)
            cv2.polylines(img_fine_grid, [pts], False, color, 1)
    
    ax2.imshow(cv2.cvtColor(img_fine_grid, cv2.COLOR_BGR2RGB))
    ax2.set_title('2m Grid (Vertical Lines)')
    ax2.axis('off')
    
    # 3. ピッチ特徴
    ax3 = axes[0, 2]
    img_features = img.copy()
    
    # センターサークル
    angles = np.linspace(0, 2*np.pi, 200)
    circle_pts = []
    for angle in angles:
        x = 52.5 + 9.15 * np.cos(angle)
        y = 34.0 + 9.15 * np.sin(angle)
        pitch_pt = np.array([[[x, y]]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
        if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
            circle_pts.append(img_pt.astype(int))
    
    if len(circle_pts) > 10:
        cv2.polylines(img_features, [np.array(circle_pts)], False, (0, 0, 255), 3)
    
    # センターライン
    center_pts = []
    for y in range(0, 69):
        pitch_pt = np.array([[[52.5, y]]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
        if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
            center_pts.append(img_pt.astype(int))
    
    if len(center_pts) > 1:
        cv2.polylines(img_features, [np.array(center_pts)], False, (255, 0, 0), 3)
    
    ax3.imshow(cv2.cvtColor(img_features, cv2.COLOR_BGR2RGB))
    ax3.set_title('Pitch Features (Circle & Center Line)')
    ax3.axis('off')
    
    # 4. 横線グリッド
    ax4 = axes[1, 0]
    img_horizontal = img.copy()
    
    for y in range(0, 69, 5):
        pts = []
        for x in np.linspace(0, 70, 100):
            pitch_pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
            if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
                pts.append(img_pt)
        
        if len(pts) > 10:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img_horizontal, [pts], False, (0, 255, 255), 1)
    
    ax4.imshow(cv2.cvtColor(img_horizontal, cv2.COLOR_BGR2RGB))
    ax4.set_title('5m Grid (Horizontal Lines)')
    ax4.axis('off')
    
    # 5. ペナルティエリア
    ax5 = axes[1, 1]
    img_penalty = img.copy()
    
    # 左側ペナルティエリア
    penalty_box = [
        [0, 13.85], [16.5, 13.85], [16.5, 54.15], [0, 54.15]
    ]
    penalty_pts = []
    for pt in penalty_box:
        pitch_pt = np.array([[pt]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
        if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
            penalty_pts.append(img_pt.astype(int))
    
    if len(penalty_pts) >= 2:
        cv2.polylines(img_penalty, [np.array(penalty_pts)], False, (255, 0, 255), 3)
    
    ax5.imshow(cv2.cvtColor(img_penalty, cv2.COLOR_BGR2RGB))
    ax5.set_title('Penalty Area')
    ax5.axis('off')
    
    # 6. 統合ビュー
    ax6 = axes[1, 2]
    img_all = img.copy()
    
    # 10mグリッド
    for x in range(0, 70, 10):
        pts = []
        for y in range(0, 69):
            pitch_pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
            if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
                pts.append(img_pt.astype(int))
        if len(pts) > 1:
            cv2.polylines(img_all, [np.array(pts)], False, (255, 255, 0), 2)
    
    for y in range(0, 69, 10):
        pts = []
        for x in range(0, 70):
            pitch_pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
            if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
                pts.append(img_pt.astype(int))
        if len(pts) > 1:
            cv2.polylines(img_all, [np.array(pts)], False, (255, 255, 0), 2)
    
    ax6.imshow(cv2.cvtColor(img_all, cv2.COLOR_BGR2RGB))
    ax6.set_title('10m Grid (Final Result)')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig('homography_correct.png', dpi=150)
    plt.show()

def main():
    print("=== 正しいホモグラフィ計算 ===\n")
    
    # 1. 正確な点を見つける
    H, img_pts, pitch_pts, mask = find_correct_points()
    
    # 2. 品質テスト
    print("\n=== 品質テスト ===")
    img = cv2.imread('sample.jpg')
    test_homography_quality(H, img)
    
    # 3. 可視化
    print("\n=== 可視化 ===")
    visualize_correct_homography(H, img_pts, pitch_pts, mask)
    
    # 4. 保存
    np.save('homography_correct.npy', H)
    print("\n✓ homography_correct.npyに保存しました")
    
    # デバッグ情報
    print("\nホモグラフィ行列:")
    print(H)

if __name__ == "__main__":
    main()