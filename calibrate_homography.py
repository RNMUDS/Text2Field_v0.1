import cv2
import numpy as np

def get_homography_matrix():
    """
    sample.jpg用に調整されたホモグラフィ行列を返す
    手動で特定した特徴点を使用
    """
    # 画像上の特徴点（ピクセル座標）
    # sample.jpgを詳細に観察して特定
    image_points = np.array([
        [450, 240],  # センターマーク
        [450, 195],  # センターサークル上部
        [450, 285],  # センターサークル下部
        [380, 240],  # センターサークル左
    ], dtype=np.float32)
    
    # 対応するピッチ座標（メートル）
    pitch_points = np.array([
        [52.5, 34.0],    # センターマーク
        [52.5, 24.85],   # 上交点（34 - 9.15）
        [52.5, 43.15],   # 下交点（34 + 9.15）
        [43.35, 34.0],   # 左端（52.5 - 9.15）
    ], dtype=np.float32)
    
    # ホモグラフィ計算
    H, mask = cv2.findHomography(image_points, pitch_points, cv2.RANSAC)
    
    return H, image_points, pitch_points

def test_homography():
    """ホモグラフィの精度をテスト"""
    H, img_pts, pitch_pts = get_homography_matrix()
    
    print("=== ホモグラフィ精度テスト ===")
    print("\nホモグラフィ行列:")
    print(H)
    
    print("\n再投影誤差:")
    errors = []
    for i, (img_pt, pitch_pt) in enumerate(zip(img_pts, pitch_pts)):
        # 変換
        projected = cv2.perspectiveTransform(
            np.array([[img_pt]], dtype=np.float32), H
        )[0][0]
        
        error = np.linalg.norm(projected - pitch_pt)
        errors.append(error)
        
        print(f"点{i+1}: ({pitch_pt[0]:.1f}, {pitch_pt[1]:.1f}) → ({projected[0]:.1f}, {projected[1]:.1f}) (誤差: {error:.2f}m)")
    
    print(f"\n平均誤差: {np.mean(errors):.2f}m")
    
    # テスト変換
    print("\n=== 選手位置の変換例 ===")
    test_positions = [
        (300, 260, "左の赤い選手"),
        (500, 300, "右の青い選手"),
        (450, 240, "センター"),
    ]
    
    for x, y, desc in test_positions:
        pt = cv2.perspectiveTransform(
            np.array([[[x, y]]], dtype=np.float32), H
        )[0][0]
        print(f"{desc}: ({x},{y}) → ({pt[0]:.1f}m, {pt[1]:.1f}m)")
    
    # 保存
    np.save('homography_matrix.npy', H)
    print("\n✓ homography_matrix.npyに保存しました")
    
    return H

def visualize_homography(image_path='sample.jpg'):
    """ホモグラフィを可視化"""
    import matplotlib.pyplot as plt
    
    # 画像読み込み
    img = cv2.imread(image_path)
    H, img_pts, pitch_pts = get_homography_matrix()
    H_inv = np.linalg.inv(H)
    
    # グリッド描画
    img_grid = img.copy()
    
    # 10m間隔のグリッド
    for x in range(0, 106, 10):
        pts = []
        for y in range(0, 69):
            pitch_pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
            if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
                pts.append(img_pt)
        
        if len(pts) > 1:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img_grid, [pts], False, (255, 255, 0), 2)
    
    for y in range(0, 69, 10):
        pts = []
        for x in range(0, 106):
            pitch_pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
            if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
                pts.append(img_pt)
        
        if len(pts) > 1:
            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(img_grid, [pts], False, (255, 255, 0), 2)
    
    # キャリブレーション点を描画
    for pt in img_pts:
        cv2.circle(img_grid, tuple(pt.astype(int)), 8, (0, 0, 255), -1)
    
    # 表示
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_grid, cv2.COLOR_BGR2RGB))
    plt.title('Homography Grid (10m spacing)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('homography_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ homography_result.pngに保存しました")

if __name__ == "__main__":
    # テスト実行
    H = test_homography()
    
    # 可視化
    visualize_homography()
    
    print("\n=== 使用方法 ===")
    print("import numpy as np")
    print("H = np.load('homography_matrix.npy')")
    print("# 画像座標をピッチ座標に変換:")
    print("pitch_pt = cv2.perspectiveTransform(img_pt, H)")