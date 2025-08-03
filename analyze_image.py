import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_sample_image():
    """sample.jpgを詳細に分析"""
    img = cv2.imread('sample.jpg')
    h, w = img.shape[:2]
    
    print(f"画像サイズ: {w}x{h}")
    print("\n画像の特徴:")
    print("- カメラは左側スタンドから撮影")
    print("- センターラインが画像中央を縦に走る")
    print("- センターサークルが見える")
    print("- 左側のペナルティエリアが部分的に見える")
    print("- 右側（遠い側）は歪みが大きい")
    
    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # エッジ検出
    edges = cv2.Canny(gray, 50, 150)
    
    # ライン検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                           minLineLength=100, maxLineGap=10)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 元画像
    axes[0,0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title('Original Image')
    axes[0,0].axis('off')
    
    # 2. エッジ検出
    axes[0,1].imshow(edges, cmap='gray')
    axes[0,1].set_title('Edge Detection')
    axes[0,1].axis('off')
    
    # 3. 検出されたライン
    img_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    axes[1,0].imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title('Detected Lines')
    axes[1,0].axis('off')
    
    # 4. 手動で特定した重要な点
    img_points = img.copy()
    
    # 確実に識別できる点をマーク
    points = {
        "センター（推定）": (450, 270),
        "センターサークル上": (450, 210),
        "センターサークル下": (450, 330),
        "ペナルティエリア角": (165, 220),
        "タッチライン": (80, 180),
    }
    
    for name, (x, y) in points.items():
        cv2.circle(img_points, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(img_points, name, (x+10, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    axes[1,1].imshow(cv2.cvtColor(img_points, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title('Key Points')
    axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.savefig('image_analysis.png')
    plt.show()
    
    return points

def simple_homography_approach():
    """
    シンプルなアプローチ：
    センターラインとセンターサークルのみを使用
    """
    # 画像上の点（手動で微調整）
    image_points = np.array([
        [450, 270],   # センターマーク
        [450, 210],   # センターサークル上
        [450, 330],   # センターサークル下
        [390, 270],   # センターサークル左
        [510, 270],   # センターサークル右
    ], dtype=np.float32)
    
    # 対応するピッチ座標
    pitch_points = np.array([
        [52.5, 34.0],     # センターマーク
        [52.5, 24.85],    # 上（34 - 9.15）
        [52.5, 43.15],    # 下（34 + 9.15）
        [43.35, 34.0],    # 左（52.5 - 9.15）
        [61.65, 34.0],    # 右（52.5 + 9.15）
    ], dtype=np.float32)
    
    # ホモグラフィ計算
    H, mask = cv2.findHomography(image_points, pitch_points, cv2.RANSAC)
    
    return H, image_points, pitch_points

def test_simple_homography():
    """シンプルなホモグラフィをテスト"""
    H, img_pts, pitch_pts = simple_homography_approach()
    
    img = cv2.imread('sample.jpg')
    H_inv = np.linalg.inv(H)
    
    # 結果画像
    result = img.copy()
    
    # キャリブレーション点を描画
    for pt in img_pts:
        cv2.circle(result, tuple(pt.astype(int)), 10, (0, 255, 0), -1)
    
    # センターラインを描画
    for y in range(0, 69, 2):
        pitch_pt = np.array([[[52.5, y]]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
        if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
            cv2.circle(result, tuple(img_pt.astype(int)), 2, (255, 0, 0), -1)
    
    # センターサークルを描画
    for angle in np.linspace(0, 2*np.pi, 100):
        x = 52.5 + 9.15 * np.cos(angle)
        y = 34.0 + 9.15 * np.sin(angle)
        pitch_pt = np.array([[[x, y]]], dtype=np.float32)
        img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
        if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
            cv2.circle(result, tuple(img_pt.astype(int)), 2, (0, 0, 255), -1)
    
    # 10mグリッド（シンプル）
    for x in range(10, 90, 10):
        color = (0, 255, 255) if x == 50 else (255, 255, 0)
        for y in range(0, 69, 5):
            pitch_pt = np.array([[[x, y]]], dtype=np.float32)
            img_pt = cv2.perspectiveTransform(pitch_pt, H_inv)[0][0]
            if 0 <= img_pt[0] < img.shape[1] and 0 <= img_pt[1] < img.shape[0]:
                cv2.circle(result, tuple(img_pt.astype(int)), 1, color, -1)
    
    # 保存
    cv2.imwrite('simple_homography_test.png', result)
    
    # matplotlib表示
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Simple Homography Test')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('simple_homography_matplotlib.png')
    plt.show()
    
    return H

def main():
    print("=== 画像分析 ===")
    points = analyze_sample_image()
    
    print("\n=== シンプルなホモグラフィテスト ===")
    H = test_simple_homography()
    
    print("\n=== 結果 ===")
    print("ホモグラフィ行列:")
    print(H)
    
    # テスト変換
    test_pts = [
        ([450, 270], "画像中心"),
        ([300, 250], "左側の選手位置"),
        ([600, 280], "右側の選手位置"),
    ]
    
    print("\n位置変換テスト:")
    for img_pt, desc in test_pts:
        pitch_pt = cv2.perspectiveTransform(
            np.array([[img_pt]], dtype=np.float32), H
        )[0][0]
        print(f"{desc}: {img_pt} → ({pitch_pt[0]:.1f}m, {pitch_pt[1]:.1f}m)")
    
    # 保存
    np.save('homography_simple.npy', H)
    print("\n✓ homography_simple.npyに保存")

if __name__ == "__main__":
    main()