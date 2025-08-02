# このコードは、サッカー場の中央円を検出し、楕円をフィッティングして描画するものですが、
# 結果として上手くい楕円が描画されません。
import cv2
import numpy as np

def detect_center_circle(image_path):
    # 画像を読み込む
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: 画像を読み込めません: {image_path}")
        return
    
    # HSVに変換
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 緑色の範囲を定義（フィールド）
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 明るい部分を抽出（白線）
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # 緑色でない明るい部分 = 白線
    white_lines = cv2.bitwise_and(bright, cv2.bitwise_not(green_mask))
    
    # ノイズ除去
    kernel = np.ones((3,3), np.uint8)
    white_lines = cv2.morphologyEx(white_lines, cv2.MORPH_CLOSE, kernel)
    
    # 画像の中央部分のみを対象にする
    h, w = white_lines.shape
    mask = np.zeros_like(white_lines)
    cv2.circle(mask, (w//2, h//2), min(w,h)//3, 255, -1)
    white_center = cv2.bitwise_and(white_lines, mask)
    
    # 輪郭を検出
    contours, _ = cv2.findContours(white_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 最大の輪郭を探す
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 楕円をフィッティング
        if len(largest_contour) >= 5:
            ellipse = cv2.fitEllipse(largest_contour)
            
            # 楕円を描画（赤色、線の太さ3）
            cv2.ellipse(img, ellipse, (0, 0, 255), 3)
            
            # 中心点を描画
            center = (int(ellipse[0][0]), int(ellipse[0][1]))
            cv2.circle(img, center, 5, (0, 0, 255), -1)
            
            print(f"楕円検出成功: 中心{center}, サイズ{ellipse[1]}")
        else:
            print("楕円フィッティングに失敗しました")
    else:
        print("輪郭が見つかりませんでした")
    
    # 結果をプレビュー表示
    cv2.imshow("Preview - Press any key to save", img)
    cv2.waitKey(0)  # キー入力を待つ
    cv2.destroyAllWindows()
    
    # 結果を保存
    cv2.imwrite("output.png", img)
    print("output.pngに保存しました")

# 実行
if __name__ == "__main__":
    detect_center_circle("sample1.png")