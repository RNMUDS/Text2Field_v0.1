import cv2
import numpy as np
import matplotlib.pyplot as plt

class ManualPerspectiveTransform:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.display_image = self.image.copy()
        self.points = []
        self.window_name = 'Click 4 corners (ESC to reset, Q to quit)'
        
    def mouse_callback(self, event, x, y, flags, param):
        """マウスクリックのコールバック関数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                self.points.append([x, y])
                self.draw_points()
                print(f"Point {len(self.points)}: ({x}, {y})")
                
                if len(self.points) == 4:
                    print("\n4点選択完了！")
                    print("Enter: 変換実行")
                    print("ESC: リセット")
                    print("Q: 終了")
    
    def draw_points(self):
        """選択した点を描画"""
        self.display_image = self.image.copy()
        
        # 点と番号を描画
        for i, pt in enumerate(self.points):
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][i]
            cv2.circle(self.display_image, tuple(pt), 10, color, -1)
            cv2.putText(self.display_image, str(i+1), 
                       (pt[0]+15, pt[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # 4点揃ったら線で結ぶ
        if len(self.points) == 4:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(self.display_image, [pts], True, (255, 255, 255), 2)
        
        # ガイドテキスト
        guide_text = f"Click point {len(self.points)+1}/4" if len(self.points) < 4 else "Press Enter to transform"
        cv2.putText(self.display_image, guide_text, 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def select_corners(self):
        """四隅を選択"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n=== 手動射影変換 ===")
        print("四隅を以下の順番でクリックしてください：")
        print("1. 左上（赤）")
        print("2. 右上（緑）") 
        print("3. 右下（青）")
        print("4. 左下（黄）")
        print("\nESC: リセット, Q: 終了")
        
        self.draw_points()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                self.points = []
                self.draw_points()
                print("\nリセットしました")
            
            elif key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return None
            
            elif key == 13:  # Enter
                if len(self.points) == 4:
                    cv2.destroyAllWindows()
                    return np.array(self.points, dtype=np.float32)
                else:
                    print("4点選択してください")
    
    def transform(self, src_points, output_size=(1200, 800)):
        """射影変換を実行"""
        if src_points is None:
            return None
        
        # 出力サイズ
        out_width, out_height = output_size
        
        # 変換先の四隅（長方形）
        dst_points = np.float32([
            [0, 0],              # 左上
            [out_width, 0],      # 右上
            [out_width, out_height],  # 右下
            [0, out_height]      # 左下
        ])
        
        # 射影変換行列を計算
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 画像を変換
        warped = cv2.warpPerspective(self.image, M, (out_width, out_height))
        
        return warped, M
    
    def visualize_result(self, src_points, warped, M):
        """結果を可視化"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 元画像と選択点
        img_with_points = self.image.copy()
        for i, pt in enumerate(src_points):
            color = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)][i]
            cv2.circle(img_with_points, tuple(pt.astype(int)), 10, color[::-1], -1)
        pts = src_points.astype(int)
        cv2.polylines(img_with_points, [pts], True, (255, 255, 255), 2)
        
        axes[0].imshow(cv2.cvtColor(img_with_points, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original with Selected Corners')
        axes[0].axis('off')
        
        # 変換結果
        axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Transformed Result')
        axes[1].axis('off')
        
        # グリッド付き結果
        grid_img = warped.copy()
        h, w = warped.shape[:2]
        
        # グリッド描画
        for x in range(0, w, 100):
            cv2.line(grid_img, (x, 0), (x, h), (200, 200, 200), 1)
        for y in range(0, h, 100):
            cv2.line(grid_img, (0, y), (w, y), (200, 200, 200), 1)
        
        axes[2].imshow(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Result with Grid (100px spacing)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('manual_transform_result.png', dpi=150)
        plt.close()
        
        # 変換情報を表示
        print("\n=== 変換情報 ===")
        print("選択した四隅:")
        for i, pt in enumerate(src_points):
            print(f"  点{i+1}: ({pt[0]:.0f}, {pt[1]:.0f})")
        print(f"\n出力サイズ: {w}x{h}")
        print("\n変換行列:")
        print(M)

def main():
    """メイン関数"""
    import sys
    
    # 画像ファイルを指定
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # デフォルトで利用可能な画像を探す
        import os
        candidates = ['simple.png', 'sample.jpg', 'image.png']
        image_path = None
        for candidate in candidates:
            if os.path.exists(candidate):
                image_path = candidate
                break
        
        if image_path is None:
            print("画像ファイルが見つかりません")
            print("使用方法: python manual_click_transform.py <image_file>")
            return
    
    print(f"画像を読み込み: {image_path}")
    
    # 変換器を作成
    transformer = ManualPerspectiveTransform(image_path)
    
    # 四隅を選択
    src_points = transformer.select_corners()
    
    if src_points is not None:
        # 出力サイズを選択
        print("\n出力サイズを選択:")
        print("1. 1200x800 (デフォルト)")
        print("2. 1050x680 (105m x 68m, 10px/m)")
        print("3. カスタム")
        
        choice = input("選択 (1-3) [1]: ").strip() or "1"
        
        if choice == "1":
            output_size = (1200, 800)
        elif choice == "2":
            output_size = (1050, 680)
        else:
            width = int(input("幅: "))
            height = int(input("高さ: "))
            output_size = (width, height)
        
        # 変換実行
        warped, M = transformer.transform(src_points, output_size)
        
        # 結果を保存
        output_name = image_path.replace('.', '_transformed.')
        cv2.imwrite(output_name, warped)
        np.save('manual_transform_matrix.npy', M)
        
        print(f"\n✓ 変換完了:")
        print(f"  - {output_name} (変換画像)")
        print(f"  - manual_transform_matrix.npy (変換行列)")
        
        # 結果を可視化
        transformer.visualize_result(src_points, warped, M)
        print(f"  - manual_transform_result.png (可視化)")
        
        # 変換行列の使い方を説明
        print("\n=== 変換行列の使い方 ===")
        print("# 行列を読み込み")
        print("M = np.load('manual_transform_matrix.npy')")
        print("# 新しい画像を変換")
        print(f"warped = cv2.warpPerspective(img, M, {output_size})")
        print("# 点を変換")
        print("pt_transformed = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), M)[0][0]")

if __name__ == "__main__":
    main()