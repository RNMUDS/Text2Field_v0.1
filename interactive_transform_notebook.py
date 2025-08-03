
# Jupyter/Colab用インタラクティブ射影変換
import cv2
import numpy as np
import matplotlib.pyplot as plt

def interactive_perspective_transform(image_path):
    """インタラクティブに四隅を選択して射影変換"""
    
    # 画像を読み込み
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    # 表示
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.title('四隅を左上→右上→右下→左下の順にクリック')
    
    # 4点をクリック
    points = plt.ginput(4)
    plt.close()
    
    # 変換
    src = np.float32(points)
    dst = np.float32([[0, 0], [1200, 0], [1200, 800], [0, 800]])
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (1200, 800))
    
    # 結果を表示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 元画像と選択点
    ax1.imshow(img)
    for i, pt in enumerate(points):
        ax1.plot(pt[0], pt[1], 'ro', markersize=10)
        ax1.text(pt[0]+10, pt[1], f'{i+1}', color='red', fontsize=12)
    ax1.set_title('Original with Selected Points')
    ax1.axis('off')
    
    # 変換結果
    ax2.imshow(warped)
    ax2.set_title('Transformed Result')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return warped, M

# 使用例
warped, M = interactive_perspective_transform('simple.png')
