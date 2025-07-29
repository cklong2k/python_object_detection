#!/usr/bin/env python3
"""
下載 YOLOv8 模型檔案
"""

import os
from ultralytics import YOLO

def download_yolov8_model():
    """下載 YOLOv8 模型"""
    
    # 檢查模型是否已存在
    if os.path.exists('yolov8.pt'):
        print("✅ yolov8.pt 已存在")
        return
    
    print("🔄 正在下載 YOLOv8 模型...")
    print("⏱️  首次下載需要一些時間，請耐心等候...")
    
    try:
        # 這會自動下載 YOLOv8n (nano) 模型
        model = YOLO('yolov8n.pt')
        
        # 將下載的模型重命名為 yolov8.pt
        if os.path.exists('yolov8n.pt'):
            os.rename('yolov8n.pt', 'yolov8.pt')
            print("✅ YOLOv8 模型下載完成！")
            print("📁 模型檔案: yolov8.pt")
        else:
            print("❌ 模型下載失敗")
            
    except Exception as e:
        print(f"❌ 下載模型時發生錯誤: {str(e)}")

def show_model_info():
    """顯示可用的 YOLOv8 模型資訊"""
    print("\n📊 YOLOv8 模型選項:")
    print("yolov8n.pt - Nano (最小、最快)")
    print("yolov8s.pt - Small")
    print("yolov8m.pt - Medium") 
    print("yolov8l.pt - Large")
    print("yolov8x.pt - Extra Large (最大、最精確)")
    print("\n💡 預設使用 yolov8n (Nano) 版本")
    print("💡 如需更高精確度，可手動下載其他版本並重命名為 yolov8.pt")

if __name__ == "__main__":
    show_model_info()
    download_yolov8_model()
    
    # 測試模型載入
    try:
        print("\n🧪 測試模型載入...")
        model = YOLO('yolov8.pt')
        print("✅ 模型載入測試成功！")
        print(f"📝 支援類別數量: {len(model.names)}")
        print("🏷️  支援的物件類別包括:")
        
        # 顯示前10個類別
        class_names = list(model.names.values())[:10]
        print(f"   {', '.join(class_names)}... 等共 {len(model.names)} 種類別")
        
    except Exception as e:
        print(f"❌ 模型載入測試失敗: {str(e)}")