from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
import eventlet
import os
from ultralytics import YOLO
import torch

eventlet.monkey_patch()  # 必要：讓 OpenCV 在 eventlet 環境中正常執行

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 添加密鑰
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')  # 明確指定 async_mode

# 載入 YOLOv8 模型
print("🔄 正在載入 YOLOv8 模型...")
try:
    # 解決 PyTorch 2.6+ weights_only 問題
    import torch.serialization
    torch.serialization.default_load = torch.serialization.load
    model = YOLO('yolov8n.pt')  # 載入模型檔案
    print("✅ YOLOv8 模型載入成功")
    print(f"📊 使用設備: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
except Exception as e:
    print(f"❌ YOLOv8 模型載入失敗: {str(e)}")
    print("💡 請確認 yolov8n.pt 檔案存在於專案目錄中")
    exit(1)

# YOLOv8 物件辨識
def yolov8_object_detection(image):
    """
    使用 YOLOv8 進行物件辨識
    """
    try:
        # 執行推論
        results = model(image, conf=0.3, iou=0.5)  # conf: 信心度閾值, iou: NMS閾值
        
        detected_objects = []
        
        # 處理結果
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 取得座標 (xyxy format)
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = coords
                    
                    # 計算寬高
                    width = x2 - x1
                    height = y2 - y1
                    
                    # 取得類別和信心度
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # 取得類別名稱
                    class_name = model.names[class_id]
                    
                    detected_objects.append({
                        "label": class_name,
                        "confidence": confidence,
                        "bbox": [int(x1), int(y1), int(width), int(height)],
                        "class_id": class_id
                    })
        
        return detected_objects
        
    except Exception as e:
        print(f"YOLOv8 推論錯誤: {str(e)}")
        return []

@app.route('/')
def index():
    return render_template('index.html')  # 前端頁面放在 templates/index.html

@socketio.on('connect')
def handle_connect():
    print(f"客戶端已連接: {request.sid}")
    emit('status', {'message': '已成功連接到伺服器'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"客戶端已斷線: {request.sid}")

@socketio.on('image')
def handle_image(data):
    try:
        # 檢查資料格式
        if not isinstance(data, dict):
            emit('error', {'error': '資料格式錯誤'})
            return
            
        b64_image = data.get('image_base64')
        if not b64_image:
            emit('error', {'error': '未收到影像資料'})
            return

        # base64 解碼，添加錯誤處理
        try:
            img_data = base64.b64decode(b64_image)
        except Exception as decode_error:
            emit('error', {'error': f'Base64 解碼失敗: {str(decode_error)}'})
            return
            
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            emit('error', {'error': '影像無法解碼，請檢查影像格式'})
            return

        print(f"收到影像，尺寸: {image.shape}")

        # 執行 YOLOv8 物件識別
        results = yolov8_object_detection(image)
        
        print(f"🔍 檢測到 {len(results)} 個物件")

        # 回傳結果給前端
        emit('result', {
            'objects': results,
            'image_size': {'width': image.shape[1], 'height': image.shape[0]},
            'status': 'success'
        })

    except Exception as e:
        print(f"處理影像時發生錯誤: {str(e)}")
        emit('error', {'error': f'伺服器錯誤：{str(e)}'})

@socketio.on_error_default
def default_error_handler(e):
    print(f"SocketIO 錯誤: {str(e)}")
    emit('error', {'error': '連線發生錯誤'})

if __name__ == '__main__':
    # 檢查 templates 目錄是否存在
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("已創建 templates 目錄")
    
    # SSL 憑證路徑
    cert_dir = 'certs'
    key_file = os.path.join(cert_dir, 'key.pem')  # 私鑰檔案
    cert_file = os.path.join(cert_dir, 'cert.pem')  # 憑證檔案
    
    # 檢查憑證檔案是否存在
    if not os.path.exists(key_file):
        print(f"❌ 找不到私鑰檔案: {key_file}")
        exit(1)
    
    if not os.path.exists(cert_file):
        print(f"❌ 找不到憑證檔案: {cert_file}")
        exit(1)
    
    print("✅ SSL 憑證檔案檢查通過")
    print("🚀 HTTPS 伺服器啟動中...")
    print("📱 請在瀏覽器中開啟: https://localhost:9001")
    print("🌐 或使用你的 IP: https://your-ip:9001")
    print("⚠️  如果是自簽憑證，瀏覽器可能會顯示安全警告，請選擇「繼續前往」")
    
    # 使用 eventlet 相容的 SSL 設定方式
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=9001, 
        debug=True,
        certfile=cert_file,  # 直接傳入憑證檔案路徑
        keyfile=key_file     # 直接傳入私鑰檔案路徑
    )