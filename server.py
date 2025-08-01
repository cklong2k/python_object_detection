from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
import eventlet
import os
from ultralytics import YOLO
import torch
import clip
from datetime import datetime
import qdrant
from PIL import Image
import uuid

eventlet.monkey_patch()  # 必要：讓 OpenCV 在 eventlet 環境中正常執行

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 添加密鑰
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')  # 明確指定 async_mode

# 初始化 Qdrant 客戶端
qdrant_client = qdrant.QdrantCRUD(host="localhost", port=6333, collection_name="test_collection")

# 確保 Qdrant collection 已存在
try:
    qdrant_client.create_collection(vector_size=512, distance="Cosine")
    print("✅ Qdrant collection 已成功創建或已存在")
except Exception as e:
    print(f"❌ 創建 Qdrant collection 時發生錯誤: {str(e)}")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)  # 1024 維輸出

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
        results = model(image, conf=0.3, iou=0.5, verbose=False, augment=False, agnostic_nms=False, retina_masks=False, classes=None, device='cpu')  # conf: 信心度閾值, iou: NMS閾值
        
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

# 轉成 1024維向量資料
def base64_to_vector(base64_image):
    try:
        # 解碼 base64
        img_data = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        # 轉換成 PIL 圖片
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV 是 BGR，PIL 是 RGB

        # 預處理
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)

        # 推論並取得 512 維向量
        with torch.no_grad():
            embedding = clip_model.encode_image(input_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # 向量正規化

        return embedding.squeeze(0).cpu().tolist()  # 回傳為 Python list
        
    except Exception as e:
        print(f"轉換錯誤: {str(e)}")
        return None


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

        # save image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = f"static/images/{timestamp}.jpg"
        cv2.imwrite(image_path, image)

        # 執行 YOLOv8 物件識別
        results = yolov8_object_detection(image)

        # crop bounding boxes
        for index, result in enumerate(results):
            x, y, w, h = result["bbox"]

            # 擷取時間戳與保存圖片
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_path = f"static/images/{timestamp}_{index}.jpg"

            # 裁切並儲存目標區塊
            canvas = image[y:y+h, x:x+w].copy()
            cv2.imwrite(image_path, canvas)
        # draw bounding boxes
        for index, result in enumerate(results):
            x, y, w, h = result["bbox"]

            # 擷取時間戳與保存圖片
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            image_path = f"static/images/{timestamp}_{index}.jpg"

            label = result["label"]

            # 繪製框線與標籤
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        

        # save image
        image_path = f"static/images/{timestamp}_bounding.jpg"
        cv2.imwrite(image_path, image)
        
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

@socketio.on('searchVector')
def handle_search_vector(data):
    try:
        if data.get("vector") is None:
            emit('error', {'error': '影像轉換為向量時發生錯誤'})
            return
        print(f"影像轉換為向量成功，向量長度: {len(data.get('vector'))}")

        # 在 Qdrant 中搜尋相似向量
        vector = data.get("vector")
        # print(f"🔄 正在搜尋向量: {vector}")
        results = qdrant_client.search(
                vector=vector,
                top=5
            )
        
        # 將 ScoredPoint 轉為可 JSON 的 dict
        results_json = [
            {
                "id": r.id,
                "score": r.score,
                "payload": r.payload,  # 確保 payload 是 dict
                "vector": r.vector if hasattr(r, "vector") else None  # 可選
            }
            for r in results
        ]

        print(f"搜尋到 {len(results)} 個相似向量")
        print(f"第一個搜尋結果: {results[0].payload['image_path']}")

        # 回傳搜尋結果給前端
        emit('search_result', {
            'image': results_json,
            'status': 'success'
        })

    except Exception as e:
        print(f"搜尋向量時發生錯誤: {str(e)}")
        emit('error', {'error': f'伺服器錯誤：{str(e)}'})

@socketio.on('createVector')
def handle_create_vector(data):
    try:
        # 檢查資料格式 data base64/image
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

        # save image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        image_path = f"static/images/{timestamp}.jpg"
        cv2.imwrite(image_path, image)

        # convert base64image to vector
        vector = base64_to_vector(b64_image)

        if vector is None:
            emit('error', {'error': '影像轉換為向量時發生錯誤'})
            return

        print(f"影像轉換為向量成功，向量長度: {len(vector)}")

        # 儲存向量到 Qdrant
        point_id = str(uuid.uuid4())  # 使用時間戳和 UUID 作為唯一 ID
        qdrant_client.insert_point(point_id, vector, payload={"timestamp": timestamp, "image_path": image_path})

        # 回傳結果給前端
        emit('vector', {
            'status': 'success',
            'vector': vector
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