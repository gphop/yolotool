import os
from ultralytics import YOLO
current_dir = os.path.dirname(os.path.abspath(__file__))   #获取当前文件夹目录
wd = os.getcwd()  #获取当前工作空间路径
# ==================== 配置 ===================
epochs = 2000 # 训练轮次，因为会提前停止可以填多
imgsz = 640 # 照片尺寸，因为运算矩阵为正方形所有以最长边填
batch = 32  # 训练批次，一次训练32张，数值越大越耗性能但也越快
mosaic_sing = 1.0 # 不开启为0.0，打开改为1.0。训练增强参数，针对新元素添加时发现新元素占比较少，添加照片较少       使用此参数可以将一张图片增强分裂

# ================融合训练配置==================
Trained_model_path = "/home/gph/YOLOv11/run_v11/detect/train2/weights/best.pt" # 训练好的模型best文件路径
Inheritance_model = False  # 是否融合训练
# ============================================


#判断是否使用自己的模型训练
if Inheritance_model:  #使用自己的模型进行训练(融合训练)
    model = YOLO(Trained_model_path)
    results = model.train(data="voc_dataset_v11/user_coco.yaml", epochs=epochs, imgsz=imgsz, batch=batch, mosaic = mosaic_sing)
else: #使用yolo11n.pt使用官方预训练模型进行训练
    model = YOLO(f"{wd}/yolo11n.pt")
    results = model.train(data="voc_dataset_v11/user_coco.yaml", epochs=epochs, imgsz=imgsz, batch=batch, mosaic = mosaic_sing)
    

