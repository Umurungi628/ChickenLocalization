if __name__ == '__main__':
   from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
   model = YOLO("yolo12n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
   results = model.train(data=r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\dataset.yaml', epochs=100, imgsz=640)

