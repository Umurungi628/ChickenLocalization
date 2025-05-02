from ultralytics import YOLO

def train_model():
    # Build a YOLO model from pretrained weights
    model = YOLO("yolov10s.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the dataset specified in the 'dataset.yaml' file for 100 epochs
    results = model.train(data=r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\dataset.yaml', epochs=100, imgsz=640)

if __name__ == "__main__":
    train_model()
