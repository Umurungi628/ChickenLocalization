if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO('yolov9n.pt')  # Adjust model path as needed
    model.train(data=r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\dataset.yaml',
                epochs=100,  # Adjust epochs as needed
                imgsz=640,  # Adjust image size as needed
                batch=16,   # Adjust batch size as needed
                workers=4)  # Adjust number of workers as needed
