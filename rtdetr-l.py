from ultralytics import RTDETR  # Move the import outside the `if __name__ == "__main__":` block

if __name__ == "__main__":
    # Load a COCO-pretrained RT-DETR-l model
    model = RTDETR("rtdetr-l.pt")

    # Display model information (optional)
    model.info()

    # Train the model on the COCO8 example dataset for 100 epochs
    results = model.train(data=r'C:\Users\User\PycharmProjects\ChickenLocalization\ChickenLocalization-main\dataset.yaml', epochs=100, imgsz=640,batch=8)



