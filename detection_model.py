from ultralytics import YOLO
import cv2



class KidneyStonesDetectionModel:

    def __init__(self, model_path) -> None:
        
        self.model = YOLO(model=model_path)
        self.results = []


    def run_inference(self, image):
        self.results = self.model(image, conf=0.75)

    def draw_bboxes_on_image(self, image):

        modif_image = image.copy()

        for result in self.results:

            boxes = result.boxes

            for box in boxes:

                b = box.xyxy[0]

                x1, y1 = int(b[0]), int(b[1])
                x2, y2 = int(b[2]), int(b[3])
                
                cv2.rectangle(modif_image, [x1, y1], [x2, y2], (0, 255, 0), 1)

        return modif_image
    

if __name__=="__main__":

    model_path = "./ks_detection.pt"

    print("Loading model..")
    model = KidneyStonesDetectionModel(model_path=model_path)

    imgpath = "./sample_image.jpg"

    print("Reading image..")
    image = cv2.imread(imgpath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Running inference..")
    model.run_inference(image=image)

    print("Drawing results on image..")
    image_with_detections = model.draw_bboxes_on_image(image=image)

    cv2.imshow("results", image_with_detections)
    cv2.waitKey(0)