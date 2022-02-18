import cv2
import numpy as np


class Detector:

    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        
        self.net = cv2.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(350,350)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

        
        self.readClasses()
        self.onVideo()

    def readClasses(self):
        with open(self.classesPath,'r') as f:
            self.classesList = f.read().splitlines()
        self.classesList.insert(0,'__Background__')  #class 0. index Backgroun

        self.colorList = np.random.uniform(low=0,high=255, size=(len(self.classesList),3))

        print(self.classesList)

    def onVideo(self):
        cap = cv2.VideoCapture(self.videoPath)

        if(cap.isOpened()==False):
            print("Error")
            return
        
        
        (success,image) = cap.read()
        

        while success:
            
            classLabelIDs, confidences, bboxs = self.net.detect(image,confThreshold=0.5)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1,-1)[0])
            confidences = list(map(float,confidences))

            bboxIdx = cv2.dnn.NMSBoxes(bboxs,confidences,score_threshold=0.5,nms_threshold=0.2)

            if len(bboxs) != 0:
                for i in range(0,len(bboxIdx)):
                    bbox = bboxs[np.squeeze(bboxIdx[i])]
                    classConfidence = confidences[np.squeeze(bboxIdx[i])]
                    classLabelId = np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                    classLabel = self.classesList[classLabelId]
                    classColor = [int(c) for c in self.colorList[classLabelId]]

                    displayText = "{} : {:.2f}".format(classLabel,classConfidence)

                    x,y,w,h = bbox

                    cv2.rectangle(image,(x, y), (x+w, y+h), color=classColor,thickness=2)
                    cv2.putText(image,displayText,(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,classColor,2)

            
            
            cv2.imshow('result',image)
            

            key = cv2.waitKey(1) & 0xFF

            if key ==ord('q'):
                break

            (success,image) = cap.read()
        
        
        cv2.destroyAllWindows()