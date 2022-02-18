from detector import *
import os

def main():

    videoPath = r'C:\Users\onury\Desktop\cv2\araba.jpg'

    configPath = r'C:\Users\onury\Desktop\cv2\RealTime-ObjectDetection-OpenCV\model_data\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    classesPath = r"C:\Users\onury\Desktop\cv2\RealTime-ObjectDetection-OpenCV\model_data\coco.names"
    modelPath = r'C:\Users\onury\Desktop\cv2\RealTime-ObjectDetection-OpenCV\model_data\frozen_inference_graph.pb'

    Detector(videoPath,configPath,modelPath,classesPath)
    #Detector.onVideo()

if __name__ == '__main__':
    main()
