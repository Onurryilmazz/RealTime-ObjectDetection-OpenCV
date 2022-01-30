from detector import *
import os

def main():

    videoPath = "test_videos/test.mp4"

    configPath = os.path.join("model_data","ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    classesPath = os.path.join("model_data","coco.names")
    modelPath = os.path.join("model_data","frozen_inference_graph.pb")

    Detector(videoPath,configPath,modelPath,classesPath)
    #Detector.onVideo()

if __name__ == '__main__':
    main()
