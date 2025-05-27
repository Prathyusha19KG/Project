def outdoor_detect():
    import cv2
    import pyttsx3

    engine=pyttsx3.init()
    thres = 0.45 # Threshold to detect object
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    cap.set(10,70)

    classNames= []
    classFile = 'coco.data'
    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'dataset.pb'

    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)


    font = cv2.FONT_HERSHEY_COMPLEX
    # org
    org = (20, 100)  # coloum-row
    org1 = (10, 50)  # 10,50
    org2 = (420, 370)
    org3 = (250, 440)
    org4 = (290, 470)
    # fontScale
    fontScale = 0.9
    fontScale1 = 1.0
    # Blue color in BGR
    color = (25, 255, 50)
    color1 = (255, 191, 0)  # sky blue deep
    color2 = (255, 255, 255)
    # Line thickness of 2 px
    thickness1 = 4
    thickness = 2



    while True:
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=0.45)


        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):

                object_name=(classNames[classId-1])
                object_id=[classId-1]


                if object_id == [0]:
                     print(" person detected")
                     cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                     cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2)
                     cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                     cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                     pyttsx3.speak("person detected")

                if object_id == [2]:
                    print(" Car detected ")
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    pyttsx3.speak("car detected")

                if object_id == [7]:
                    print(" truck detected  ")
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    pyttsx3.speak("truck detected")

        cv2.putText(img, 'ADHIYAMAAN ENGINEERING COLLEGE', (00, 370), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("", img)
        cv2.waitKey(1)

        k = cv2.waitKey(27) & 0xff
        if k == 27:
              break


