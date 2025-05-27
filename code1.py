import cv2

st=cv2.VideoCapture(0)
while 1:
    x,y=st.read(0)
    cv2.imshow("f",y)
    cv2.waitKey(1)