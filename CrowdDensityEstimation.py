import cv2
import numpy as np
from matplotlib import pyplot as plt
import linegraph as lg
import pandas as ps
import timer

people=[]
time=[]
counter=0
table = {'people':[],
        'time':[]
        }

runningTimer = False
# iteration = 0
# blac = np.zeros((720, 1080), dtype=np.uint8)
kernel1 = np.ones((2,2),np.float32)/4
kernel2 = np.ones((15,15),np.float32)/225

# Change video source here
VIDEO_SRC = 'crowd2.mp4'

cap = cv2.VideoCapture(VIDEO_SRC)

ret, back = cap.read()
# while(1):
#     cv2.imshow('back', back)
#     cv2.imwrite('_back.png', back)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     break
#
# backG = cv2.imread('_back.png')

input('result')
while(1):
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    # sub = cv2.subtract(backG, frame)
    imggray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(imggray, 9)
    ret, imgthresh = cv2.threshold(med, 100, 255, cv2.THRESH_BINARY_INV)
    opening = cv2.morphologyEx(imgthresh, cv2.MORPH_OPEN, kernel1)
    closing = cv2.dilate(opening, kernel1, iterations=1)
    # cv2.imshow('thresh', closing)
    count = (cv2.countNonZero(closing))/1700
    # print(int(count))

    remainder=counter%5
    if remainder==0:
        people.append(int(count))
        time.append(counter)
    counter=counter+1
    if counter == 100:
        break

    # hisdata = 224 - (int(count)*5)
    # histo.histo(blac, hisdata, iteration=iteration)
    # iteration = iteration+1

    h, w = frame.shape[:2]
    contours0, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]



    cv2.waitKey(200)
    print('\n')

    def update(levels):
        vis = np.zeros((h, w, 3), np.uint8)
        levels = levels - 3
        cv2.drawContours(vis, contours, (-1, 2)[levels <= 0], (128, 255, 255), 3, cv2.LINE_AA, hierarchy, abs(levels))
        cv2.imshow('contours', vis)
        cv2.waitKey(100)

    update(5)
    if timer.isTimerRunning() == -1:
        print("timer started")
        timer.startTimer(3)
    if timer.isTimerRunning() == 0:
        print("timer ended")
        timer.resetTimer()
        print("Timer reset")
    # print("------\nFrame Area")
    # for contour in contours:
    #     print(cv2.contourArea(contour))
    # if counter==200:
    #     break

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
table['people']=people
table['time']=time

df = ps.DataFrame(table, columns= ['people', 'time'])
print(df)
# export_csv = df.to_csv (r'export_dataframe1.csv', index = None, header=True)
# lg.linegraph('export_dataframe1.csv')
# cv2.waitKey(0)

def checkDistance():
    ret, frame = cap.read()
    # cv2.imshow('frame', frame)
    # sub = cv2.subtract(backG, frame)
    imggray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    med = cv2.medianBlur(imggray, 9)
    ret, imgthresh = cv2.threshold(med, 100, 255, cv2.THRESH_BINARY_INV)
    opening = cv2.morphologyEx(imgthresh, cv2.MORPH_OPEN, kernel1)
    closing = cv2.dilate(opening, kernel1, iterations=1)



