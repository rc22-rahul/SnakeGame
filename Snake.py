import cvzone
import cv2
import numpy as np
import math
import random
from cvzone.HandTrackingModule import HandDetector


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)


class SnakeGameClass:
    def __init__(self, Foodpath):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each points
        self.current_length = 0  # total length of the snake
        self.allowed_length = 150  # initial starting threshold allowed length of the snake
        self.previous_head = 0, 0  # previous position of heads of snake
        self.score =0
        self.img_food = cv2.imread(Foodpath, cv2.IMREAD_UNCHANGED)
        self.w_food, self.h_food, _ = self.img_food.shape
        self.food_points = 0, 0
        self.random_food_location()
        self.game_over = False

    def random_food_location(self):
        self.food_points = random.randint(100, 1000), random.randint(100,600)

    def update(self,img_main ,current_head):
        if self.game_over:
            cvzone.putTextRect(img_main, 'Game Over ', [300, 400], scale = 7, thickness= 5, offset= 20)
            cvzone.putTextRect(img_main, f"Your Score {self.score}", [300, 550], scale =7 , thickness= 5, offset= 20)
        else:
            px, py = self.previous_head  # previous x and previous y
            cx, cy = current_head  # current x and current y

            distance = math.hypot(cx-px, cy-py)   # difference between previous and current head or points

            self.points.append([cx, cy])
            self.lengths.append(distance)   # appending the distance
            self.current_length += distance   # updating the current length of snake
            self.previous_head = cx, cy    # updating the previous head

            # length reduction
            if self.current_length > self.allowed_length:
                for i, length in enumerate(self.lengths):
                    self.current_length -= length
                    self.points.pop(i)
                    self.lengths.pop(i)
                    if self.current_length < self.allowed_length:
                        break

            # check if snake ate the food
            rx, ry = self.food_points
            if rx - self.w_food//2 < cx < rx + self.w_food//2 and ry - self.h_food//2 < cy < ry + self.h_food//2:
                self.random_food_location()
                self.allowed_length += 50
                self.score += 1
                print(self.score)
            # pass

            # now drawing the snake
            for i, point in enumerate(self.points):
                if i != 0:
                    cv2.line(img_main, self.points[i-1], self.points[i], (0, 0, 255), 15)
                cv2.circle(img_main, self.points[-1], 25, (200, 0, 200), cv2.FILLED)

            # check for the collision by making a polygon and checking the distance between them
            pts = np.array(self.points[:-3], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img_main, [pts], False, (0, 200, 0), 3)
            collision_distance = cv2.pointPolygonTest(pts, (cx, cy), True)
            # print(collision_distance)

            if -0.5 <= collision_distance <= 0.5:
                print('Hit')
                self.game_over = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each points
                self.current_length = 0  # total length of the snake
                self.allowed_length = 150  # initial starting threshold allowed length of the snake
                self.previous_head = 0, 0  # previous position of heads of snake
                self.food_points = 0, 0
                # self.score=0
                self.random_food_location()
            # Draw Food
            # food image must be transparent of main background image cvzone has this feature
            img_main = cvzone.overlayPNG(img_main, self.img_food, (rx-self.w_food//2, ry-self.h_food//2))
            cvzone.putTextRect(img_main, f"Score {self.score}", [30, 80], scale=3, thickness=3, offset=10)


        return img_main


game = SnakeGameClass('Donut.png')


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)    # here 1 means vertical flip , 0 will mean horizontal flip
    hands, img = detector.findHands(img, flipType=False) # this flip type also solve the problem of proper hand movement
    # print(hands)

    # as max hands is 1 we will get the desired one only
    # the hand is totally divided into 21 points each point representing one position , so index finger tip is
    # accessible at point 8
    if hands:
        # print((hands[0]['lmList']))
        lmlist = hands[0]['lmList']  # lmlist is landmark list and data is inside lmList
        pointIndex = lmlist[8][:2]   # index 8 will give the index finger x,y,z but we want only x and y so indexing :2
        # cv2.circle(img, pointIndex, 25, (200, 0, 200), cv2.FILLED) # drawing a cirle at index finger
        img = game.update(img, pointIndex)
    cv2.imshow('Video', img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.score =0
        game.game_over = False