import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

cap = cv2.VideoCapture(0)

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img,(700,500))               # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        print(mp_pose)
        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose

# cap = cv2.VideoCapture(0)
# bg = cv2.imread('D:/code/pyskl/demo/windows_bg.jpg')   # 載入 windows 經典背景

# with mp_pose.Pose(
#     min_detection_confidence=0.5,
#     enable_segmentation=True,       # 額外設定 enable_segmentation 參數
#     min_tracking_confidence=0.5) as pose:

#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#     while True:
#         ret, img = cap.read()
#         if not ret:
#             print("Cannot receive frame")
#             break
#         img = cv2.resize(img,(520,300))
#         img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = pose.process(img2)
#         try:
#             # 使用 try 避免抓不到姿勢時發生錯誤
#             condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
#             # 如果滿足模型判斷條件 ( 表示要換成背景 )，回傳 True
#             img = np.where(condition, img, bg)
#             print(bg)
#             # 將主體與背景合成，如果滿足背景條件，就更換為 bg 的像素，不然維持原本的 img 的像素
#         except:
#             pass
#         mp_drawing.draw_landmarks(
#             img,
#             results.pose_landmarks,
#             mp_pose.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

#         cv2.imshow('oxxostudio', img)
#         if cv2.waitKey(5) == ord('q'):
#             break     # 按下 q 鍵停止
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp

# mp_drawing = mp.solutions.drawing_utils         # mediapipe 繪圖方法
# mp_drawing_styles = mp.solutions.drawing_styles # mediapipe 繪圖樣式
# mp_holistic = mp.solutions.holistic             # mediapipe 全身偵測方法

# cap = cv2.VideoCapture(0)

# # mediapipe 啟用偵測全身
# with mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as holistic:

#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#     while True:
#         ret, img = cap.read()
#         if not ret:
#             print("Cannot receive frame")
#             break
#         img = cv2.resize(img,(520,300))
#         img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
#         results = holistic.process(img2)              # 開始偵測全身
#         # 面部偵測，繪製臉部網格
#         mp_drawing.draw_landmarks(
#             img,
#             results.face_landmarks,
#             mp_holistic.FACEMESH_CONTOURS,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=mp_drawing_styles
#             .get_default_face_mesh_contours_style())
#         # 身體偵測，繪製身體骨架
#         mp_drawing.draw_landmarks(
#             img,
#             results.pose_landmarks,
#             mp_holistic.POSE_CONNECTIONS,
#             landmark_drawing_spec=mp_drawing_styles
#             .get_default_pose_landmarks_style())

#         cv2.imshow('oxxostudio', img)
#         if cv2.waitKey(5) == ord('q'):
#             break    # 按下 q 鍵停止
# cap.release()
# cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp
# import math

# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands

# # 根據兩點的座標，計算角度
# def vector_2d_angle(v1, v2):
#     v1_x = v1[0]
#     v1_y = v1[1]
#     v2_x = v2[0]
#     v2_y = v2[1]
#     try:
#         angle_= math.degrees(math.acos((v1_x*v2_x+v1_y*v2_y)/(((v1_x**2+v1_y**2)**0.5)*((v2_x**2+v2_y**2)**0.5))))
#     except:
#         angle_ = 180
#     return angle_

# # 根據傳入的 21 個節點座標，得到該手指的角度
# def hand_angle(hand_):
#     angle_list = []
#     # thumb 大拇指角度
#     angle_ = vector_2d_angle(
#             ((int(hand_[0][0])- int(hand_[2][0])),(int(hand_[0][1])-int(hand_[2][1]))),
#             ((int(hand_[3][0])- int(hand_[4][0])),(int(hand_[3][1])- int(hand_[4][1])))
#         )
#     angle_list.append(angle_)
#     # index 食指角度
#     angle_ = vector_2d_angle(
#             ((int(hand_[0][0])-int(hand_[6][0])),(int(hand_[0][1])- int(hand_[6][1]))),
#             ((int(hand_[7][0])- int(hand_[8][0])),(int(hand_[7][1])- int(hand_[8][1])))
#         )
#     angle_list.append(angle_)
#     # middle 中指角度
#     angle_ = vector_2d_angle(
#             ((int(hand_[0][0])- int(hand_[10][0])),(int(hand_[0][1])- int(hand_[10][1]))),
#             ((int(hand_[11][0])- int(hand_[12][0])),(int(hand_[11][1])- int(hand_[12][1])))
#         )
#     angle_list.append(angle_)
#     # ring 無名指角度
#     angle_ = vector_2d_angle(
#         ((int(hand_[0][0])- int(hand_[14][0])),(int(hand_[0][1])- int(hand_[14][1]))),
#         ((int(hand_[15][0])- int(hand_[16][0])),(int(hand_[15][1])- int(hand_[16][1])))
#         )
#     angle_list.append(angle_)
#     # pink 小拇指角度
#     angle_ = vector_2d_angle(
#             ((int(hand_[0][0])- int(hand_[18][0])),(int(hand_[0][1])- int(hand_[18][1]))),
#             ((int(hand_[19][0])- int(hand_[20][0])),(int(hand_[19][1])- int(hand_[20][1])))
#         )
#     angle_list.append(angle_)
#     return angle_list

# # 根據手指角度的串列內容，返回對應的手勢名稱
# def hand_pos(finger_angle):
#     f1 = finger_angle[0]   # 大拇指角度
#     f2 = finger_angle[1]   # 食指角度
#     f3 = finger_angle[2]   # 中指角度
#     f4 = finger_angle[3]   # 無名指角度
#     f5 = finger_angle[4]   # 小拇指角度

#     # 小於 50 表示手指伸直，大於等於 50 表示手指捲縮
#     if f1<50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
#         return 'good'
#     elif f1>=50 and f2>=50 and f3<50 and f4>=50 and f5>=50:
#         return 'no!!!'
#     elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5<50:
#         return 'ROCK!'
#     elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5>=50:
#         return '0'
#     elif f1>=50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
#         return 'pink'
#     elif f1>=50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
#         return '1'
#     elif f1>=50 and f2<50 and f3<50 and f4>=50 and f5>=50:
#         return '2'
#     elif f1>=50 and f2>=50 and f3<50 and f4<50 and f5<50:
#         return 'ok'
#     elif f1<50 and f2>=50 and f3<50 and f4<50 and f5<50:
#         return 'ok'
#     elif f1>=50 and f2<50 and f3<50 and f4<50 and f5>50:
#         return '3'
#     elif f1>=50 and f2<50 and f3<50 and f4<50 and f5<50:
#         return '4'
#     elif f1<50 and f2<50 and f3<50 and f4<50 and f5<50:
#         return '5'
#     elif f1<50 and f2>=50 and f3>=50 and f4>=50 and f5<50:
#         return '6'
#     elif f1<50 and f2<50 and f3>=50 and f4>=50 and f5>=50:
#         return '7'
#     elif f1<50 and f2<50 and f3<50 and f4>=50 and f5>=50:
#         return '8'
#     elif f1<50 and f2<50 and f3<50 and f4<50 and f5>=50:
#         return '9'
#     else:
#         return ''

# cap = cv2.VideoCapture(0)            # 讀取攝影機
# fontFace = cv2.FONT_HERSHEY_SIMPLEX  # 印出文字的字型
# lineType = cv2.LINE_AA               # 印出文字的邊框

# # mediapipe 啟用偵測手掌
# with mp_hands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:

#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()
#     w, h = 540, 310                                  # 影像尺寸
#     while True:
#         ret, img = cap.read()
#         img = cv2.resize(img, (w,h))                 # 縮小尺寸，加快處理效率
#         if not ret:
#             print("Cannot receive frame")
#             break
#         img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 轉換成 RGB 色彩
#         results = hands.process(img2)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 finger_points = []                   # 記錄手指節點座標的串列
#                 fx = []                              # 記錄所有 x 座標的串列
#                 fy = []                              # 記錄所有 y 座標的串列
#                 for i in hand_landmarks.landmark:
#                     # 將 21 個節點換算成座標，記錄到 finger_points
#                     x = i.x*w                        # 計算 x 座標
#                     y = i.y*h                        # 計算 y 座標
#                     finger_points.append((x,y))
#                     fx.append(int(x))                # 記錄 x 座標
#                     fy.append(int(y))                # 記錄 y 座標
#                 if finger_points:
#                     finger_angle = hand_angle(finger_points) # 計算手指角度，回傳長度為 5 的串列
#                     #print(finger_angle)             # 印出角度 ( 有需要就開啟註解 )
#                     text = hand_pos(finger_angle)    # 取得手勢所回傳的內容
#                     if text == 'no!!!':
#                         x_max = max(fx)              # 如果是比中指，取出 x 座標最大值
#                         y_max = max(fy)              # 如果是比中指，取出 y 座標最大值
#                         x_min = min(fx) - 10         # 如果是比中指，取出 x 座標最小值
#                         y_min = min(fy) - 10         # 如果是比中指，取出 y 座標最小值
#                         if x_max > w: x_max = w      # 如果最大值超過邊界，將最大值等於邊界
#                         if y_max > h: y_max = h      # 如果最大值超過邊界，將最大值等於邊界
#                         if x_min < 0: x_min = 0      # 如果最小值超過邊界，將最小值等於邊界
#                         if y_min < 0: y_min = 0      # 如果最小值超過邊界，將最小值等於邊界
#                         mosaic_w = x_max - x_min     # 計算四邊形的寬
#                         mosaic_h = y_max - y_min     # 計算四邊形的高
#                         mosaic = img[y_min:y_max, x_min:x_max]     # 取出四邊形區域
#                         mosaic = cv2.resize(mosaic, (8,8), interpolation=cv2.INTER_LINEAR)  # 根據縮小尺寸縮小
#                         mosaic = cv2.resize(mosaic, (mosaic_w,mosaic_h), interpolation=cv2.INTER_NEAREST) # 放大到原本的大小
#                         img[y_min:y_max, x_min:x_max] = mosaic    # 馬賽克區域
#                     else:
#                         cv2.putText(img, text, (30,120), fontFace, 5, (255,255,255), 10, lineType) # 印出文字

#         cv2.imshow('oxxostudio', img)
#         if cv2.waitKey(5) == ord('q'):
#             break
# cap.release()
# cv2.destroyAllWindows()

