import cv2
import math
import numpy as np
import mediapipe as mp
import time

pTime = 0
NUM_FACE = 1

def init_facemesh():
  mpDraw = mp.solutions.drawing_utils
  mpFaceMesh = mp.solutions.face_mesh
  faceMesh = mpFaceMesh.FaceMesh(max_num_faces=NUM_FACE)
  faceMesh._num_face_coordinates = 468
  drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)
  return faceMesh, mpDraw, drawSpec, mpFaceMesh

def get_landmarks(img):
  image = img.copy()
  imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  faceMesh, mpDraw, drawSpec, mpFaceMesh = init_facemesh()
  results = faceMesh.process(imgRGB)
  land_list = []
  if results.multi_face_landmarks:
    for faceLms in results.multi_face_landmarks:
      mpDraw.draw_landmarks(image, faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
      for id,lm in enumerate(faceLms.landmark):
        ih, iw, ic = image.shape
        x, y = int(lm.x*iw), int(lm.y*ih)
        land_list.append([x, y])
  
  faceMesh.close()
  return land_list


def center_with_height_face(img):
  image = img.copy()
  imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  faceMesh, mpDraw, drawSpec, mpFaceMesh = init_facemesh()
  results = faceMesh.process(imgRGB)
  if results.multi_face_landmarks:
    land_list_x = []
    land_list_y = []
    for faceLms in results.multi_face_landmarks:
      mpDraw.draw_landmarks(image, faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
      for id,lm in enumerate(faceLms.landmark):
        ih, iw, ic = image.shape
        x, y = int(lm.x*iw), int(lm.y*ih)
        land_list_x.append(x)
        land_list_y.append(y)
        # uncomment the below line to see the 468 facial landmark
        if id == 6:
          center = (x,y)
        # cv2.putText(img, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
  w = max(land_list_x) - min(land_list_x)
  h = max(land_list_y) - min(land_list_y)
  faceMesh.close()
  return list(center), w, h


def plot(img1, print_on, color="b"):
  image = img1.copy()
  imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  faceMesh, mpDraw, drawSpec, mpFaceMesh = init_facemesh()
  results = faceMesh.process(imgRGB)
  if results.multi_face_landmarks:
    x_list = []
    y_list = []
    for faceLms in results.multi_face_landmarks:
      mpDraw.draw_landmarks(image, faceLms,mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
      for id,lm in enumerate(faceLms.landmark):
        ih, iw, ic = image.shape
        x, y = int(lm.x*iw), int(lm.y*ih)
        x_list.append(x)
        y_list.append(y)
        # uncomment the below line to see the 468 facial landmark
        if id == 6:
          center = (x,y)
          cv2.circle(print_on, (x, y), 1, (0,255,0), 5)
        
        else:
          if color == "r":
            cv2.circle(print_on, (x, y), 1, (0,0,255), 5)
          else:
            cv2.circle(print_on, (x, y), 1, (255,0,0), 5)

  faceMesh.close()
  return print_on


def center_margin(img1, img2):
  # fetch center lanmark and width and height of img1 , img2
  c1, _, _ = center_with_height_face(img1)
  c2, _, _ = center_with_height_face(img2)
  
  # clac diff center landmark new img1 and img2
  diff_c2_new_c = (c2[0] - c1[0], c2[1] - c1[1])
  # diff_c2_new_c = [(i - j) for i , j in zip(c2,c1)]
  # calc diff shape new img1
  trans = np.float32([[1, 0, diff_c2_new_c[0]],
                      [0, 1, diff_c2_new_c[1]]])
  new_img = cv2.warpAffine(img1, trans, (img2.shape[1], img2.shape[0]))
  return new_img   


def resize(img1, img2, b= 0.1):
  h, w, c = img1.shape
  _, w1, h1 = center_with_height_face(img1)
  _, w2, h2 = center_with_height_face(img2)
  # calc diff width and height img2 and img1
  diff_w = (w2 - w1) * 2
  diff_h = (h2 - h1) * 2
  # calc and resize new size of img1 with respect of diffs
  new_w, new_h = w + diff_w, h + diff_h 
  # img1 = cv2.resize(img1, (int(new_w), int(new_h)))
  # img1 = cv2.resize(img1, (0,0), fx=round(new_w / w, 1) + b, fy=round(new_w / h, 1)+ b)
  img1 = cv2.resize(img1, (0,0), fx= 1 + b, fy= 1 + b)
  new_img = center_margin(img1, img2)
  return new_img


def image_verificaion(img1, img2):
  land_1 = get_landmarks(img1)
  land_2 = get_landmarks(img2)
  assert len(land_1) == len(land_1), "can not find any face in image or one image hase not close shot of face"
  score = np.linalg.norm(np.array(land_1) - np.array(land_2))
  return score


def face_crop(img1_p:str, img2_p:str):
  """
  img1_p: Source image
  img2_p: Target image
  change face coordinate and resize Source image base on Target image then rewrite image Source.
  """
  img1 = cv2.imread(img1_p) 
  img2 = cv2.imread(img2_p)
  assert img1.shape == img2.shape, "both image should have same size"
    
  flg = True
  bias = 0.001
  img1 = center_margin(img1, img2)
  loss = [image_verificaion(img1, img2)]
  target_score = 700
  if loss[-1] > target_score:
    while flg:
      new_img = resize(img1, img2, bias)
      sc = image_verificaion(new_img, img2)
      print("verification score", sc)
      loss.append(sc - target_score)
      if loss[-1] > loss[-2]:
        bias -= 0.1
      else:
        bias += 0.1
      if (sc < target_score) or (loss[-1] < 1):
        flg = False     
    cv2.imwrite(img1_p, new_img)
    print(f"image {img1_p} changed")


  