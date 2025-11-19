import cv2
import numpy as np
import subprocess
import sys
from open_close_op import *


def adjust(x):
  temp=x.copy()
  #print(temp)
  for i in range(4):
    cnt1=0;cnt2=0
    for j in range(4):
      if x[i][0]<x[j][0]:
        cnt1=cnt1+1
      if x[i][1]<x[j][1]:
        cnt2=cnt2+1
    #print cnt1,cnt2
    if cnt1>=2 and cnt2>=2:
      temp[0]=x[i]
    if cnt1<2 and cnt2>=2:
      temp[1]=x[i]
    if cnt1>=2 and cnt2<2:
      temp[2]=x[i]
    if cnt1<2 and cnt2<2:
      temp[3]=x[i]
  #print temp
  return temp


def solve_function(lines):
    pos = []
    for i in range(len(lines)):
        for j in range(i+1,len(lines)):
            k1 = (lines[i][3]-lines[i][1])/(lines[i][2]-lines[i][0])
            k2 = (lines[j][3] - lines[j][1]) / (lines[j][2] - lines[j][0])
            try:
                x = (lines[j][1]-k2*lines[j][0]+k1*lines[i][0]-lines[i][1])/(k1-k2)
                y = k1*(x-lines[i][0])+lines[i][1]
                if x>1920 or x<0 or y>1920 or y<0:
                    continue
                pos.append([x, y])
            except:
                pass
    for i in range(len(pos)):
        for j in range(len(pos)-1,i,-1):
            if pow(pos[i][0]-pos[j][0],2)+pow(pos[i][0]-pos[j][0],2)<400:
                pos[i]=[(pos[i][0]+pos[j][0])/2,(pos[i][1]+pos[j][1])/2]
                del pos[j]
                pass
    for i in range(len(pos)):
        pos[i][0] = int(pos[i][0])
        pos[i][1] = int(pos[i][1])
    return pos


def screen_detect(img):
    #img=cv2.resize(img,(540,960))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, blackwhite = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)

    cv2.imwrite("raw.jpg",img)

    blackwhite = open_op_large(blackwhite)
    blackwhite = close_op_large(blackwhite)

    edges = cv2.Canny(blackwhite, 50, 200)
   #cv2.imshow("edes", edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 10, minLineLength=300, maxLineGap=30)
    lines1 = lines[:, 0, :]
    points=solve_function(lines1)
    print(points)

    points=np.array(points)
    pts2 = np.float32([[0, 0], [540, 0], [0, 960], [540, 960]])
    box = adjust(points)
    box = box.astype('float32')
    M = cv2.getPerspectiveTransform(box, pts2)
    screen = cv2.warpPerspective(img, M, (540, 960))
    # cv2.imshow('screen',screen)  # Disabled for speed

    print(lines1)
    for x1, y1, x2, y2 in lines1[:]:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    for point in points:
        cv2.circle(img,tuple(point),4,(0,0,255),thickness=5,lineType=1)

    cv2.imwrite("edge.jpg", edges)
    cv2.imwrite("screen.jpg", screen)
    cv2.imwrite("dealed.jpg",img)

    #cv2.imshow("img",img)
    #cv2.imshow("test",blackwhite)
    #cv2.waitKey(0)
    return screen


def pull_screenshot():
    """Capture the device screen via adb and return an OpenCV image (BGR).

    Uses raw format for maximum speed (no PNG compression overhead).
    Falls back to PNG format if raw fails.
    """
    try:
        # Method 1: Raw format (fastest - no compression, ~50-70% faster than PNG)
        try:
            proc = subprocess.Popen(['adb', 'exec-out', 'screencap'],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw_data = proc.stdout.read()

            if len(raw_data) > 100:
                # Parse raw screencap header (first 12-16 bytes contain width, height, format)
                # Format: width(4) height(4) format(4)
                import struct
                if len(raw_data) >= 12:
                    w, h, fmt = struct.unpack('<III', raw_data[:12])
                    # RGBA format (most common)
                    if fmt == 1 and w > 0 and h > 0 and w < 10000 and h < 10000:
                        pixel_data = raw_data[12:]  # Skip header
                        expected_size = w * h * 4
                        if len(pixel_data) >= expected_size:
                            # Create numpy array from raw RGBA data
                            img = np.frombuffer(pixel_data[:expected_size], dtype=np.uint8)
                            img = img.reshape((h, w, 4))
                            # Convert RGBA to BGR
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                            return img
        except Exception as e:
            pass  # Fall back to PNG method

        # Method 2: PNG format (fallback - slower but more compatible)
        proc = subprocess.Popen(['adb', 'exec-out', 'screencap', '-p'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        screenshot = proc.stdout.read()

        # Normalize CRLF for Windows
        if sys.platform == 'win32' and isinstance(screenshot, bytes):
            screenshot = screenshot.replace(b'\r\n', b'\n')

        if len(screenshot) > 100:
            arr = np.frombuffer(screenshot, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img

        return None

    except Exception as e:
        print('pull_screenshot error:', e)
        return None


def self_detect(img):
    region_upper=int(img.shape[0]*0.3)
    region_lower=int(img.shape[0]*0.7)
    region=img[region_upper:region_lower]

    hsv_img=cv2.cvtColor(region,cv2.COLOR_BGR2HSV)
    color_lower=np.int32([105,25,45])
    color_upper=np.int32([135,125,130])

    color_mask = cv2.inRange(hsv_img, color_lower, color_upper)

    # Robust findContours across OpenCV versions
    cnts = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1] if len(cnts) >= 2 else []

    # Filter out invalid contours by safely computing area
    valid_contours = []
    for c in contours:
        try:
            if c is None:
                continue
            arr = np.asarray(c)
            if arr.size == 0:
                continue
            # Ensure contour has shape (N,1,2) or (N,2)
            if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
                pass
            elif arr.ndim == 2 and arr.shape[1] == 2:
                arr = arr.reshape((-1,1,2))
            else:
                continue
            # Convert to int32 which OpenCV expects for contour functions
            arr = arr.astype(np.int32)
            area = cv2.contourArea(arr)
            if area and area > 10:  # ignore tiny noise
                valid_contours.append(arr)
        except Exception:
            continue

    if not valid_contours:
        return None

    try:
        max_contour = max(valid_contours, key=cv2.contourArea)
    except Exception as e:
        print('self_detect: failed to get max contour:', e)
        return None
    max_contour = cv2.convexHull(max_contour)
    rect = cv2.boundingRect(max_contour)
    x, y, w, h = rect
    center_coord=(x+int(w/2),y+h+region_upper - 20)
    cv2.circle(img, center_coord, 5, (0,255,0), -1)
    return center_coord


def goal_detect(img,body_position):
    region_upper=int(img.shape[0]*0.3)
    region_lower=int(img.shape[0]*0.6)
    #img = cv2.GaussianBlur(img, (11,11), 0)
    #img = cv2.blur(img,(11,11))
    if body_position[0]<(img.shape[1]/2.0):
        region_left=body_position[0]+30
        region_right=img.shape[1]-30
    else:
        region_left=30
        region_right=body_position[0]-30

    region = img[region_upper:region_lower, region_left:region_right]

    edge_list=[0,0,0,0]
    for i in range(3):
        region_gray=cv2.cvtColor(region,cv2.COLOR_BGR2HSV)[:,:,i]
        #region_gray=cv2.equalizeHist(region_gray)
        edge_list[i]=cv2.Canny(region_gray,100,160)

    region_gray=cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
    #region_gray = cv2.equalizeHist(region_gray)
    #egion_gray = cv2.GaussianBlur(region_gray, (5, 5), 0)
    #cv2.imshow("gray",region_gray)
    edge_list[3] = cv2.Canny(region_gray, 100, 160,apertureSize=5)

    edge_list[1]=np.bitwise_or(edge_list[0],edge_list[1])
    edge_list[2]=np.bitwise_or(edge_list[2],edge_list[1])
    edge_final=np.bitwise_or(edge_list[3],edge_list[2])

    # cv2.imshow('edge',edge_final)  # Disabled for speed

    # Replace contours extraction in goal_detect as well with robust pattern
    cnts = cv2.findContours(edge_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1] if len(cnts) >= 2 else []

    # Filter valid contours (non-empty)
    if not contours:
        print('goal_detect: no contours found')
        return None

    y=99999
    raw_x = raw_y = None
    x = None
    for contour in contours:
        try:
            if contour is None:
                continue
            arr = np.asarray(contour)
            if arr.size == 0:
                continue
            # normalize shape
            if arr.ndim == 3 and arr.shape[1] == 1 and arr.shape[2] == 2:
                pass
            elif arr.ndim == 2 and arr.shape[1] == 2:
                arr = arr.reshape((-1,1,2))
            else:
                continue
            arr = arr.astype(np.int32)
            sorted_top=sorted(arr, key=lambda p: p[0][1])
        except Exception:
            continue
        if len(sorted_top) == 0:
            continue
        if sorted_top[0][0][1]<y:
            raw_x = x = int(sorted_top[0][0][0])
            raw_y = y = int(sorted_top[0][0][1])

    # If we never set x/raw_x then something went wrong
    if x is None or raw_x is None:
        print('goal_detect: failed to locate top-most point in contours')
        return None

    # Use safe fallbacks for debug printing
    px = x if x is not None else 0
    py = y if y is not None else 0
    print((int(px + region_left), int(py + region_upper)))

    try:
        mask = np.zeros((region_lower-region_upper+ 2, region_right-region_left + 2), np.uint8)
        mask=cv2.floodFill(region, mask, (raw_x, raw_y+16), [255,25,255])[2]
    except Exception as e:
        print('goal_detect: floodFill failed:', e)
        return None
    cv2.circle(img, (int(x + region_left), int(y + region_upper)), 5, (255, 0, 5), -1)
    #cv2.imshow("region",region)

    M = cv2.moments(mask)
    if M.get("m00", 0) == 0:
        print('goal_detect: zero moment, cannot compute centroid')
        return None
    x = int(M["m10"] / M["m00"])
    y = int(M["m01"] / M["m00"])

    if y<raw_y or abs(x-raw_x)>40:
        x=raw_x;y=raw_y
        y += region_upper
        x += region_left
        y = (-abs(x-body_position[0])/pow(3,0.5)+body_position[1])
        print("TESTTEST")

    #cv2.imshow("test",mask)
    #cv2.imshow("edge", edge_final)
    else:
        y += region_upper
        x += region_left

    #y = (-abs(x-body_position[0])/pow(3,0.5)+body_position[1])
    cv2.circle(img, (int(x),int(y)), 5, (0,0,255), -1)
    # cv2.imshow('dealed',img)  # Disabled for speed
    return [x, y]




if __name__ == '__main__':
    #pull_screenshot()
    img=cv2.imread('autojump.png')
    img=screen_detect(img)
    #img=cv2.imread('TEST7.png')
    img=cv2.resize(img,(720,1280))
    body_position = self_detect(img)
    [x,y]=goal_detect(img,body_position)
    #cv2.imshow('raw',img)
    cv2.waitKey()