# coding: utf-8
from numpy.ma.testutils import approx

from img import *
import random
import os
import time
import subprocess

press_coefficient = pow(3,0.5)  #1.63
swipe_x1 = 0; swipe_y1 = 0; swipe_x2 = 0; swipe_y2 = 0

SCREENSHOT_WAY = 1

# Performance optimization flags
ENABLE_DISPLAY = False  # Set to True to see detection visualization (slower)
FAST_MODE = True  # Use lower resolution for faster processing


# pull_screenshot is provided by img.py (in-memory adb capture)


def set_button_position(im):
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    w, h = im
    left = int(w / 2)
    top = int(1584 * (h / 1920.0))
    # Removed random offset for maximum precision
    left = int(random.uniform(left, left))
    top = int(random.uniform(top, top))
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top


def jump(distance):
    global press_coefficient
    set_button_position([1080,1920])
    press_time = distance * press_coefficient
    press_time = max(press_time, 100)   # 设置 200ms 是最小的按压时间
    press_time = int(press_time/50.0 + 0.5)*50  # 四舍五入到最接近的50ms
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    print(cmd)
    os.system(cmd)


def ensure_device_connected(retries=3, delay=1.0):
    """Check adb devices and ensure at least one device is connected."""
    for i in range(retries):
        try:
            out = subprocess.check_output(['adb', 'devices'], stderr=subprocess.STDOUT)
            out = out.decode('utf-8', errors='ignore')
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            # first line is header 'List of devices attached'
            dev_lines = [l for l in lines[1:] if l and 'device' in l]
            if len(dev_lines) > 0:
                return True
            print('No adb device found, retrying... ({}/{})'.format(i+1, retries))
        except Exception as e:
            print('adb check failed:', e)
        time.sleep(delay)
    return False


def main():
    if not ensure_device_connected():
        print('Error: no adb device detected. Please connect a device and ensure adb is in PATH.')
        return

    print('Starting auto jump... (FAST_MODE={}, DISPLAY={})'.format(FAST_MODE, ENABLE_DISPLAY))
    frame_count = 0
    total_time = 0

    while 1:
        loop_start = time.time()

        # capture image via adb into memory (img.pull_screenshot)
        im = pull_screenshot()
        if im is None:
            print('Error: pull_screenshot() failed. Retrying...')
            time.sleep(0.5)  # Wait before retry
            continue

        # Optional: downsample for faster processing
        if FAST_MODE and im.shape[0] > 1920:
            scale = 1920.0 / im.shape[0]
            im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # run detections
        self_pos = self_detect(im)
        if self_pos is None:
            print('Warning: self_detect failed')
            time.sleep(0.3)  # Wait before retry
            continue

        goal_pos = goal_detect(im, self_pos)
        if goal_pos is None:
            print('Warning: goal_detect failed')
            time.sleep(0.3)  # Wait before retry
            continue

        # Calculate distance and jump
        distance = pow(pow(goal_pos[0] - self_pos[0], 2) + pow(goal_pos[1] - self_pos[1], 2), 0.5)

        # Optional display before jump (for visual confirmation)
        if ENABLE_DISPLAY:
            try:
                im_disp = cv2.resize(im, (540, 960))
                cv2.imshow('test', im_disp)
                cv2.waitKey(500)  # Brief preview before jump
            except Exception as e:
                print('Display error:', e)

        jump(distance)

        # Wait for jump animation and game state to stabilize
        # Shorter distances need less time, longer distances need more time
        wait_time = max(1.5, min(2.0, distance / 300))  # 1-2 seconds based on distance
        time.sleep(wait_time)

        # Performance stats
        loop_time = time.time() - loop_start
        total_time += loop_time
        frame_count += 1
        if frame_count % 10 == 0:
            avg_time = total_time / frame_count
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print('Stats: avg={:.2f}s/frame, FPS={:.1f}, total={}'.format(avg_time, fps, frame_count))

main()
