import cv2
import os
import serial

def take_picture(
    cam_ID: int,
    width: int, height: int,
    name_ID: int,
    crop: bool = False):
    '''Takes a picture, crops it, and saves it.'''

    throw_away_frames: int = 20
    path: str = '/home/michael/Cax_Box_Pics/temp'

    cap = cv2.VideoCapture(cam_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if cap.isOpened():
        for i in range(throw_away_frames):
            _,_ = cap.read()
        _, frame = cap.read()
        if crop:
            crop_img = frame[0:-100, 150:-75]
            cv2.imwrite(
                os.path.join(path , "litterbox_cropped_" + str(name_ID) + ".png"),
                crop_img)
        else:
            cv2.imwrite(
                os.path.join(path , "litterbox_" + str(name_ID) + ".png"),
                frame)
        cap.release()
        cv2.destroyAllWindows()
        return frame

def toggle_fan(fan_state: int) -> None:
    '''Sends a Serial command to the teensy.'''
    s: serial.Serial = serial.Serial('/dev/ttyACM0', 9600)

    if fan_state == 0:
        s.write(b'fan_off\n')
    elif fan_state == 1:
        s.write(b'fan_on\n')
    s.close()
