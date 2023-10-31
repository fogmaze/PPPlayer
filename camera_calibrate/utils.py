import numpy as np
from ctypes import *
from pupil_apriltags import Detector, Detection
import sys
import os
sys.path.append(os.getcwd())
import core.Equation3d as equ
import camera_calibrate.Calibration as calib
from core.Constants import *
from camera_reciever.CameraReceiver import CameraReceiver
import cv2
import csv
import time
import pickle
import multiprocessing as mp


def p(s, f ) :
    cap = cv2.VideoCapture(s)
    wri = cv2.VideoWriter(f, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640, 480))
    while True :
        ret, frame = cap.read()
        if ret :
            wri.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xff == ord('q') :
                break
    wri.release()


def rec() :
    p1 = mp.Process(target=p, args=(0, '0.mp4'))
    p2 = mp.Process(target=p, args=(1, '1.mp4'))
    p1.start()
    p2.start()
    p1.join()
    p2.join()




def takePicture_and():
    t = time.time()
    cap = CameraReceiver("172.20.10.2")
    #lprint(cap.get(cv2.CAP_PROP_FPS))
    a = False
    i = 0
    cap.connect()
    while True :
        ret, frame = cap.read()
        #print(1/(time.time() - t))
        t = time.time()
        if ret :
            cv2.imshow('frame', frame)
            key = cv2.waitKey(10) & 0xff

            if a :
                cv2.imwrite('pic0.jpg', frame)
                break

            if key == ord('w') :
                cv2.imwrite('D{}.jpg'.format(i), frame)
                i += 1


            if key == ord('q') :
                #sleep for one second
                #time.sleep(5)
                a = True
    cap.close()
    cv2.destroyAllWindows()

def takePicture():
    t = time.time()

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FPS, 60)
    print(cap.get(cv2.CAP_PROP_FPS))
    a = False
    i = 0
    while True :
        ret, frame = cap.read()
        #print(1/(time.time() - t))
        t = time.time()
        if ret :
            cv2.imshow('frame', frame)
            key = cv2.waitKey(10) & 0xff

            if a :
                cv2.imwrite('B.jpg', frame)
                break

            if key == ord('w') :
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pos = calculateCameraPosition(pickle.load(open('calibration1_old', 'rb')), g)
                print(pos)
                cv2.imwrite('A13{}.jpg'.format(i), frame)
                i += 1


            if key == ord('q') :
                #sleep for one second
                #time.sleep(5)
                a = True
    cap.release()
    cv2.destroyAllWindows()

def calculateCameraPosition(cameraMatrix:np.ndarray, frame_gray, tagSize=APRILTAG_SIZE) :
    try :
        detector = Detector()
        results = detector.detect(frame_gray, 
                                    estimate_tag_pose=True, 
                                    camera_params=(cameraMatrix[0][0],cameraMatrix[1][1],cameraMatrix[0][2],cameraMatrix[1][2]),
                                    tag_size=tagSize)
        if len(results) == 1:
            res:Detection = results[0]
            position = np.matmul(np.linalg.inv(res.pose_R), -res.pose_t)
            print("origin: ", position)
            p = equ.Point3d(position[0][0] + (tagSize/2) - 2.74/2, -position[2][0] - 1.525/2, -position[1][0] + (tagSize/2))
            print("after: ", p.to_str())
            return equ.Point3d(position[0][0] + (tagSize/2) - 2.74/2, -position[2][0] - 1.525/2, -position[1][0] + (tagSize/2))
        else:
            return None
    except Exception as e:
        print(e)
        return None
    

class _Matd(Structure):
    _fields_ = [
        ("nrows", c_int),
        ("ncols", c_int),
        ("data", c_double * 1),
    ]

    
class mat_d9(Structure):
    _fields_ = [
        ("nrows", c_uint),
        ("ncols", c_uint),
        ("data", c_double * 9)
    ]


def _ptr_to_array2d(datatype, ptr, rows, cols):
    array_type = (datatype * cols) * rows
    array_buf = array_type.from_address(addressof(ptr))
    return np.ctypeslib.as_array(array_buf, shape=(rows, cols))

def _matd_get_array(mat_ptr):
    return _ptr_to_array2d(
        c_double,
        mat_ptr.contents.data,
        int(mat_ptr.contents.nrows),
        int(mat_ptr.contents.ncols),
    )
    
class _ApriltagDetection(Structure):
    _fields_ = [
        ("family", c_void_p),
        ("id", c_int),
        ("hamming", c_int),
        ("decision_margin", c_float),
        ("H", POINTER(mat_d9)),
        ("c", c_double * 2),
        ("p", (c_double * 2) * 4),
    ]


class _ApriltagDetectionInfo(Structure):
    _fields_ = [
        ("det", POINTER(_ApriltagDetection)),
        ("tagsize", c_double),
        ("fx", c_double),
        ("fy", c_double),
        ("cx", c_double),
        ("cy", c_double),
    ]


class _ApriltagPose(Structure):
    """Wraps apriltag_pose C struct."""

    _fields_ = [("R", POINTER(_Matd)), ("t", POINTER(_Matd))]

def _np2doublep(np_array):
    res = ((c_double * 2) * 4)()

class TableInfo :
    homo:np.ndarray    = None
    corners:np.ndarray = None # (-1,1), (1,1), (1,-1), and (-1,-1))
    inmtx:np.ndarray   = None
    width              = 2.74
    height             = 1.525


def calculateCameraPosition_table(info:TableInfo) :
    libc = CDLL("build/libpose.so")
    libc.estimate.restype = c_double
    libc.estimate.argtypes = [c_void_p, (c_double * 2) * 4, c_double, c_double, c_double, c_double, c_double, c_double, c_void_p]
    #detector = Detector()
    #detector.libc.homography_to_pose.restype = c_void_p
    #detector.libc.homography_to_pose.argtypes = [c_void_p, c_double, c_double, c_double, c_double]
    #d = Detector().detect(cv2.imread("exp/718.jpg", cv2.IMREAD_GRAYSCALE), estimate_tag_pose=True, camera_params=(c[0][0],c[1][1],c[0][2],c[1][2]), tag_size=APRILTAG_SIZE)

    src = np.float32([[-info.w/2, info.h/2], [info.w/2, info.h/2], [info.w/2, -info.h/2], [-info.w/2, -info.h/2]])
    homo = cv2.findHomography(srcPoints=src, dstPoints=info.corners)[0]
    mat_d_ho = mat_d9()  # homography matrix
    mat_d_ho.nrows = 3
    mat_d_ho.ncols = 3
    mat_d_ho.data = (c_double * 9)(*homo.flatten())
    corners = ((c_double * 2) * 4)(*[(c_double * 2) (*a.tolist()) for a in info.corners])
    pose1 = _ApriltagPose()
    libc.estimate(cast(byref(mat_d_ho), c_void_p), corners, cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2], info.width, info.height, cast(byref(pose1), c_void_p))
    #pv_mat_d_o = detector.libc.homography_to_pose(cast(byref(mat_d_ho), c_void_p), -cameraMatrix[0][0], cameraMatrix[1][1], cameraMatrix[0][2], cameraMatrix[1][2])
    #dete = _ApriltagDetection(
            #family = None,
            #id = 0,
            #hamming = 0,
            #decision_margin = 0,
            #H = cast(byref(mat_d_ho), POINTER(mat_d9)),
            #c = (c_double * 2)(0, 0),
            #p = ((c_double * 2) * 4)((c_double * 2)(*d[0].corners[0]), (c_double * 2)(*d[0].corners[1]), (c_double * 2)(*d[0].corners[2]), (c_double * 2)(*d[0].corners[3]))
        #)
    #info = _ApriltagDetectionInfo(
        #det = cast(byref(dete), POINTER(_ApriltagDetection)),
        #tagsize = APRILTAG_SIZE,
        #fx = cameraMatrix[0][0],
        #fy = cameraMatrix[1][1],
        #cx = cameraMatrix[0][2],
        #cy = cameraMatrix[1][2]
    #)
    #pose2 = _ApriltagPose()
    #err = detector.libc.estimate_tag_pose(byref(info), byref(pose2))

    R = _matd_get_array(pose1.R)
    t = _matd_get_array(pose1.t)

    return np.matmul(np.linalg.inv(R), t)
    return r, t
    pass

def getCameraPosition_realTime(cameraMatrix) :
    cap = cv2.VideoCapture(0)
    while True :
        ret, frame = cap.read()
        time.sleep(1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        res = calculateCameraPosition(cameraMatrix, image)

        if res is not None :
            print(res.to_str())
        
def getBallPixelSize(distance, cameraMatrix) :
    BALL_REAL_SIZE = 0.038
    return BALL_REAL_SIZE * cameraMatrix[0][0] / distance


def runExerment() :
    cameraMatrix = pickle.load(open('calibration1_old', 'rb'))
    with open('experiment_AprilTag/data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for c in ('A', 'B', 'C', 'D') :
            res_x = []
            res_y = []
            res_z = []
            for i in range(10) :
                img = cv2.imread('experiment_AprilTag/{}/{}{}.jpg'.format(c, c, i), cv2.IMREAD_GRAYSCALE)
                pos = calculateCameraPosition(cameraMatrix, img)
                res_x.append(pos.x)
                res_y.append(pos.y)
                res_z.append(pos.z)
                print(pos.to_str())
            writer.writerow(res_x)
            writer.writerow(res_y)
            writer.writerow(res_z)
            

_x, _y = 0, 0
def on_mouse_move(event, x, y, flags, param):
    global _x, _y
    if event == cv2.EVENT_MOUSEMOVE:
        _x = x
        _y = y

_table_mouse_state = "up"
_table_points = []
_table_scale = 5
_table_scale_base_pos = (0, 0)
_table_mouse_now_pos = (0, 0)

def _setup_table_mouse_event(event, x, y, flags, param) :
    global _table_scale_base_pos, _table_mouse_state, _table_points, _table_scale, _table_mouse_now_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        _table_mouse_state = "down"
        _table_scale_base_pos = (x, y)
    if event == cv2.EVENT_LBUTTONUP :
        _table_mouse_state = "up"
        _table_points.append([
            (x - _table_scale_base_pos[0]) / _table_scale + _table_scale_base_pos[0],
            (y - _table_scale_base_pos[1]) / _table_scale + _table_scale_base_pos[1]
        ])
    if event == cv2.EVENT_MOUSEMOVE :
        if _table_mouse_state == "down" :
            _table_mouse_now_pos = (
                (x - _table_scale_base_pos[0]) / _table_scale + _table_scale_base_pos[0],
                (y - _table_scale_base_pos[1]) / _table_scale + _table_scale_base_pos[1]
            )
        else :
            _table_mouse_now_pos = (x, y)
    if event == cv2.EVENT_RBUTTONDOWN :
        _table_points.pop()

def setup_table_img(img, hw = 2.74/2, hh = 1.525/2) :
    
    global _table_mouse_state, _table_scale_base_pos, _table_points, _table_scale, _table_mouse_now_pos
    cv2.namedWindow("setup table")
    cv2.setMouseCallback("setup table", _setup_table_mouse_event)

    while True :
        try: 
            frame = img.copy()
            k = cv2.waitKey(round(1/30*1000))
            # draw poly
            if len(_table_points) >= 1 :
                cv2.polylines(frame, np.array([[(round(d[0]), round(d[1])) for d in _table_points] + [(round(_table_mouse_now_pos[0]), round(_table_mouse_now_pos[1])) ]]), True, (0, 255, 0), 1)
            if _table_mouse_state == "down" :
                frame = cv2.warpAffine(frame, np.array(
                    [
                        [_table_scale, 0, (1 - _table_scale) * _table_scale_base_pos[0]],
                        [0, _table_scale, (1 - _table_scale) * _table_scale_base_pos[1]] ], dtype=np.float32), img.shape[:2][::-1])
            else :
                if len(_table_points) == 4 :
                    break
            cv2.imshow("setup table", frame)
        except Exception as e :
            print(e)
    cv2.destroyAllWindows()
    result = np.array(_table_points)
    _table_mouse_state = "up"
    _table_points = []
    _table_scale = 3
    _table_scale_base_pos = (0, 0)
    _table_mouse_now_pos = (0, 0)
    tar = np.float32([[-hw, -hh], [-hw, hh], [hw, hh], [hw, -hh]])
    return cv2.findHomography(result, tar)[0]

def setup_table(source, hw = 2.74/2, hh = 1.525/2) :
    global _table_mouse_state, _table_scale_base_pos, _table_points, _table_scale, _table_mouse_now_pos
    cam = cv2.VideoCapture(source)
    cv2.namedWindow("setup table")
    cv2.setMouseCallback("setup table", _setup_table_mouse_event)

    while True :
        ret, frame = cam.read()
        if ret :
            try: 
                k = cv2.waitKey(round(1/30*1000))
                # draw poly
                if len(_table_points) >= 1 :
                    cv2.polylines(frame, np.array([[(round(d[0]), round(d[1])) for d in _table_points] + [(round(_table_mouse_now_pos[0]), round(_table_mouse_now_pos[1])) ]]), True, (0, 255, 0), 1)
                if _table_mouse_state == "down" :
                    frame = cv2.warpAffine(frame, np.array(
                        [
                            [_table_scale, 0, (1 - _table_scale) * _table_scale_base_pos[0]],
                            [0, _table_scale, (1 - _table_scale) * _table_scale_base_pos[1]] ], dtype=np.float32), frame.shape[:2][::-1])
                else :
                    if len(_table_points) == 4 :
                        break
                cv2.imshow("setup table", frame)
            except Exception as e :
                print(e)
    cv2.destroyAllWindows()
    result = np.array(_table_points)
    _table_mouse_state = "up"
    _table_points = []
    _table_scale = 3
    _table_scale_base_pos = (0, 0)
    _table_mouse_now_pos = (0, 0)
    tar = np.float32([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
    return cv2.findHomography(result, tar)[0]

def find_homography_matrix_to_apriltag(img_gray) -> np.ndarray | None:
    tag_len   = APRILTAG_SIZE 
    detector  = Detector()
    detection = detector.detect(img_gray)
    if len(detection) == 0 :
        return None
    coners = detection[0].corners
    # draw coners
    for i in range(4) :
        cv2.line(img_gray, tuple(coners[i-1].astype(int)), tuple(coners[i].astype(int)), (0, 0, 255), 2)
    tar    = np.float32([[0, 0], [tag_len, 0], [tag_len, tag_len], [0, tag_len]])
    homography = cv2.findHomography(coners, tar)[0]
    return homography

def test_homo(img, ho) :
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', on_mouse_move)
    #ho = pickle.load(open('ball_detection/result/s480p30_a15_/homography_matrix', 'rb'))
    while True :
        cv2.imshow("frame", img)
        cv2.waitKey(round(1/30*1000))
        a = np.matmul(ho, np.array([_x, _y, 1]))
        a = a / a[2]
        b = np.matmul(np.linalg.inv(ho), np.array([_x, _y, 1]))
        b = b / b[2]
        print("a: %2f, %2f" %(a[0], a[1]), a[2])
        print("b: %2f, %2f" %(b[0], b[1]), b[2])
    print(ho)

if __name__ == "__main__" :
    #img = cv2.imread("exp/718.jpg")
    #m = setup_table_img(img, APRILTAG_SIZE/2, APRILTAG_SIZE/2)
    #pickle.dump(m, open('ho_table', 'wb'))
    #exit()
    c = pickle.load(open('calibration', 'rb'))
    d = Detector().detect(cv2.imread("exp/718.jpg", cv2.IMREAD_GRAYSCALE), estimate_tag_pose=True, camera_params=(c[0][0],c[1][1],c[0][2],c[1][2]), tag_size=APRILTAG_SIZE)
    m = pickle.load(open('ho_table', 'rb'))
    #test_homo(cv2.imread("exp/718.jpg"), m)
    p_t = calculateCameraPosition_table(c, m)
    p_a = np.matmul(np.linalg.inv(d[0].pose_R), d[0].pose_t)

    exit()

    

    exit()
        #runExerment()
    cameraMatrix = pickle.load(open('calibration', 'rb'))
    #takePicture()
    img = cv2.imread("718-2.jpg", cv2.IMREAD_GRAYSCALE)
    #dec = Detector()
    #d = dec.detect(img, estimate_tag_pose=True, camera_params=(cameraMatrix[0][0],cameraMatrix[1][1],cameraMatrix[0][2],cameraMatrix[1][2]), tag_size=29.8)
    #cv2.imshow('frame', img)
    #cv2.waitKey(0)
    
    print(calculateCameraPosition(cameraMatrix, img).to_str())


    #getCameraPosition_realTime(cameraMatrix)
    #calculateCameraPosition(cameraMatrix, cv2.imread('test.jpg'))
    #print(cameraMatrix)
    #getCameraPosition_realTime(cameraMatrix)
    pass