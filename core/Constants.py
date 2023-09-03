import math

APRILTAG_SIZE = 0.164
CHESS_BOARD_SIZE = 31.5

MODEL_INPUT_SIZE = 5
MODEL_OUTPUT_SIZE = 3

BALL_AREA_HALF_LENGTH = 3/2
BALL_AREA_HALF_WIDTH = 2/2
BALL_AREA_HEIGHT = 1

CAMERA_AREA_HALF_LENGTH = 5/2
CAMERA_AREA_HALF_WIDTH = 4/2
CAMERA_AREA_HEIGHT = 1.5

CURVE_SHOWING_GAP = 0.05

stepTime = 1./600.
G = 9.8
FPS = 30

SHUTTER_RANDOM_ERROR_STD     = None 
SHUTTER_SYSTEMATIC_ERROR_STD = None 
CAMERA_POSITION_ERROR_STD    = None 
BALL_POSITION_ERROR_STD      = None
INPUT_IGNORE_AREA_MEAN       = None
INPUT_IGNORE_AREA_STD        = None
INPUT_IGNORE_WIDTH_MEAN      = None
INPUT_IGNORE_WIDTH_STD       = None

def set2NoError() :
    global SHUTTER_RANDOM_ERROR_STD
    global SHUTTER_SYSTEMATIC_ERROR_STD
    global CAMERA_POSITION_ERROR_STD
    global BALL_POSITION_ERROR_STD
    global INPUT_IGNORE_AREA_MEAN
    global INPUT_IGNORE_AREA_STD
    global INPUT_IGNORE_WIDTH_MEAN
    global INPUT_IGNORE_WIDTH_STD
    global SIMULATE_INPUT_LEN
    global SIMULATE_TEST_LEN
    global normer

    SHUTTER_RANDOM_ERROR_STD = 0
    SHUTTER_SYSTEMATIC_ERROR_STD = 0
    CAMERA_POSITION_ERROR_STD = 0
    BALL_POSITION_ERROR_STD = 0
    INPUT_IGNORE_AREA_MEAN = 0
    INPUT_IGNORE_AREA_STD = 0
    INPUT_IGNORE_WIDTH_MEAN = 0
    INPUT_IGNORE_WIDTH_STD = 0
    SIMULATE_INPUT_LEN = 40
    SIMULATE_TEST_LEN = 250
    normer = Normer()

def set2Normal() :
    global SIMULATE_INPUT_LEN
    global SIMULATE_TEST_LEN
    global SHUTTER_RANDOM_ERROR_STD
    global SHUTTER_SYSTEMATIC_ERROR_STD
    global CAMERA_POSITION_ERROR_STD
    global BALL_POSITION_ERROR_STD
    global INPUT_IGNORE_AREA_MEAN
    global INPUT_IGNORE_AREA_STD
    global INPUT_IGNORE_WIDTH_MEAN
    global INPUT_IGNORE_WIDTH_STD
    global normer

    SIMULATE_TEST_LEN = 50
    SIMULATE_INPUT_LEN = 40
    SHUTTER_RANDOM_ERROR_STD = 0.005
    SHUTTER_SYSTEMATIC_ERROR_STD = 0.01
    CAMERA_POSITION_ERROR_STD = 0.05
    BALL_POSITION_ERROR_STD = 0.05
    INPUT_IGNORE_AREA_MEAN = 3
    INPUT_IGNORE_AREA_STD = 2
    INPUT_IGNORE_WIDTH_MEAN = 4
    INPUT_IGNORE_WIDTH_STD = 3
    normer = Normer()

def set2NormalB60() :
    global SIMULATE_INPUT_LEN
    global SIMULATE_TEST_LEN
    global SHUTTER_RANDOM_ERROR_STD
    global SHUTTER_SYSTEMATIC_ERROR_STD
    global CAMERA_POSITION_ERROR_STD
    global BALL_POSITION_ERROR_STD
    global INPUT_IGNORE_AREA_MEAN
    global INPUT_IGNORE_AREA_STD
    global INPUT_IGNORE_WIDTH_MEAN
    global INPUT_IGNORE_WIDTH_STD
    global normer
    global FPS

    FPS = 60
    SIMULATE_TEST_LEN = 50
    SIMULATE_INPUT_LEN = 80
    SHUTTER_RANDOM_ERROR_STD = 0.005
    SHUTTER_SYSTEMATIC_ERROR_STD = 0.03
    CAMERA_POSITION_ERROR_STD = 0.05
    BALL_POSITION_ERROR_STD = 0.05
    INPUT_IGNORE_AREA_MEAN = 4
    INPUT_IGNORE_AREA_STD = 3
    INPUT_IGNORE_WIDTH_MEAN = 8
    INPUT_IGNORE_WIDTH_STD = 8
    normer = Normer()


def set2NormalB() :
    global SIMULATE_INPUT_LEN
    global SIMULATE_TEST_LEN
    global SHUTTER_RANDOM_ERROR_STD
    global SHUTTER_SYSTEMATIC_ERROR_STD
    global CAMERA_POSITION_ERROR_STD
    global BALL_POSITION_ERROR_STD
    global INPUT_IGNORE_AREA_MEAN
    global INPUT_IGNORE_AREA_STD
    global INPUT_IGNORE_WIDTH_MEAN
    global INPUT_IGNORE_WIDTH_STD
    global normer

    SIMULATE_TEST_LEN = 50
    SIMULATE_INPUT_LEN = 40
    SHUTTER_RANDOM_ERROR_STD = 0.005
    SHUTTER_SYSTEMATIC_ERROR_STD = 0.03
    CAMERA_POSITION_ERROR_STD = 0.05
    BALL_POSITION_ERROR_STD = 0.05
    INPUT_IGNORE_AREA_MEAN = 4
    INPUT_IGNORE_AREA_STD = 3
    INPUT_IGNORE_WIDTH_MEAN = 4
    INPUT_IGNORE_WIDTH_STD = 4
    normer = Normer()

def set2Fitting() :
    global SIMULATE_TEST_LEN
    global SIMULATE_INPUT_LEN
    global SHUTTER_RANDOM_ERROR_STD
    global SHUTTER_SYSTEMATIC_ERROR_STD
    global CAMERA_POSITION_ERROR_STD
    global BALL_POSITION_ERROR_STD
    global INPUT_IGNORE_AREA_MEAN
    global INPUT_IGNORE_AREA_STD
    global INPUT_IGNORE_WIDTH_MEAN
    global INPUT_IGNORE_WIDTH_STD
    global normer

    SIMULATE_TEST_LEN = 100
    SIMULATE_INPUT_LEN = 100
    SHUTTER_RANDOM_ERROR_STD = 0.005
    SHUTTER_SYSTEMATIC_ERROR_STD = 0.01
    CAMERA_POSITION_ERROR_STD = 0.05
    BALL_POSITION_ERROR_STD = 0.05
    INPUT_IGNORE_AREA_MEAN = 3
    INPUT_IGNORE_AREA_STD = 2
    INPUT_IGNORE_WIDTH_MEAN = 4
    INPUT_IGNORE_WIDTH_STD = 3
    normer = Normer()

def set2Predict() :
    global SIMULATE_INPUT_LEN
    global SIMULATE_TEST_LEN
    global SHUTTER_RANDOM_ERROR_STD
    global SHUTTER_SYSTEMATIC_ERROR_STD
    global CAMERA_POSITION_ERROR_STD
    global BALL_POSITION_ERROR_STD
    global INPUT_IGNORE_AREA_MEAN
    global INPUT_IGNORE_AREA_STD
    global INPUT_IGNORE_WIDTH_MEAN
    global INPUT_IGNORE_WIDTH_STD
    global normer

    SIMULATE_TEST_LEN = 250
    SIMULATE_INPUT_LEN = 40
    SHUTTER_RANDOM_ERROR_STD = 0.005
    SHUTTER_SYSTEMATIC_ERROR_STD = 0.01
    CAMERA_POSITION_ERROR_STD = 0.05
    BALL_POSITION_ERROR_STD = 0.05
    INPUT_IGNORE_AREA_MEAN = 3
    INPUT_IGNORE_AREA_STD = 2
    INPUT_IGNORE_WIDTH_MEAN = 4
    INPUT_IGNORE_WIDTH_STD = 3
    normer = Normer()


class Normer :
    def __init__(self) :
        self.CAM_MEAN = [0.0, 0.0, CAMERA_AREA_HEIGHT/2]
        self.CAM_STD = [CAMERA_AREA_HALF_LENGTH, CAMERA_AREA_HALF_WIDTH, CAMERA_AREA_HEIGHT/2]
        self.LINE_MEAN = [0, 0]
        self.LINE_STD = [math.pi/2, math.pi/2]
        self.BALL_MEAN = [0.0, 0.0, BALL_AREA_HEIGHT/2]
        self.BALL_STD = [BALL_AREA_HALF_LENGTH*2, BALL_AREA_HALF_WIDTH*2, BALL_AREA_HEIGHT/2]
        self.TIME_MEAN = [CURVE_SHOWING_GAP*SIMULATE_TEST_LEN/2]
        self.TIME_STD = [CURVE_SHOWING_GAP*SIMULATE_TEST_LEN/2]

    def norm(self, data) :
        data.inputs[0].camera_x = (data.inputs[0].camera_x - self.CAM_MEAN[0]) / self.CAM_STD[0]
        data.inputs[0].camera_y = (data.inputs[0].camera_y - self.CAM_MEAN[1]) / self.CAM_STD[1]
        data.inputs[0].camera_z = (data.inputs[0].camera_z - self.CAM_MEAN[2]) / self.CAM_STD[2]
        data.inputs[1].camera_x = (data.inputs[1].camera_x - self.CAM_MEAN[0]) / self.CAM_STD[0]
        data.inputs[1].camera_y = (data.inputs[1].camera_y - self.CAM_MEAN[1]) / self.CAM_STD[1]
        data.inputs[1].camera_z = (data.inputs[1].camera_z - self.CAM_MEAN[2]) / self.CAM_STD[2]

        for i in range(SIMULATE_INPUT_LEN):
            data.inputs[0].line_rad_xy[i] = (data.inputs[0].line_rad_xy[i] - self.LINE_MEAN[0]) / self.LINE_STD[0]
            data.inputs[0].line_rad_xz[i] = (data.inputs[0].line_rad_xz[i] - self.LINE_MEAN[1]) / self.LINE_STD[1]
            data.inputs[1].line_rad_xy[i] = (data.inputs[1].line_rad_xy[i] - self.LINE_MEAN[0]) / self.LINE_STD[0]
            data.inputs[1].line_rad_xz[i] = (data.inputs[1].line_rad_xz[i] - self.LINE_MEAN[1]) / self.LINE_STD[1]

            data.inputs[0].timestamps[i] = (data.inputs[0].timestamps[i] - self.TIME_MEAN[0]) / self.TIME_STD[0]
            data.inputs[1].timestamps[i] = (data.inputs[1].timestamps[i] - self.TIME_MEAN[0]) / self.TIME_STD[0]

        for i in range(SIMULATE_TEST_LEN) :
            data.curvePoints[i].x = (data.curvePoints[i].x - self.BALL_MEAN[0]) / self.BALL_STD[0]
            data.curvePoints[i].y = (data.curvePoints[i].y - self.BALL_MEAN[1]) / self.BALL_STD[1]
            data.curvePoints[i].z = (data.curvePoints[i].z - self.BALL_MEAN[2]) / self.BALL_STD[2]
            data.curveTimestamps[i] = (data.curveTimestamps[i] - self.TIME_MEAN[0]) / self.TIME_STD[0]
        
    def unnorm(self, data) :
        data.inputs[0].camera_x = data.inputs[0].camera_x * self.CAM_STD[0] + self.CAM_MEAN[0]
        data.inputs[0].camera_y = data.inputs[0].camera_y * self.CAM_STD[1] + self.CAM_MEAN[1]
        data.inputs[0].camera_z = data.inputs[0].camera_z * self.CAM_STD[2] + self.CAM_MEAN[2]
        data.inputs[1].camera_x = data.inputs[1].camera_x * self.CAM_STD[0] + self.CAM_MEAN[0]
        data.inputs[1].camera_y = data.inputs[1].camera_y * self.CAM_STD[1] + self.CAM_MEAN[1]
        data.inputs[1].camera_z = data.inputs[1].camera_z * self.CAM_STD[2] + self.CAM_MEAN[2]

        for i in range(SIMULATE_INPUT_LEN) :
            data.inputs[0].line_rad_xy[i] = data.inputs[0].line_rad_xy[i] * self.LINE_STD[0] + self.LINE_MEAN[0]
            data.inputs[0].line_rad_xz[i] = data.inputs[0].line_rad_xz[i] * self.LINE_STD[1] + self.LINE_MEAN[1]
            data.inputs[1].line_rad_xy[i] = data.inputs[1].line_rad_xy[i] * self.LINE_STD[0] + self.LINE_MEAN[0]
            data.inputs[1].line_rad_xz[i] = data.inputs[1].line_rad_xz[i] * self.LINE_STD[1] + self.LINE_MEAN[1]

            data.inputs[0].timestamps[i] = data.inputs[0].timestamps[i] * self.TIME_STD[0] + self.TIME_MEAN[0]
            data.inputs[1].timestamps[i] = data.inputs[1].timestamps[i] * self.TIME_STD[0] + self.TIME_MEAN[0]

        for i in range(SIMULATE_TEST_LEN) :
            data.curvePoints[i].x = data.curvePoints[i].x * self.BALL_STD[0] + self.BALL_MEAN[0]
            data.curvePoints[i].y = data.curvePoints[i].y * self.BALL_STD[1] + self.BALL_MEAN[1]
            data.curvePoints[i].z = data.curvePoints[i].z * self.BALL_STD[2] + self.BALL_MEAN[2]
            data.curveTimestamps[i] = data.curveTimestamps[i] * self.TIME_STD[0] + self.TIME_MEAN[0]

    def unnorm_ans_tensor(self, data) :
        for i in range(len(data)) :
            data[i][0] = data[i][0] * self.BALL_STD[0] + self.BALL_MEAN[0]
            data[i][1] = data[i][1] * self.BALL_STD[1] + self.BALL_MEAN[1]
            data[i][2] = data[i][2] * self.BALL_STD[2] + self.BALL_MEAN[2]

    def unorm_input_tensor(self, data) :
        for i in range(len(data)) :
            data[i][0] = data[i][0] * self.CAM_STD[0] + self.CAM_MEAN[0]
            data[i][1] = data[i][1] * self.CAM_STD[1] + self.CAM_MEAN[1]
            data[i][2] = data[i][2] * self.CAM_STD[2] + self.CAM_MEAN[2]
            data[i][3] = data[i][3] * self.LINE_STD[0] + self.LINE_MEAN[0]
            data[i][4] = data[i][4] * self.LINE_STD[1] + self.LINE_MEAN[1]
normer = None