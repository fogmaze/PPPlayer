import math
SIMULATE_INPUT_LEN = 100
SIMULATE_TEST_LEN = 250

BALL_AREA_HALF_LENGTH = 3
BALL_AREA_HALF_WIDTH = 2
BALL_AREA_HEIGHT = 1

CAMERA_AREA_HALF_LENGTH = 7/2
CAMERA_AREA_HALF_WIDTH = 5/2
CAMERA_AREA_HEIGHT = 1.5

CURVE_SHOWING_GAP = 0.05

stepTime = 1./600.
G = 9.8
FPS = 30

SHUTTER_RANDOM_ERROR_STD = 0.005
SHUTTER_SYSTEMATIC_ERROR_STD = 0.01
CAMERA_POSITION_ERROR_STD = 0.05
BALL_POSITION_ERROR_STD = 0.05

INPUT_IGNORE_AREA_MEAN = 3
INPUT_IGNORE_AREA_STD = 2
INPUT_IGNORE_WIDTH_MEAN = 4
INPUT_IGNORE_WIDTH_STD = 3

def set2NoError() :
    global SHUTTER_RANDOM_ERROR_STD
    global SHUTTER_SYSTEMATIC_ERROR_STD
    global CAMERA_POSITION_ERROR_STD
    global BALL_POSITION_ERROR_STD
    global INPUT_IGNORE_AREA_MEAN
    global INPUT_IGNORE_AREA_STD
    global INPUT_IGNORE_WIDTH_MEAN
    global INPUT_IGNORE_WIDTH_STD

    SHUTTER_RANDOM_ERROR_STD = 0
    SHUTTER_SYSTEMATIC_ERROR_STD = 0
    CAMERA_POSITION_ERROR_STD = 0
    BALL_POSITION_ERROR_STD = 0
    INPUT_IGNORE_AREA_MEAN = 0
    INPUT_IGNORE_AREA_STD = 0
    INPUT_IGNORE_WIDTH_MEAN = 0
    INPUT_IGNORE_WIDTH_STD = 0

def set2Fitting() :
    global SIMULATE_TEST_LEN

    SIMULATE_TEST_LEN = 100


class Normer :
    CAM_MEAN = [0.0, 0.0, CAMERA_AREA_HEIGHT/2]
    CAM_STD = [CAMERA_AREA_HALF_LENGTH, CAMERA_AREA_HALF_WIDTH, CAMERA_AREA_HEIGHT/2]
    LINE_MEAN = [0, 0]
    LINE_STD = [math.pi/2, math.pi/2]
    BALL_MEAN = [0.0, 0.0, BALL_AREA_HEIGHT/2]
    BALL_STD = [BALL_AREA_HALF_LENGTH*2, BALL_AREA_HALF_WIDTH*2, BALL_AREA_HEIGHT/2]
    TIME_MEAN = [CURVE_SHOWING_GAP*SIMULATE_TEST_LEN/2]
    TIME_STD = [CURVE_SHOWING_GAP*SIMULATE_TEST_LEN/2]

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

normer = Normer()