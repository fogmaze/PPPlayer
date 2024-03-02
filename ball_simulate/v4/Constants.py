import math

now_mode = "none"

MODEL3_OUTPUT_LEN = None

APRILTAG_SIZE = 0.164
CHESS_BOARD_SIZE = 31.5

MAX_SINGLE_FILE_DATA_LEN = 100000

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
INPUT_RANDOM_ERROR_RATE      = None


def set2NormalBR4() :
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
    global INPUT_RANDOM_ERROR_RATE
    global MODEL3_OUTPUT_LEN
    global normer
    global now_mode

    SIMULATE_TEST_LEN = MODEL3_OUTPUT_LEN = 50
    SIMULATE_INPUT_LEN = 40
    SHUTTER_RANDOM_ERROR_STD = 0.005    #second
    SHUTTER_SYSTEMATIC_ERROR_STD = 0.03 #second
    CAMERA_POSITION_ERROR_STD = 0.05    #meter
    BALL_POSITION_ERROR_STD = 0.05      #meter
    INPUT_IGNORE_AREA_MEAN = 3
    INPUT_IGNORE_AREA_STD = 3
    INPUT_IGNORE_WIDTH_MEAN = 4
    INPUT_IGNORE_WIDTH_STD = 3
    INPUT_RANDOM_ERROR_RATE = 0.03
    normer = Normer()
    now_mode = "normalBR3"


def set2Better4() :
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
    global INPUT_RANDOM_ERROR_RATE
    global MODEL3_OUTPUT_LEN
    global normer
    global now_mode

    SIMULATE_TEST_LEN = MODEL3_OUTPUT_LEN = 60
    SIMULATE_INPUT_LEN = 40
    SHUTTER_RANDOM_ERROR_STD = 0.005    #second
    SHUTTER_SYSTEMATIC_ERROR_STD = 0.03 #second
    CAMERA_POSITION_ERROR_STD = 0.025    #meter
    BALL_POSITION_ERROR_STD = 0.025      #meter
    INPUT_IGNORE_AREA_MEAN = 1
    INPUT_IGNORE_AREA_STD = 3
    INPUT_IGNORE_WIDTH_MEAN = 4
    INPUT_IGNORE_WIDTH_STD = 3
    INPUT_RANDOM_ERROR_RATE = 0.03
    normer = Normer()
    now_mode = "better3"

def set2Short4() :
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
    global INPUT_RANDOM_ERROR_RATE
    global MODEL3_OUTPUT_LEN
    global normer
    global now_mode

    SIMULATE_TEST_LEN = MODEL3_OUTPUT_LEN = 30
    SIMULATE_INPUT_LEN = 40
    SHUTTER_RANDOM_ERROR_STD = 0.005    #second
    SHUTTER_SYSTEMATIC_ERROR_STD = 0.03 #second
    CAMERA_POSITION_ERROR_STD = 0.025    #meter
    BALL_POSITION_ERROR_STD = 0.025      #meter
    INPUT_IGNORE_AREA_MEAN = 1
    INPUT_IGNORE_AREA_STD = 3
    INPUT_IGNORE_WIDTH_MEAN = 4
    INPUT_IGNORE_WIDTH_STD = 3
    INPUT_RANDOM_ERROR_RATE = 0.03
    normer = Normer()
    now_mode = "short4"

SIMULATE_INPUT_LEN_DICT = {
    "normalBR4": 40,
    "better4": 40,
    "short4": 40
    # Add values for other modes if needed
}

SIMULATE_TEST_LEN_DICT = {
    "normalBR4": 50,
    "better4": 60,
    "short4": 30,
    # Add values for other modes if needed
}


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

        for i in range(SIMULATE_TEST_LEN) :
            data.curvePoints[i].x = (data.curvePoints[i].x - self.BALL_MEAN[0]) / self.BALL_STD[0]
            data.curvePoints[i].y = (data.curvePoints[i].y - self.BALL_MEAN[1]) / self.BALL_STD[1]
            data.curvePoints[i].z = (data.curvePoints[i].z - self.BALL_MEAN[2]) / self.BALL_STD[2]
        
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

        for i in range(SIMULATE_TEST_LEN) :
            data.curvePoints[i].x = data.curvePoints[i].x * self.BALL_STD[0] + self.BALL_MEAN[0]
            data.curvePoints[i].y = data.curvePoints[i].y * self.BALL_STD[1] + self.BALL_MEAN[1]
            data.curvePoints[i].z = data.curvePoints[i].z * self.BALL_STD[2] + self.BALL_MEAN[2]

    def unnorm_ans_tensor(self, data) :
        d = data.view(-1, 3)
        d.transpose_(0, 1)
        d[0] = d[0] * self.BALL_STD[0] + self.BALL_MEAN[0]
        d[1] = d[1] * self.BALL_STD[1] + self.BALL_MEAN[1]
        d[2] = d[2] * self.BALL_STD[2] + self.BALL_MEAN[2]
        d.transpose_(0, 1)
        return d
    def norm_ans_tensor(self, data) :
        d = data.view(-1, 3)
        d.transpose_(0, 1)
        d[0] = (d[0] - self.BALL_MEAN[0]) / self.BALL_STD[0]
        d[1] = (d[1] - self.BALL_MEAN[1]) / self.BALL_STD[1]
        d[2] = (d[2] - self.BALL_MEAN[2]) / self.BALL_STD[2]
        d.transpose_(0, 1)
        return d
    def norm_t_tensor(self, data) :
        d = data.view(-1, 1)
        d.transpose_(0, 1)
        d[0] = (d[0] - self.TIME_MEAN[0]) / self.TIME_STD[0]
        d.transpose_(0, 1)
        return d
    def unnorm_t_tensor(self, data) :
        d = data.view(-1, 1)
        d.transpose_(0, 1)
        d[0] = d[0] * self.TIME_STD[0] + self.TIME_MEAN[0]
        d.transpose_(0, 1)
        return d
    def norm_input_tensor(self, data) :
        d = data.view(-1, 5)
        d.transpose_(0, 1)
        d[0] = (d[0] - self.CAM_MEAN[0]) / self.CAM_STD[0]
        d[1] = (d[1] - self.CAM_MEAN[1]) / self.CAM_STD[1]
        d[2] = (d[2] - self.CAM_MEAN[2]) / self.CAM_STD[2]
        d[3] = (d[3] - self.LINE_MEAN[0]) / self.LINE_STD[0]
        d[4] = (d[4] - self.LINE_MEAN[1]) / self.LINE_STD[1]
        d.transpose_(0, 1)
        return d
    def unorm_input_tensor(self, data) :
        d = data.view(-1, 5)
        d.transpose_(0, 1)
        d[0] = d[0] * self.CAM_STD[0] + self.CAM_MEAN[0]
        d[1] = d[1] * self.CAM_STD[1] + self.CAM_MEAN[1]
        d[2] = d[2] * self.CAM_STD[2] + self.CAM_MEAN[2]
        d[3] = d[3] * self.LINE_STD[0] + self.LINE_MEAN[0]
        d[4] = d[4] * self.LINE_STD[1] + self.LINE_MEAN[1]
        d.transpose_(0, 1)
        return d
normer = None