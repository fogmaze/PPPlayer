import numpy as np
import cv2
import glob
import pickle
import os

class Calibrator(object):
    def __init__(self, source, shape_inner_corner, size_grid, visualization=True):
        """
        --parameters--
        img_dir: the directory that save images for calibration, str
        shape_inner_corner: the shape of inner corner, Array of int, (h, w)
        size_grid: the real size of a grid in calibrator, float
        visualization: whether visualization, bool
        """
        self.shape_inner_corner = shape_inner_corner
        self.size_grid = size_grid
        self.visualization = visualization
        self.mat_intri = None # intrinsic matrix
        self.coff_dis = None # cofficients of distortion
        self.points_world = None # the points in world space
        self.points_pixel = None # the points in pixel space (relevant to points_world)
        self.imgSize = None # the size of image

        # create the conner in world space
        w, h = shape_inner_corner
        # cp_int: corner point in int form, save the coordinate of corner points in world sapce in 'int' form
        # like (0,0,0), (1,0,0), (2,0,0) ...., (10,7,0)
        cp_int = np.zeros((w * h, 3), np.float32)
        cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        # cp_world: corner point in world space, save the coordinate of corner points in world space
        self.cp_world = cp_int * size_grid

        # from file
        if type(source) == str:
            self.load_Images(source)

        # from camera
        elif type(source) == int:
            self.load_Camera(source)

    def load_Camera(self, source):
        cap = cv2.VideoCapture(source)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        w, h = self.shape_inner_corner
        self.points_world = []
        self.points_pixel = []
        while True:
            ret, img = cap.read()
            cv2.imshow('img', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('c'):
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
                cp_img = cv2.cornerSubPix(gray_img, cp_img, (11,11), (-1,-1), criteria)
                if ret:
                    self.points_world.append(self.cp_world)
                    self.points_pixel.append(cp_img)
                    if self.visualization:
                        cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
                        cv2.imshow('FoundCorners', img)
                        cv2.waitKey(500)
                break

    def load_Images(self, img_dir):
        # images
        img_paths = []
        for extension in ["jpg", "png", "jpeg"]:
            img_paths += glob.glob(os.path.join(img_dir, "*.{}".format(extension)))
        assert len(img_paths), "No images for calibration found!"

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        w, h = self.shape_inner_corner
        self.points_world = [] # the points in world space
        self.points_pixel = [] # the points in pixel space (relevant to points_world)
        for img_path in img_paths:
            img = cv2.imread(img_path)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the corners, cp_img: corner points in pixel space
            ret, cp_img = cv2.findChessboardCorners(gray_img, (w, h), None)
            # refine the corner points
            cp_img = cv2.cornerSubPix(gray_img, cp_img, (11,11), (-1,-1), criteria)
            # if ret is True, save
            if ret:
                # cv2.cornerSubPix(gray_img, cp_img, (11,11), (-1,-1), criteria)
                self.points_world.append(self.cp_world)
                self.points_pixel.append(cp_img)
                # view the corners
                if self.visualization:
                    cv2.drawChessboardCorners(img, (w, h), cp_img, ret)
                    cv2.imshow('FoundCorners', img)
                    cv2.waitKey(500)
        self.imgSize = gray_img.shape[::-1]

    def calibrate_camera(self):
        # calibrate the camera
        ret, mat_intri, coff_dis, v_rot, v_trans = cv2.calibrateCamera(self.points_world, self.points_pixel, self.imgSize, None, None)
        print ("ret: {}".format(ret))
        print ("intrinsic matrix: \n {}".format(mat_intri))
        # in the form of (k_1, k_2, p_1, p_2, k_3)
        print ("distortion cofficients: \n {}".format(coff_dis))
        print ("rotation vectors: \n {}".format(v_rot))
        print ("translation vectors: \n {}".format(v_trans))

        # calculate the error of reproject
        total_error = 0
        for i in range(len(self.points_world)):
            points_pixel_repro, _ = cv2.projectPoints(self.points_world[i], v_rot[i], v_trans[i], mat_intri, coff_dis)
            error = cv2.norm(self.points_pixel[i], points_pixel_repro, cv2.NORM_L2) / len(points_pixel_repro)
            total_error += error
        print("Average error of reproject: {}".format(total_error / len(self.points_world)))

        self.mat_intri = mat_intri
        self.coff_dis = coff_dis
        return mat_intri, coff_dis

