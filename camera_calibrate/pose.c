#include "apriltag/apriltag.h"
#include "apriltag/common/matd.h"
#include "apriltag/apriltag_pose.h"
#include "math.h"
#include "stdio.h"

double orthogonal_iteration(matd_t** v, matd_t** p, matd_t** t, matd_t** R, int n_points, int n_steps);
matd_t* fix_pose_ambiguities(matd_t** v, matd_t** p, matd_t* t, matd_t* R, int n_points);

apriltag_detection_info_t* createInfo(matd_t* H, double p[4][2], double fx, double fy, double cx, double cy) {
    apriltag_detection_info_t* info = malloc(sizeof(apriltag_detection_info_t));// = new apriltag_detection_info_t();
    info->fx = fx;
    info->fy = fy;
    info->cx = cx;
    info->cy = cy;
    info->tagsize = 2;
    info->det = malloc(sizeof(apriltag_detection_t));
    info->det->H = H;
    for (int i = 0; i < 4; i++) {
        info->det->p[i][0] = p[i][0];
        info->det->p[i][1] = p[i][1];
    }
    return info;
}

void estimate_tag_pose_orthogonal_iteration_free(
        apriltag_detection_info_t* info,
        double* err1,
        apriltag_pose_t* solution1,
        double* err2,
        apriltag_pose_t* solution2,
        int nIters,
        double width, 
        double height ) {

    matd_t* p[4] = {
        matd_create_data(3, 1, (double[]) {-width/2, height/2, 0}),
        matd_create_data(3, 1, (double[]) {width/2, height/2, 0}),
        matd_create_data(3, 1, (double[]) {width/2, -height/2, 0}),
        matd_create_data(3, 1, (double[]) {-width/2, -height/2, 0})};
    matd_t* v[4];
    for (int i = 0; i < 4; i++) {
        v[i] = matd_create_data(3, 1, (double[]) {
        (info->det->p[i][0] - info->cx)/info->fx, (info->det->p[i][1] - info->cy)/info->fy, 1});
    }

    estimate_pose_for_tag_homography(info, solution1);
    *err1 = orthogonal_iteration(v, p, &solution1->t, &solution1->R, 4, nIters);
    solution2->R = fix_pose_ambiguities(v, p, solution1->t, solution1->R, 4);
    if (solution2->R) {
        solution2->t = matd_create(3, 1);
        *err2 = orthogonal_iteration(v, p, &solution2->t, &solution2->R, 4, nIters);
    } else {
        *err2 = HUGE_VAL;
    }

    for (int i = 0; i < 4; i++) {
        matd_destroy(p[i]);
        matd_destroy(v[i]);
    }
}


double estimate(
        matd_t* homo, 
        double corners[4][2], 
        double fx,
        double fy,
        double cx,
        double cy,
        double width, 
        double height, 
        apriltag_pose_t* pose ) {
    apriltag_detection_info_t* info = createInfo(homo, corners, fx, fy, cx, cy);
    double err1, err2;
    apriltag_pose_t pose1, pose2;
    estimate_tag_pose_orthogonal_iteration_free(info, &err1, &pose1, &err2, &pose2, 50, width, height);
    if (err1 <= err2) {
        pose->R = pose1.R;
        pose->t = pose1.t;
        if (pose2.R) {
            matd_destroy(pose2.t);
        }
        matd_destroy(pose2.R);
        free(info);
        return err1;
    } else {
        pose->R = pose2.R;
        pose->t = pose2.t;
        matd_destroy(pose1.R);
        matd_destroy(pose1.t);
        free(info);
        return err2;
    }
}

int main() {
    return 0;
}