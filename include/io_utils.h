#pragma once

#include "cv_ext/cv_ext.h"
#include <string>
#include <vector>

bool readFileNamesFromFolder ( const std::string& input_folder_name,
                               std::vector< std::string >& names );
bool loadCameraParams( const std::string &file_name, cv::Size &image_size,
                       cv::Mat &camera_matrix, cv::Mat &dist_coeffs );
void saveCameraParams( const std::string &file_name, const cv::Size &image_size,
                       const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs );
bool loadStereoCameraParams( const std::string &file_name, cv::Size &image_size,
                       cv::Mat &camera_matrix, cv::Mat &dist_coeffs, double& baseline );
void saveStereoCameraParams( const std::string &file_name, const cv::Size &image_size,
                       const cv::Mat &camera_matrix, const cv::Mat &dist_coeffs, const double baseline );