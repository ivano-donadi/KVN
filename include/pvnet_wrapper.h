#pragma once

#include <string>
#include <memory>
#include <cv_ext/cv_ext.h>

class PVNetWrapper
{
 public:
  PVNetWrapper( const std::string &python_root );
  ~PVNetWrapper();

  bool registerObject( int obj_id, const std::string &model_fn, const std::string &inference_meta_fn );
  bool localize( cv::Mat &img, int obj_id, cv::Mat_<double> &r_mat, cv::Mat_<double> &t_vec );

 private:
  class Impl;
  std::unique_ptr<Impl> pimpl_;
};