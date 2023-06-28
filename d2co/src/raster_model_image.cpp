#include "raster_model_image.h"

int RasterModelImage::getQuantizedOrientation(float dir, int n0)
{
  if (dir<0) dir+=M_PI;
  //else if(dir>M_PI) dir -= M_PI;
  float alfa=M_PI/n0;
  for (int i=0; i<n0; i++){
    if (dir<=alfa*(i+1)) 
    {
      return i;
    }
  }
}

void RasterModelImage::computeNormalsQuantization(const std::vector<float> &normals, std::vector<unsigned char> &q_normals)
{
  q_normals.resize(normals.size());
  for(int i=0; i < int(normals.size()); i++){
    q_normals[i]=(unsigned char)getQuantizedOrientation(normals[i],8);
  }
}

void RasterModelImage::storeModel(const Eigen::Quaterniond &r, const Eigen::Vector3d &t, std::vector<cv::Point2f> &pts, std::vector<float> &normals)
{
  Eigen::Quaternionf rot(r);
  precomputed_rq_.push_back(rot);
  Eigen::Vector3f tr((double)t(0), (double)t(1), (double)t(2));
  precomputed_t_.push_back(tr);
  precomputed_pts_.push_back(pts);
  precomputed_normals_.push_back(normals);
  std::vector<unsigned char> q_normals;
  computeNormalsQuantization(normals, q_normals);
  precomputed_quantized_normals_.push_back(q_normals);
}

void RasterModelImage::storeModelsFromOther(RasterModelImagePtr &other)
{
  precomputed_pts_=other->precomputed_pts_;
  precomputed_normals_=other->precomputed_normals_;
  precomputed_quantized_normals_=other->precomputed_quantized_normals_;
  precomputed_rq_=other->precomputed_rq_;
  precomputed_t_=other->precomputed_t_;
}

void RasterModelImage::getModel(int idx, Eigen::Quaterniond &r, Eigen::Vector3d &t, std::vector<cv::Point2f> &pts, std::vector<float> &normals)
{
  r.x()=(double)precomputed_rq_[idx].x();
  r.y()=(double)precomputed_rq_[idx].y();
  r.z()=(double)precomputed_rq_[idx].z();
  r.w()=(double)precomputed_rq_[idx].w();
  t(0)=(double)precomputed_t_[idx](0);
  t(1)=(double)precomputed_t_[idx](1);
  t(2)=(double)precomputed_t_[idx](2);
  pts=precomputed_pts_[idx];
  normals=precomputed_normals_[idx];
}

void RasterModelImage::getModel(int idx, Eigen::Quaterniond &r, Eigen::Vector3d &t, std::vector<cv::Point2f> &pts, std::vector<unsigned char> &quantized_normals)
{
  r.x()=(double)precomputed_rq_[idx].x();
  r.y()=(double)precomputed_rq_[idx].y();
  r.z()=(double)precomputed_rq_[idx].z();
  r.w()=(double)precomputed_rq_[idx].w();
  t(0)=(double)precomputed_t_[idx](0);
  t(1)=(double)precomputed_t_[idx](1);
  t(2)=(double)precomputed_t_[idx](2);
  pts=precomputed_pts_[idx];
  quantized_normals=precomputed_quantized_normals_[idx];
}

void RasterModelImage::saveModels(std::string base_name, int step_points)
{
  cv::Mat model_views(precomputed_rq_.size(), 7, cv::DataType<float>::type );
  for( int i = 0; i < int(precomputed_rq_.size()); i++ )
  {
    model_views.at<float>(i,0) = precomputed_rq_[i].w();
    model_views.at<float>(i,1) = precomputed_rq_[i].x();
    model_views.at<float>(i,2) = precomputed_rq_[i].y();
    model_views.at<float>(i,3) = precomputed_rq_[i].z();
    model_views.at<float>(i,4) = precomputed_t_[i](0);
    model_views.at<float>(i,5) = precomputed_t_[i](1);
    model_views.at<float>(i,6) = precomputed_t_[i](2);
  }
  
  std::stringstream sname;
  sname<<base_name<<"_image_model_views.yml";
  cv::FileStorage fs(sname.str().data(), cv::FileStorage::WRITE);
  
  fs << "views" << model_views;
  
  for( int i = 0; i < int(precomputed_pts_.size()); i++ )
  {
    const std::vector<cv::Point2f> &pts = precomputed_pts_[i];
    const std::vector<float> &q_normals = precomputed_normals_[i];
    cv::Mat model_views_pts(precomputed_pts_[i].size()/step_points, 2, cv::DataType<float>::type );
    for( int j = 0, k=0; j < model_views_pts.rows, k < int(precomputed_pts_[i].size()); j++, k+=step_points)
    {
      model_views_pts.at<float>(j,0) = pts[k].x;
      model_views_pts.at<float>(j,1) = pts[k].y;
    }   
    std::stringstream sname2;
    sname2<<"pts_"<<i;
    fs << sname2.str().data() << model_views_pts;
    
    cv::Mat model_views_norm(precomputed_normals_[i].size()/step_points, 1, cv::DataType<float>::type );
    for( int j = 0, k=0; j < model_views_norm.rows, k<precomputed_normals_[i].size(); j++, k+=step_points)
    {
      model_views_norm.at<float>(j,0) = q_normals[k];
    }   
    sname2.clear();
    sname2<<"norm_"<<i;
    fs << sname2.str().data() << model_views_norm;
  }
  
  fs.release();
}

void RasterModelImage::loadModels(std::string base_name)
{
  precomputed_rq_.clear();
  precomputed_t_.clear();
 
  std::stringstream sname;
  sname<<base_name<<"_image_model_views.yml";
  cv::FileStorage fs(sname.str().data(), cv::FileStorage::READ);
  
  cv::Mat model_views;
  fs["views"] >> model_views;
  
  for( int i = 0; i< model_views.rows; i++ )
  {
    Eigen::Quaternionf quat(model_views.at<float>(i,0),
                            model_views.at<float>(i,1),
                            model_views.at<float>(i,2),
                            model_views.at<float>(i,3));
    Eigen::Vector3f t(model_views.at<float>(i,4),
                      model_views.at<float>(i,5),
                      model_views.at<float>(i,6));
    
    precomputed_rq_.push_back(quat);
    precomputed_t_.push_back(t);
  }

  precomputed_pts_.clear();
  precomputed_normals_.clear();
  
  int n_precomputed_views = precomputed_rq_.size();
  
  for( int i = 0; i < n_precomputed_views; i++ )
  {
    precomputed_pts_.push_back(std::vector<cv::Point2f>());
    precomputed_normals_.push_back(std::vector<float>());
    
    cv::Mat model_views_pts, model_views_norm;
    std::stringstream sname2;
    sname2<<"pts_"<<i;
    fs[sname2.str().data()] >> model_views_pts;
   
    for( int j = 0; j < model_views_pts.rows; j++ )
    {
      cv::Point2f pt;
      pt.x = model_views_pts.at<float>(j,0);
      pt.y = model_views_pts.at<float>(j,1);
      
      precomputed_pts_[i].push_back(pt);
    }
    
    sname2.clear();
    sname2<<"norm_"<<i;
    fs[sname2.str().data()] >> model_views_norm;
   
    for( int j = 0; j < model_views_norm.rows; j++ )
    {
      float pt;
      pt = model_views_norm.at<float>(j,0);
      
      precomputed_normals_[i].push_back(pt);
    }
  } 

  fs.release();
}

void RasterModelImage::saveModelsQuantizedNormals(std::string base_name, int step_points)
{
  cv::Mat model_views(precomputed_rq_.size(), 7, cv::DataType<float>::type );
  for( int i = 0; i < int(precomputed_rq_.size()); i++ )
  {
    model_views.at<float>(i,0) = precomputed_rq_[i].w();
    model_views.at<float>(i,1) = precomputed_rq_[i].x();
    model_views.at<float>(i,2) = precomputed_rq_[i].y();
    model_views.at<float>(i,3) = precomputed_rq_[i].z();
    model_views.at<float>(i,4) = precomputed_t_[i](0);
    model_views.at<float>(i,5) = precomputed_t_[i](1);
    model_views.at<float>(i,6) = precomputed_t_[i](2);
  }
  
  std::stringstream sname;
  sname<<base_name<<"_image_model_views.yml";
  cv::FileStorage fs(sname.str().data(), cv::FileStorage::WRITE);
  
  fs << "views" << model_views;
  
  for( int i = 0; i < int(precomputed_pts_.size()); i++ )
  {
    const std::vector<cv::Point2f> &pts = precomputed_pts_[i];
    const std::vector<unsigned char> &q_normals = precomputed_quantized_normals_[i];
    cv::Mat model_views_pts(precomputed_pts_[i].size()/step_points, 2, cv::DataType<float>::type );
    for( int j = 0, k=0; j < model_views_pts.rows, k<precomputed_pts_[i].size(); j++, k+=step_points)
    {
      model_views_pts.at<float>(j,0) = pts[k].x;
      model_views_pts.at<float>(j,1) = pts[k].y;
    }   
    std::stringstream sname2;
    sname2<<"pts_"<<i;
    fs << sname2.str().data() << model_views_pts;
    
    cv::Mat model_views_norm(precomputed_quantized_normals_[i].size()/step_points, 1, cv::DataType<unsigned char>::type );
    for( int j = 0, k=0; j < model_views_norm.rows, k<precomputed_quantized_normals_[i].size(); j++, k+=step_points)
    {
      model_views_norm.at<unsigned char>(j,0) = q_normals[k];
    }   
    sname2.clear();
    sname2<<"norm_"<<i;
    fs << sname2.str().data() << model_views_norm;
  }
  
  fs.release();
}

void RasterModelImage::loadModelsQuantizedNormals(std::string base_name)
{
  precomputed_rq_.clear();
  precomputed_t_.clear();
 
  std::stringstream sname;
  sname<<base_name<<"_image_model_views.yml";
  cv::FileStorage fs(sname.str().data(), cv::FileStorage::READ);
  
  cv::Mat model_views;
  fs["views"] >> model_views;
  
  for( int i = 0; i< model_views.rows; i++ )
  {
    Eigen::Quaternionf quat(model_views.at<float>(i,0),
                            model_views.at<float>(i,1),
                            model_views.at<float>(i,2),
                            model_views.at<float>(i,3));
    Eigen::Vector3f t(model_views.at<float>(i,4),
                      model_views.at<float>(i,5),
                      model_views.at<float>(i,6));
    
    precomputed_rq_.push_back(quat);
    precomputed_t_.push_back(t);
  }

  precomputed_pts_.clear();
  precomputed_quantized_normals_.clear();
  
  int n_precomputed_views = precomputed_rq_.size();
  
  for( int i = 0; i < n_precomputed_views; i++ )
  {
    precomputed_pts_.push_back(std::vector<cv::Point2f>());
    precomputed_quantized_normals_.push_back(std::vector<unsigned char>());
    
    cv::Mat model_views_pts, model_views_norm;
    std::stringstream sname2;
    sname2<<"pts_"<<i;
    fs[sname2.str().data()] >> model_views_pts;
   
    for( int j = 0; j < model_views_pts.rows; j++ )
    {
      cv::Point2f pt;
      pt.x = model_views_pts.at<float>(j,0);
      pt.y = model_views_pts.at<float>(j,1);
      
      precomputed_pts_[i].push_back(pt);
    }
    
    sname2.clear();
    sname2<<"norm_"<<i;
    fs[sname2.str().data()] >> model_views_norm;
   
    for( int j = 0; j < model_views_norm.rows; j++ )
    {
      unsigned char pt;
      pt = model_views_norm.at<unsigned char>(j,0);
      
      precomputed_quantized_normals_[i].push_back(pt);
    }
  } 

  fs.release();
}




