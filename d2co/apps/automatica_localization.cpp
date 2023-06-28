#include "automatica_localization.h"

#include <utility>

static const double OUTER_POINT_SCORE = 0.6;

bool objIstanceCompareScore(ObjIstance o1,ObjIstance o2) { return (o1.score > o2.score); }
bool objIstanceCompareZ(ObjIstance o1,ObjIstance o2) { return (o1.t_vec(2) < o2.t_vec(2)); }


void normalizeMatch( TemplateMatch &m, RasterObjectModel3D &obj_model )
{
  if( m.img_offset == Point(0,0) )
    return;
  bool roi_enabled = false;
  if( obj_model.cameraModel().regionOfInterestEnabled() )
  {
    roi_enabled = true;
    obj_model.cameraModel().enableRegionOfInterest(false);
  }
  
  obj_model.setModelView(m.r_quat, m.t_vec);
  vector <Point3f> obj_pts = obj_model.getPoints();
  vector <Point2f> proj_pts;
  obj_model.projectRasterPoints( proj_pts );
  obj_model.cameraModel().enableRegionOfInterest(roi_enabled);
  
  Point2f off_p(m.img_offset.x , m.img_offset.y );

  for( auto &p : proj_pts )
    p += off_p;
  
  cv::Mat r_vec(3,1,cv::DataType<double>::type);
  cv::Mat t_vec(3,1,cv::DataType<double>::type);
  
  cv::solvePnP( obj_pts, proj_pts, obj_model.cameraModel().cameraMatrix(), 
                obj_model.cameraModel().distorsionCoeff(), r_vec, t_vec );
  
  double r_quat[4];
  ceres::AngleAxisToQuaternion( (const double*)(r_vec.data), r_quat) ;
  m.r_quat = Eigen::Quaterniond( r_quat[0], r_quat[1], r_quat[2], r_quat[3] );
  m.t_vec = Eigen::Map< const Eigen::Vector3d>( (const double*)(t_vec.data) );
  m.img_offset = Point(0,0);
}

AutomaticaLocalization::AutomaticaLocalization ( const PinholeCameraModel cam_models[2], const Mat& stereo_r_mat, 
                                                 const Mat& stereo_t_vec, Rect roi[2] )

{
  stereo_rect_.setCameraParameters ( cam_models, stereo_r_mat, stereo_t_vec );
  stereo_rect_.update();
  stereo_rect_.getCamModels ( rect_cam_models_ );
  stereo_rect_.getCamDisplacement( stereo_disp_ );
  
  for( int k = 0; k < 2; k++ )
  {
    roi_[k] = roi[k];
    rect_cam_models_[k].setRegionOfInterest(roi[k]);
  }
  stereo_bb_offset_ = roi[0].x - roi[1].y;
}

void AutomaticaLocalization::addObj ( string model_filename, string template_filename[2], double threshold )
{
  cout << "Loading model from file : "<<model_filename<< endl;  

  obj_model_ptrs_.push_back(vector < RasterObjectModel3DPtr >());
  vector < RasterObjectModel3DPtr > &obj_model_vec = obj_model_ptrs_.back();
  for( int i = 0; i < 2; i++ )
  {
    obj_model_vec.push_back( RasterObjectModel3DPtr( new RasterObjectModel3D() ) );
    RasterObjectModel3D &obj_model = *(obj_model_vec.back());
    obj_model.setCamModel( rect_cam_models_[i] );
    if(!obj_model.setModelFile( model_filename ) )
      return;
    obj_model.computeRaster();
  }
  
  ts_ptrs_.push_back(vector < TemplateSetPtr >());
  std::vector < TemplateSetPtr > &ts_ptr = ts_ptrs_.back();
  ts_ptr.resize(2);
  for(int k = 0; k < 2; k++ )
  {
    ts_ptr[k] = make_shared<TemplateSet>();
    cout << "Loading templates from file : "<<template_filename[k]<< endl;
    readFromFile(template_filename[k],*ts_ptr[k]);
  }
  
  thresholds_.push_back(threshold);
}

void AutomaticaLocalization::initialize()
{
  // WARNING UGLY WORKAROUND
  for( auto &obj_model : obj_model_ptrs_ )
    for( int k = 0; k < 2; k++ )
      obj_model[k]->cameraModel().enableRegionOfInterest(true);
    
  dc_.setDistThreshold(30);
  dc_.enableParallelism(true);
  
  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
  edge_detector_ptr->setLowThreshold(40);
  edge_detector_ptr->setRatio(2);
  edge_detector_ptr->enableRGBmodality(true);

  dc_.setEdgeDetector(std::move(edge_detector_ptr));

  dcm_.resize(obj_model_ptrs_.size());
  mdcm_.resize(obj_model_ptrs_.size());
  for( int i = 0; i < int(obj_model_ptrs_.size()); i++ )
  {
    dcm_[i] = make_shared<DirectionalChamferMatching>();
    dcm_[i]->setTemplateModel(obj_model_ptrs_[i][0]);
    dcm_[i]->setNumDirections(num_directions);
    dcm_[i]->enableParallelism(true);
    dcm_[i]->setTemplates( ts_ptrs_[i][0] );
    
    mdcm_[i] = make_shared<MultiViewsDirectionalChamferMatching>();
    mdcm_[i]->setNumDirections(num_directions);
    mdcm_[i]->setTemplateModel(obj_model_ptrs_[i][0]);
    mdcm_[i]->enableParallelism(true);
  }
  
  Size img_size = rect_cam_models_[0].imgSize();
  
  
  img_pair_ = Mat( Size ( 2*img_size.width, img_size.height ), cv::DataType<Vec3b>::type );
  display_ = Mat( Size ( 2*img_size.width, img_size.height ), cv::DataType<Vec3b>::type );
  score_mask_ = Mat( Size ( img_size.width, img_size.height ), cv::DataType<uchar>::type, Scalar(255) );
  
  int start_col = 0, w = img_size.width;
  for ( int k = 0; k < 2; k++, start_col += w )
  {
    rect_img_[k] = img_pair_.colRange ( start_col, start_col + w );
    h_display_[k] = display_.colRange ( start_col, start_col + w );
  }
}

void AutomaticaLocalization::localize ( Mat src_img[2], vector< ObjIstance >& found_obj )
{

  std::vector< std::vector< TemplateMatch > > matches(obj_model_ptrs_.size());
  stereo_rect_.rectifyImagePair ( src_img, rect_img_ );
  
  Mat img_roi[2], proc_img[2];
  
  cout<<"Edge detection"<<endl;
  cv_ext::BasicTimer timer;
  for ( int k = 0; k < 2; k++ )
  {
    img_roi[k] = rect_img_[k](roi_[k]);
    bilateralFilter(img_roi[k], proc_img[k], -1, 50, 5);
  }
  cout<<"Elapsed time : "<<timer.elapsedTimeMs()<<endl;

  ImageTensorPtr dst_map_tensor_ptrs[2];
  for ( int k = 0; k < 2; k++ )
    dc_.computeDistanceMapTensor ( proc_img[k], dst_map_tensor_ptrs[k], num_directions, tensor_lambda, smooth_tensor);
    

  vector< pair < vector<Point2f>, vector<Point2f> > > proj_pts_vec;
  vector< pair < vector<Point>, vector<Point>  > > templ_proj_pts;
  for( int i_obj = 0; i_obj < int(obj_model_ptrs_.size()); i_obj++ )
  {

    cout<<"Looking for "<<i_obj<<endl;
    timer.reset();

    dcm_[i_obj]->setInput( dst_map_tensor_ptrs[0] );
    dcm_[i_obj]->match( num_matches, matches[i_obj], (int)increment);

    cout<<"DCM elapsed time ms : "<<timer.elapsedTimeMs()<<endl;
    
    int i_m = 0;    
    for( auto iter = matches[i_obj].begin(); iter != matches[i_obj].end(); iter++, i_m++ )
    {
      TemplateMatch &match = *iter;    

      
      vector<int> right_offset;      

//       templ_proj_pts[i_m].first = ts_ptrs_[i_obj][0]->pts[match.id]; 
//       templ_proj_pts[i_m].second = ts_ptrs_[i_obj][1]->pts[match.id]; 
//       
//       for( auto &p : templ_proj_pts[i_m].first )
//         p += match.img_offset;
      
      cout<<"Not update source!"<<endl;
      exit( -1 );
      normalizeMatch( match, *(obj_model_ptrs_[i_obj][0]) );
      
      timer.reset();
      refinePosition(i_obj, match.r_quat, match.t_vec, dst_map_tensor_ptrs);
      cout<<"refinePosition ms : "<<timer.elapsedTimeMs()<<endl;
      
      if( match.t_vec(2) >= 1.09 && match.t_vec(2) < 1.2 )
      {
        proj_pts_vec.push_back(pair < vector<Point2f>, vector<Point2f> >());
        pair < vector<Point2f>, vector<Point2f> > &proj_pts = proj_pts_vec.back();
        
        obj_model_ptrs_[i_obj][0]->setModelView(match.r_quat, match.t_vec);
        vector<float> normals;
        obj_model_ptrs_[i_obj][0]->projectRasterPoints(proj_pts.first, normals);
        
        double score = evaluateScore ( proc_img[0], proj_pts.first, normals );
              
        Eigen::Quaterniond right_r_quat = match.r_quat;
        Eigen::Vector3d right_t_vec = match.t_vec;
  
        right_t_vec(0) += stereo_disp_.x;
        right_t_vec(1) += stereo_disp_.y;
        
        obj_model_ptrs_[i_obj][1]->setModelView(right_r_quat, right_t_vec);
        normals.clear();
        obj_model_ptrs_[i_obj][1]->projectRasterPoints(proj_pts.second, normals);      
        score += evaluateScore ( proc_img[1], proj_pts.second, normals );
        score/=2;
        
        if( score >= thresholds_[i_obj] )
          found_obj.push_back(ObjIstance(i_obj, score, match.r_quat, match.t_vec));       
      }
    }
  }
  
  std::sort (found_obj.begin(), found_obj.end(), objIstanceCompareScore);
  for( int i = 0; i < found_obj.size(); i++ )
  {
    img_pair_.copyTo(display_);
    Mat draw_img[2];
    for( int k = 0; k < 2; k++ )
    {
      cv::Point dbg_tl = roi_[k].tl(), dbg_br = roi_[k].br();
      dbg_tl.x -= 1; dbg_tl.y -= 1;
      dbg_br.x += 1; dbg_br.y += 1;
      cv::rectangle( h_display_[k], dbg_tl, dbg_br, cv::Scalar(255,255,255));
      draw_img[k] = h_display_[k](roi_[k]);
    }
//     cv_ext::drawCircles( draw_img[0], templ_proj_pts[i].first, 5, Scalar(0,255,0) );
//     cv_ext::drawCircles( draw_img[1], templ_proj_pts[i].second, 5, Scalar(0,255,0) );
    
    cv_ext::drawPoints( draw_img[0], proj_pts_vec[i].first, Scalar(0,0,255) );
    cv_ext::drawPoints( draw_img[1], proj_pts_vec[i].second, Scalar(0,0,255) );
    
    cout<<"Score : "<<found_obj[i].score<<" Z : "<<found_obj[i].t_vec(2)<<endl;
    cv_ext::showImage(display_,"display");   
  }
}

void AutomaticaLocalization::refinePosition ( int i_obj, Eigen::Quaterniond& r_quat, 
                                              Eigen::Vector3d& t_vec, ImageTensorPtr dst_map_tensor_ptrs[2] )
{
  Eigen::Quaterniond tmp_r_quat = r_quat;
  Eigen::Vector3d tmp_t_vec = t_vec;

  dcm_[i_obj]->refinePosition(tmp_r_quat, tmp_t_vec);

  if( tmp_t_vec(2) >= 1.09 && tmp_t_vec(2) < 1.2 )
  {
    r_quat = tmp_r_quat;
    t_vec = tmp_t_vec;     
    
    MultiViewsInputVec mv_vec(2);
    Eigen::Affine3d transf = Eigen::Affine3d::Identity();
    
    mv_vec[0].model_ptr = obj_model_ptrs_[i_obj][0];
    mv_vec[0].views.push_back(transf);
    mv_vec[0].dist_map_tensor_ptr_vec.push_back(dst_map_tensor_ptrs[0]);
    
    
    transf.matrix()(0,3) += stereo_disp_.x;
    transf.matrix()(1,3) += stereo_disp_.y;
    
    mv_vec[1].model_ptr = obj_model_ptrs_[i_obj][1];
    mv_vec[1].views.push_back(transf);
    mv_vec[1].dist_map_tensor_ptr_vec.push_back(dst_map_tensor_ptrs[1]);
    
    
    mdcm_[i_obj]->setInput(mv_vec);
    cout<<"Refine position : "<<mdcm_[i_obj]->refinePosition(tmp_r_quat, tmp_t_vec)<<endl;
    if( tmp_t_vec(2) >= 1.09 && tmp_t_vec(2) < 1.2 )
    {
       cout<<"REF STEREO"<<endl<<tmp_t_vec<<endl;
       r_quat = tmp_r_quat;
       t_vec = tmp_t_vec; 
    }
    else
    {
      cout<<"REF MONO "<<endl;
    }
  }
  else
    cout<<"REF NONE"<<endl;
  
}


double AutomaticaLocalization::evaluateScore ( cv::Mat img,
                                               vector<cv::Point2f> &raster_pts,
                                               const vector<float> &normal_directions )
{
  vector<cv::Point2f> pts;
  vector<float> ndir;
  pts.reserve(raster_pts.size());
  int i_p = 0;
  for( auto &p : raster_pts )
  {
    int x = cvRound(p.x), y = cvRound(p.y);
    if( score_mask_.at<uchar>(y,x) )
    {
      pts.push_back(p);
      ndir.push_back(normal_directions[i_p]);
      score_mask_.at<uchar>(y,x) = 0;
    }
    i_p++;
  }
  
  cout<<"Bfore : "<<raster_pts.size()<<" after : "<<pts.size()<<endl;
  cv_ext::ImageStatisticsPtr img_stats_p = cv_ext::ImageStatistics::createImageStatistics ( img, true );
  boost::shared_ptr< vector<float> > g_dir_p =  img_stats_p->getGradientDirections ( pts );
  boost::shared_ptr< vector<float> > g_mag_p =  img_stats_p->getGradientMagnitudes ( pts );
  
  vector<float> &g_dir = *g_dir_p;
  vector<float> &g_mag = *g_mag_p;

  if ( !g_dir.size() || !g_mag_p->size() )
    return 0;
  
  

  double score = 0;
  for ( int i = 0; i < int(g_dir.size()); i++ )
  {
    float &direction = g_dir[i], magnitude = g_mag[i];
    if ( img_stats_p->outOfImage ( direction ) )
      score += OUTER_POINT_SCORE;
    else
      score += ((magnitude > 0.01)?1.0:magnitude ) * abs ( cos ( double ( direction ) - ndir[i] ) );
  }
  
  for( auto &p : pts )
  {
    int x = cvRound(p.x), y = cvRound(p.y);
    score_mask_.at<uchar>(y,x) = 255;
  }  

  return score/g_dir.size();
}