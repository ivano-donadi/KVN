#include "tm_object_localization.h"

#include "chamfer_matching.h"
#include "chamfer_registration.h"
#include "scoring.h"
#include "io_utils.h"

using namespace std;
using namespace cv;
using namespace cv_ext;

class TMObjectLocalization::Impl
{
 public:

  ~Impl() = default;

  bool initialized = false;

  cv_ext::PinholeCameraModel cam_model;
  RasterObjectModel3DPtr obj_model_ptr;
  DirIdxPointSetVecPtr ts_ptr;
  DistanceTransform dt;
  DirectionalChamferMatching dcm;
  DirectionalChamferRegistration dcr;
  ObjectTemplatePnP o_pnp;

  double scale_factor = 1.0;
  double model_samplig_step = 0.6;
  int canny_low_th = 40;
  double score_th = .6;
  RasterObjectModel3D::UoM unit = RasterObjectModel3D::UoM::CENTIMETER;
  cv::Rect roi;
  double bb_xoff, bb_yoff, bb_zoff, bb_woff, bb_hoff, bb_doff;
  bool has_roi = false;
  bool has_bb_offset = false;
  cv::Mat_<double> r_vec, t_vec;
  bool is_tracking = false;
  int last_match_id = -1;
  int matching_step = 4, num_matches = 5, match_cell_size = -1;
  bool display_enabled = false;
  cv::Mat display;
};

TMObjectLocalization::TMObjectLocalization() : pimpl_(new TMObjectLocalization::Impl()){}
TMObjectLocalization::~TMObjectLocalization() = default;

void TMObjectLocalization::setScaleFactor(double s)
{
  pimpl_->scale_factor = s;
}

void TMObjectLocalization::setCannyLowThreshold(int th)
{
  pimpl_->canny_low_th = th;
}

void TMObjectLocalization::setModelSaplingStep(double step)
{
  pimpl_->model_samplig_step = step;
}


void TMObjectLocalization::setScoreThreshold( double th )
{
  pimpl_->score_th = th;
}

void TMObjectLocalization::setUnitOfMeasure(RasterObjectModel3D::UoM unit)
{
  pimpl_->unit = unit;
}

void TMObjectLocalization::setRegionOfInterest( const cv::Rect &roi )
{
  pimpl_->roi = roi;
  pimpl_->has_roi = true;
}

void TMObjectLocalization::setBoundingBoxOffset(double xoff, double yoff, double zoff,
                                                double woff, double hoff, double doff)
{
  pimpl_->bb_xoff = xoff;
  pimpl_->bb_yoff = yoff;
  pimpl_->bb_zoff = zoff;
  pimpl_->bb_woff = woff;
  pimpl_->bb_hoff = hoff;
  pimpl_->bb_doff = doff;

  pimpl_->has_bb_offset = true;
}

void TMObjectLocalization::setNumMatches( int n )
{
  pimpl_->num_matches = n;
}

void TMObjectLocalization::enableDisplay( bool enabled )
{
  pimpl_->display_enabled = enabled;
}

bool TMObjectLocalization::initialize( std::string cam_model_fn,
                                       std::string cad_model_fn,
                                       std::string templates_fn )
{
  pimpl_->is_tracking = false;

  cv::Size image_size;
  cv::Mat camera_matrix, dist_coeffs;
  if( !loadCameraParams( cam_model_fn, image_size, camera_matrix, dist_coeffs ) )
  {
    cout << "Error loading camera filename, exiting"<< endl;
    return false;
  }

  pimpl_->cam_model = cv_ext::PinholeCameraModel(camera_matrix, image_size.width, image_size.height, dist_coeffs );
  pimpl_->cam_model.setSizeScaleFactor(pimpl_->scale_factor);
//  Size scaled_img_size = pimpl_->cam_model.imgSize();

  if( pimpl_->has_roi )
  {
    // TODO
    pimpl_->cam_model.setRegionOfInterest(pimpl_->roi);
    pimpl_->cam_model.enableRegionOfInterest(true);
  }

  pimpl_->obj_model_ptr = make_shared<RasterObjectModel3D>();
  pimpl_->obj_model_ptr->setCamModel( pimpl_->cam_model );
  pimpl_->obj_model_ptr->setStepMeters(pimpl_->model_samplig_step);
  pimpl_->obj_model_ptr->setRenderZNear(1);
  pimpl_->obj_model_ptr->setRenderZFar(25);
  pimpl_->obj_model_ptr->setUnitOfMeasure ( pimpl_->unit );

  pimpl_->obj_model_ptr->enableUniformColor(cv::Scalar(255,0,0));

  if(!pimpl_->obj_model_ptr->setModelFile( cad_model_fn ) )
  {
    cout << "Unable to read model file: existing" << endl;
    return false;
  }

  pimpl_->obj_model_ptr->computeRaster();

  if( pimpl_->has_bb_offset )
  {
    auto cur_bb = pimpl_->obj_model_ptr->getBoundingBox();

    cur_bb.x += pimpl_->bb_xoff;
    cur_bb.width += pimpl_->bb_woff - pimpl_->bb_xoff;
    cur_bb.y += pimpl_->bb_yoff;
    cur_bb.height += pimpl_->bb_hoff - pimpl_->bb_yoff;
    cur_bb.z += pimpl_->bb_zoff;
    cur_bb.depth += pimpl_->bb_doff - pimpl_->bb_zoff;

    // TODO
    pimpl_->obj_model_ptr->setBoundingBox(cur_bb);
  }

  pimpl_->ts_ptr = make_shared<DirIdxPointSetVec>();
  loadTemplateVector(templates_fn, *(pimpl_->ts_ptr) );

  pimpl_->dt.setDistThreshold(30);
  pimpl_->dt.enableParallelism(true);

  CannyEdgeDetectorUniquePtr edge_detector_ptr( new CannyEdgeDetector() );
  edge_detector_ptr->setLowThreshold(pimpl_->canny_low_th);
  edge_detector_ptr->setRatio(2);
  edge_detector_ptr->enableRGBmodality(true);

  pimpl_->dt.setEdgeDetector(std::move(edge_detector_ptr));

  pimpl_->dcm.enableParallelism(true);
  pimpl_->dcm.setTemplatesVector( pimpl_->ts_ptr );

  pimpl_->dcr.setNumDirections(60);
  pimpl_->dcr.setObjectModel(pimpl_->obj_model_ptr);

  pimpl_->o_pnp.setCamModel(pimpl_->cam_model);
  pimpl_->o_pnp.fixZTranslation(true);

  pimpl_->initialized = true;

  if( pimpl_->display_enabled )
    pimpl_->display.create(pimpl_->cam_model.imgSize(), cv::DataType<Vec3b>::type );

  return true;
}

RasterObjectModel3DPtr TMObjectLocalization::objectModel()
{
  return pimpl_->obj_model_ptr;
}

bool TMObjectLocalization::localize( cv::Mat src_img )
{
  if( !pimpl_->initialized )
    return false;

  GradientDirectionScore scoring;
  vector< TemplateMatch > matches;

  cv::Mat resized_img, img_roi;

//  cv_ext::BasicTimer timer;

  if( src_img.empty() )
    return false;

  if( pimpl_->scale_factor != 1 )
    cv::resize(src_img, resized_img, pimpl_->cam_model.imgSize());
  else
    resized_img = src_img.clone();

  // TODO Enable region of interest
  if( false /*has_roi*/ )
    img_roi = resized_img(pimpl_->cam_model.regionOfInterest());
  else
    img_roi = resized_img;

//  timer.reset();
  const int num_directions = 60;
  const double tensor_lambda = 6.0;
  const bool smooth_tensor = false;

  ImageTensorPtr dst_map_tensor_ptr = std::make_shared<ImageTensor>();
  pimpl_->dt.computeDistanceMapTensor( img_roi, *dst_map_tensor_ptr, num_directions, tensor_lambda, smooth_tensor);

//  cout << "Tensor computation ms: " << timer.elapsedTimeMs() << endl;

  ImageGradient im_grad( img_roi );

  pimpl_->dcr.setInput(dst_map_tensor_ptr);

  if( pimpl_->is_tracking )
  {
    pimpl_->dcr.refinePosition((*(pimpl_->ts_ptr))[pimpl_->last_match_id], pimpl_->r_vec, pimpl_->t_vec);

    vector<Point2f> refined_proj_pts;
    vector<float> normals;
    pimpl_->obj_model_ptr->setModelView(pimpl_->r_vec, pimpl_->t_vec);
    pimpl_->obj_model_ptr->projectRasterPoints(refined_proj_pts, normals);
    double score = scoring.evaluate(im_grad, refined_proj_pts, normals);
    if( score >= pimpl_->score_th )
    {
      if( pimpl_->display_enabled )
      {
        Mat draw_img;
        resized_img.copyTo(pimpl_->display);
        // TODO Enable region of interest
        if (false /*has_roi*/)
        {
          cv::Rect cur_roi = pimpl_->cam_model.regionOfInterest();
          cv::Point dbg_tl = cur_roi.tl(), dbg_br = cur_roi.br();
          dbg_tl.x -= 1;
          dbg_tl.y -= 1;
          dbg_br.x += 1;
          dbg_br.y += 1;
          cv::rectangle(pimpl_->display, dbg_tl, dbg_br, cv::Scalar(255, 255, 255));
          draw_img = pimpl_->display(cur_roi);
        }
        else
        {
          draw_img = pimpl_->display;
        }
        cv::putText(draw_img,"Tracking",cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
        cv_ext::drawPoints(draw_img, refined_proj_pts, Scalar(0, 0, 255));
        cv_ext::showImage(pimpl_->display, "display", true, 10);
      }
      return true;
    }
    else
    {
      pimpl_->is_tracking = false;
    }
  }


//  timer.reset();

  pimpl_->dcm.setInput(dst_map_tensor_ptr);
  pimpl_->dcm.match( pimpl_->num_matches, matches,
                     pimpl_->matching_step,
                     pimpl_->match_cell_size );

  vector<Mat_<double> > r_vecs, t_vecs;
  vector<int> match_ids;
  multimap<double, int> scores;

  int i_m = 0;
  for (auto iter = matches.begin(); iter != matches.end(); iter++, i_m++)
  {
    TemplateMatch &match = *iter;

    r_vecs.push_back(Mat_<double>(3, 1));
    t_vecs.push_back(Mat_<double>(3, 1));

    pimpl_->o_pnp.solve((*(pimpl_->ts_ptr))[match.id], match.img_offset, r_vecs.back(), t_vecs.back());
    pimpl_->dcr.refinePosition((*(pimpl_->ts_ptr))[match.id], r_vecs.back(), t_vecs.back());

    vector<Point2f> refined_proj_pts;
    vector<float> normals;
    pimpl_->obj_model_ptr->setModelView(r_vecs.back(), t_vecs.back());
    pimpl_->obj_model_ptr->projectRasterPoints(refined_proj_pts, normals);
    double score = scoring.evaluate(im_grad, refined_proj_pts, normals);
    scores.insert(std::pair<double, int>(score, i_m));
    match_ids.push_back(match.id);
  }

//  cout << "DCR object registration and scoring elapsed time ms : " << timer.elapsedTimeMs() << endl;

  double score = scores.rbegin()->first;
  int idx = scores.rbegin()->second;
  std::cout<<"Score : "<<score<<std::endl;

  if( score >= pimpl_->score_th )
  {
    pimpl_->r_vec = r_vecs[idx].clone();
    pimpl_->t_vec = t_vecs[idx].clone();
    pimpl_->last_match_id = match_ids[idx];
    pimpl_->is_tracking = true;

    if( pimpl_->display_enabled )
    {
      Mat draw_img;
      resized_img.copyTo(pimpl_->display);
      // TODO Enable region of interest
      if (false /*has_roi*/)
      {
        cv::Rect cur_roi = pimpl_->cam_model.regionOfInterest();
        cv::Point dbg_tl = cur_roi.tl(), dbg_br = cur_roi.br();
        dbg_tl.x -= 1;
        dbg_tl.y -= 1;
        dbg_br.x += 1;
        dbg_br.y += 1;
        cv::rectangle(pimpl_->display, dbg_tl, dbg_br, cv::Scalar(255, 255, 255));
        draw_img = pimpl_->display(cur_roi);
      }
      else
      {
        draw_img = pimpl_->display;
      }

      vector<Point2f> refined_proj_pts;
      pimpl_->obj_model_ptr->setModelView(r_vecs[idx], t_vecs[idx]);
      pimpl_->obj_model_ptr->projectRasterPoints(refined_proj_pts);
      cv_ext::drawPoints(draw_img, refined_proj_pts, Scalar(0, 0, 255));
      cv_ext::showImage(pimpl_->display, "display", true, 10);
    }

    return true;
  }
  else
  {
    pimpl_->is_tracking = true;
    return false;
  }

}

bool TMObjectLocalization::refine( const cv::Mat &src_img, cv::Mat_<double> &r_vec, cv::Mat_<double> &t_vec )
{
  if( !pimpl_->initialized )
    return false;

  GradientDirectionScore scoring;

  cv::Mat resized_img, img_roi;

//  cv_ext::BasicTimer timer;

  if( src_img.empty() )
    return false;

  if( pimpl_->scale_factor != 1 )
    cv::resize(src_img, resized_img, pimpl_->cam_model.imgSize());
  else
    resized_img = src_img;

  // TODO Enable region of interest
  if( false /*has_roi*/ )
    img_roi = resized_img(pimpl_->cam_model.regionOfInterest());
  else
    img_roi = resized_img;

//  timer.reset();
  const int num_directions = 60;
  const double tensor_lambda = 6.0;
  const bool smooth_tensor = false;

  ImageTensorPtr dst_map_tensor_ptr = std::make_shared<ImageTensor>();
  pimpl_->dt.computeDistanceMapTensor( img_roi, *dst_map_tensor_ptr, num_directions, tensor_lambda, smooth_tensor);

//  cout << "Tensor computation ms: " << timer.elapsedTimeMs() << endl;

  ImageGradient im_grad( img_roi );

  pimpl_->dcr.setInput(dst_map_tensor_ptr);

//  // Memorize the original transformation to recover from divergences
//  cv::Mat_<double> tmp_r_vec = r_vec.clone(), tmp_t_vec = t_vec.clone();

  pimpl_->dcr.refinePosition(r_vec, t_vec);

//  std::cout<<cv::norm(t_vec,tmp_t_vec, cv::NORM_L2)<<std::endl;
//  std::cout<<cv::norm(tmp_t_vec, cv::NORM_L2)<<std::endl;
//  if(cv::norm(t_vec,tmp_t_vec, cv::NORM_L2) > 0.1*cv::norm(tmp_t_vec, cv::NORM_L2))
//  {
//    std::cout<<"Revert to initial transformation"<<std::endl;
//    // Revert to initial transformation
//    r_vec = tmp_r_vec;
//    t_vec = tmp_t_vec;
//  }

  vector<Point2f> refined_proj_pts;
  vector<float> normals;
  pimpl_->obj_model_ptr->setModelView(r_vec, t_vec);
  pimpl_->obj_model_ptr->projectRasterPoints(refined_proj_pts, normals);

  double score = scoring.evaluate(im_grad, refined_proj_pts, normals);
  std::cout<<"Score : "<<score<<std::endl;

  if( score >= pimpl_->score_th )
  {
    if( pimpl_->display_enabled )
    {
      Mat draw_img;
      resized_img.copyTo(pimpl_->display);
      // TODO Enable region of interest
      if (false /*has_roi*/)
      {
        cv::Rect cur_roi = pimpl_->cam_model.regionOfInterest();
        cv::Point dbg_tl = cur_roi.tl(), dbg_br = cur_roi.br();
        dbg_tl.x -= 1;
        dbg_tl.y -= 1;
        dbg_br.x += 1;
        dbg_br.y += 1;
        cv::rectangle(pimpl_->display, dbg_tl, dbg_br, cv::Scalar(255, 255, 255));
        draw_img = pimpl_->display(cur_roi);
      }
      else
      {
        draw_img = pimpl_->display;
      }
//      cv::putText(draw_img,"Tracking",cv::Point(50,50),cv::FONT_HERSHEY_DUPLEX,1,cv::Scalar(0,255,0),2,false);
//      cv_ext::drawPoints(draw_img, refined_proj_pts, Scalar(0, 0, 255));
      Mat render_img = pimpl_->obj_model_ptr->getRenderedModel();
      draw_img += .5*render_img;
      cv_ext::showImage(pimpl_->display, "display", true, 50);
    }
    return true;
  }
  else
  {
    return false;
  }
}
