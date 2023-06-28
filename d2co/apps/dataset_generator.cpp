#include <cstdio>
#include <string>
#include <sstream>
#include <map>
#include <boost/program_options.hpp>
#include <opencv2/rgbd.hpp>
#include "omp.h"

#include "cv_ext/cv_ext.h"

#include "raster_object_model3D.h"
#include "raster_object_model2D.h"
#include "chamfer_matching.h"

#include "apps_utils.h"

namespace po = boost::program_options;
using namespace std;
using namespace cv;
using namespace cv_ext;

struct Pose
{
  Pose(Mat r, Mat t, Rect b)
  {
    r_vec = r.clone();
    t_vec = t.clone();
    bb = b;
  };
  Pose(){};
  Mat r_vec, t_vec;
  Rect bb;
};


void normalizePose( const RasterObjectModel3D &obj_model, const PinholeCameraModel &cam_model, const Point &disp, Pose &p, int id )
{
  Point2f off_p(disp.x - p.bb.x, disp.y - p.bb.y);

  if( cv_ext::norm2D(off_p) == 0 )
    return;

  vector <Point3f> obj_pts = obj_model.getPrecomputedPoints(id);
  vector <Point2f> proj_pts;
  
  obj_model.projectRasterPoints( id, proj_pts );
  for( auto &p : proj_pts )
    p += off_p;
  
  cv::Mat r_vec(3,1,cv::DataType<double>::type);
  cv::Mat t_vec(3,1,cv::DataType<double>::type);
  
  cv::solvePnP( obj_pts, proj_pts, cam_model.cameraMatrix(), cam_model.distorsionCoeff(), r_vec, t_vec );
  
  p.r_vec = r_vec;
  p.t_vec = t_vec;
}


void loadACCVDataset( const string input_image_dir, vector<string> &image_names,
                       vector<Mat> &gt_r_vecs, vector<Mat> &gt_t_vecs )
{
  Mat gt_r_vec = Mat_<double>(3,1), gt_t_vec = Mat_<double>(3,1);
  int image_idx = 0;
  while( true )
  {
//     if(!(image_idx % 100))
//       cout<<"Loading data #"<<image_idx<<endl;

    std::string rot_name = cv::format("%s/rot%d.rot", input_image_dir.c_str(), image_idx),
                tra_name = cv::format("%s/tra%d.tra", input_image_dir.c_str(), image_idx);
    FILE *rot_file = fopen(rot_name.c_str(),"r"),
         *tra_file = fopen(tra_name.c_str(),"r");
    if( rot_file == NULL || tra_file == NULL )
      break;

    Eigen::Matrix3d gt_rot;
    Eigen::Vector3d gt_tra;
    double val0,val1;
    CV_Assert(fscanf(rot_file,"%lf %lf", &val0,&val1) == 2);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(0,0),&gt_rot(0,1),&gt_rot(0,2)) == 3);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(1,0),&gt_rot(1,1),&gt_rot(1,2)) == 3);
    CV_Assert(fscanf(rot_file,"%lf %lf %lf", &gt_rot(2,0),&gt_rot(2,1),&gt_rot(2,2)) == 3);

    CV_Assert(fscanf(tra_file,"%lf %lf", &val0, &val1) == 2);
    CV_Assert(fscanf(tra_file,"%lf %lf %lf", &gt_tra(0),&gt_tra(1),&gt_tra(2)) == 3);

    cv_ext::rotMat2Exp( gt_rot, gt_r_vec );

    gt_t_vec.at<double>(0) = gt_tra(0)/100.0;
    gt_t_vec.at<double>(1) = gt_tra(1)/100.0;
    gt_t_vec.at<double>(2) = gt_tra(2)/100.0;

    gt_r_vecs.push_back(gt_r_vec.clone());
    gt_t_vecs.push_back(gt_t_vec.clone());

    fclose(rot_file);
    fclose(tra_file);

    std::string image_name = cv::format("%s/color%d.jpg", input_image_dir.c_str(), image_idx);
    image_names.push_back(image_name);
    image_idx++;
  }
}

struct AppOptions
{
  string templates_filename, camera_filename, input_image_dir, object_name;
};

void parseCommandLine( int argc, char **argv, AppOptions &options )
{
  string app_name( argv[0] );

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
  ( "help,h", "Print this help messages" )
  ( "model_filename,m", po::value<string > ( &options.templates_filename )->required(),
    "DXF, STL or PLY model file" )
  ( "camera_filename,c", po::value<string > ( &options.camera_filename )->required(),
    "A YAML file that stores all the camera parameters (see the PinholeCameraModel object)" )
  ( "input_image_dir,d", po::value<string > ( &options.input_image_dir)->required(),
    "Data directory" )
  ( "object_name,n", po::value<string > ( &options.object_name)->required(),
    "Object name" );

  po::variables_map vm;

  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
                << endl << endl<<desc;
      exit(EXIT_SUCCESS);
    }

    po::notify ( vm );
  }

  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    exit(EXIT_FAILURE);
  }

  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    exit(EXIT_FAILURE);
  }

  cout << "Loading model from file : "<<options.templates_filename<< endl;
  cout << "Loading camera parameters from file : "<<options.camera_filename<< endl;
  cout << "Loading input images from directory : "<<options.input_image_dir<< endl;
}


int main(int argc, char **argv)
{
  AppOptions options;
  parseCommandLine( argc, argv, options );
  
  cv_ext::PinholeCameraModel cam_model;
  cam_model.readFromFile(options.camera_filename);

  int width = cam_model.imgWidth(), height = cam_model.imgHeight();

  RasterObjectModel3DPtr obj_model_ptr( new RasterObjectModel3D() );
  RasterObjectModel3D &obj_model = *obj_model_ptr;

  obj_model.setCamModel( cam_model );
  obj_model.setStepMeters(0.005);
  obj_model.setUnitOfMeasure(RasterObjectModel::MILLIMETER);
  obj_model.requestVertexColors();

  if(!obj_model.setModelFile( options.templates_filename ))
    return -1;

  obj_model.computeRaster();

  const vector<Point3f> pts = obj_model.getPoints(false);
  float min_x = std::numeric_limits<float>::max(), max_x = -std::numeric_limits<float>::max(),
        min_y = std::numeric_limits<float>::max(), max_y = -std::numeric_limits<float>::max(),
        min_z = std::numeric_limits<float>::max(), max_z = -std::numeric_limits<float>::max();
        
  for( auto &pt : pts )
  {
    if( pt.x > max_x ) max_x = pt.x;
    if( pt.x < min_x ) min_x = pt.x;

    if( pt.y > max_y ) max_y = pt.y;
    if( pt.y < min_y ) min_y = pt.y;

    if( pt.z > max_z ) max_z = pt.z;
    if( pt.z < min_z ) min_z = pt.z;    
  }
  
  float extent_x = max_x - min_x, 
        extent_y = max_y - min_y, 
        extent_z = max_z - min_z;
  
  
  
  vector<string> image_names;
  vector<Mat> gt_r_vecs, gt_t_vecs;
  vector<Pose> gt_poses;
  loadACCVDataset( options.input_image_dir, image_names, gt_r_vecs, gt_t_vecs );

  for( int i = 0; i < int(image_names.size()); i++ )
  {
    obj_model.setModelView(gt_r_vecs[i], gt_t_vecs[i]);
    obj_model.storeModelView();
    vector<Point2f> raster_pts;
    vector<float> raster_normals_dirs;

    obj_model.projectRasterPoints( raster_pts, raster_normals_dirs );
    Point tl(width,height), br(0,0);

    for( int j = 0; j < int(raster_pts.size()); j++ )
    {
      int x = cvRound(raster_pts[j].x), y = cvRound(raster_pts[j].y);

      if( unsigned(x) < unsigned(width) && unsigned(y) < unsigned(height) )
      {
        if( x < tl.x ) tl.x = x;
        if( y < tl.y ) tl.y = y;
        if( x > br.x ) br.x = x;
        if( y > br.y ) br.y = y;
      }
    }
    br.x++;
    br.y++;
    cv::Rect bb(tl,br);
    gt_poses.push_back(Pose(gt_r_vecs[i],gt_t_vecs[i], bb));

    Mat r_mat;
    cv_ext::exp2RotMat(gt_r_vecs[i],     );
    
    stringstream sstr_info;
    sstr_info<<"image size"<<cam_model.imgWidth()<<" "<<cam_model.imgHeight()<<endl;
    sstr_info<<options.object_name<<endl;
    sstr_info<<"rotation:"<<endl<<r_mat<<endl;
    sstr_info<<"center:"<<endl<<gt_t_vecs[i]<<endl;
    sstr_info<<"extent:"<<endl<<extent_x<<" "<<extent_y<<" "<<extent_z<<endl;
  
    cout<<sstr_info.str();
  
   // Test draw ground truths    
    
    Mat src_img = imread ( image_names[i].c_str(), cv::IMREAD_COLOR );
    cv_ext::drawPoints(src_img, raster_pts, Scalar(0,0,255));
    cv::rectangle(src_img, bb,CV_RGB(0, 0, 255));

    Mat src_template = obj_model.getRenderedModel();
    
    cv_ext::showImage(src_img, "Input image", true, 10);
    cv_ext::showImage(src_template, "Template");

  }
  
  return 0;
}
