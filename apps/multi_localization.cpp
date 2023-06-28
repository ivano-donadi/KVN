#include <string>
#include <sstream>
#include <algorithm>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "io_utils.h"
#include "apps_utils.h"

#include "tm_object_localization.h"
#include "pvnet_wrapper.h"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;
using namespace cv;
using namespace cv_ext;

int main(int argc, char **argv)
{
  string app_name( argv[0] );
  if( argc < 3 )
  {
    std::cout<<"TODO"<<std::endl;
    return 0;
  }

  int num_objects = argc - 2;
  std::vector<po::options_description> program_options(num_objects);

  std::vector<CADCameraOptions> model_opt(num_objects);
  std::vector<DefaultOptions> def_opt(num_objects);
  std::vector<BBOffsetOptions> bbo_opt(num_objects);
  std::vector<SynLocOptions> sl_opt(num_objects);

  for( int i = 0; i < num_objects; i++ )
  {
    program_options[i].add(model_opt[i].getDescription());
    program_options[i].add(sl_opt[i].getDescription());
    program_options[i].add(bbo_opt[i].getDescription());
    program_options[i].add(def_opt[i].getDescription());

    po::variables_map vm;

    try
    {
      std::ifstream ifs(argv[i + 1]);
      store(po::parse_config_file(ifs, program_options[i]), vm);

      po::notify ( vm );
    }
    catch ( boost::program_options::required_option& e )
    {
      cerr << "ERROR: " << e.what() << endl << endl;
//      cout << "USAGE: "<<app_name<<" OPTIONS"
//      << endl << endl<<program_options;
      exit(EXIT_FAILURE);
    }
    catch ( boost::program_options::error& e )
    {
      cerr << "ERROR: " << e.what() << endl << endl;
//      cout << "USAGE: "<<app_name<<" OPTIONS"
//      << endl << endl<<program_options;
      exit(EXIT_FAILURE);
    }

    model_opt[i].print();
    sl_opt[i].print();
    bbo_opt[i].print();
    def_opt[i].print();
  }

  std::cout<<"Loading images from folder :"<<argv[num_objects + 1]<<std::endl;

  std::vector<TMObjectLocalization> obj_loc(num_objects);
  PVNetWrapper pv_net_wrapper(sl_opt[0].pvnet_home);

  for( int i = 0; i < num_objects; i++ )
  {
    obj_loc[i].setNumMatches(5);
    obj_loc[i].setScaleFactor(model_opt[i].scale_factor);
    obj_loc[i].setCannyLowThreshold(40);
    obj_loc[i].setScoreThreshold(sl_opt[i].score_threshold);
    obj_loc[i].setModelSaplingStep(sl_opt[i].model_samplig_step);

    //  if( top_boundary != -1 || bottom_boundary != -1 ||
    //      left_boundary != -1 || rigth_boundary != -1 )
    //  {
    //    Point tl(0,0), br(scaled_img_size.width, scaled_img_size.height);
    //
    //    if( top_boundary != -1 ) { tl.y = top_boundary; }
    //    if( left_boundary != -1 ) { tl.x = left_boundary;  }
    //    if( bottom_boundary != -1 ) { br.y = bottom_boundary; }
    //    if( rigth_boundary != -1 ) { br.x = rigth_boundary; }
    //
    //    obl_loc.setRegionOfInterest( cv::Rect(tl, br) );
    //  }

    if ( !model_opt[i].unit.compare("m") )
      obj_loc[i].setUnitOfMeasure(RasterObjectModel::METER);
    else if ( !model_opt[i].unit.compare("cm") )
      obj_loc[i].setUnitOfMeasure(RasterObjectModel::CENTIMETER);
    else if ( !model_opt[i].unit.compare("mm") )
      obj_loc[i].setUnitOfMeasure(RasterObjectModel::MILLIMETER);

    // TODO
    //  cout << "Bounding box origin offset [off_x, off_y, off_z] : ["
    //       <<bb_xoff<<", "<<bb_yoff<<", "<<bb_zoff<<"]"<< endl;
    //  cout << "Bounding box size offset [off_width, off_height, off_depth] : ["
    //       <<bb_woff<<", "<<bb_hoff<<", "<<bb_doff<<"]"<< endl;
    //
    //  if( bb_xoff ||  bb_yoff || bb_zoff ||
    //      bb_woff || bb_hoff || bb_doff )
    //    obl_loc.setBoundingBoxOffset( bb_xoff, bb_yoff, bb_zoff,
    //                                  bb_woff, bb_hoff, bb_doff );


    obj_loc[i].enableDisplay(true);
    obj_loc[i].initialize(model_opt[i].camera_filename, model_opt[i].model_filename, sl_opt[i].templates_filename );

    pv_net_wrapper.registerObject(i,sl_opt[i].pvnet_model.c_str(),
                                  sl_opt[i].pvnet_inference_meta.c_str());
  }

  vector<string> filelist;
  if ( !readFileNamesFromFolder ( argv[num_objects + 1], filelist ) )
  {
    cerr<<"Wrong or empty folders"<<endl;
    exit ( EXIT_FAILURE );
  }

  cv_ext::BasicTimer timer;
  cv::Mat src_img, pvnet_img;
  int num_images = 0;
  cv::Mat_<double> r_mat, t_vec, r_vec(3,1);
  int last_found = -1;
  for ( int i = 0; i < static_cast<int> ( filelist.size() ); i++ )
  {
    timer.reset();
    src_img = cv::imread ( filelist[i] );
    if(src_img.empty())
      continue;

    num_images++;
    std::cout<<"Loading image : "<<filelist[i]<<endl;
    cv::cvtColor(src_img, pvnet_img, cv::COLOR_BGR2RGB);

    if(last_found < 0 )
    {
      for( int j = 0; j < num_objects; j++ )
      {
        pv_net_wrapper.localize(pvnet_img,j, r_mat, t_vec);
        cv_ext::rotMat2AngleAxis<double>(r_mat, r_vec);

        if( obj_loc[j].refine(src_img, r_vec, t_vec) )
        {
          last_found = j;
          break;
        }
      }
    }
    else
    {
      if( !obj_loc[last_found].refine(src_img, r_vec, t_vec) )
        last_found = -1;
    }

    if(last_found < 0 )
      cv_ext::showImage(src_img, "display", true, 10);

    //    timer.reset();
    //    obl_loc.localize(src_img );
    cout << "Object localization ms: " << timer.elapsedTimeMs() << endl;
  }


  return EXIT_SUCCESS;
}
