#include <iostream>
#include <sstream>

#include <boost/program_options.hpp>

#include "cv_ext/image_gradient.h"
#include "cv_ext/debug_tools.h"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char** argv)
{
  string  app_name( argv[0] ), image_name;
  unsigned int pyr_levels = 4;
  double pyr_scale_factor = 2.0;
  bool use_scharr = true, fast_magnitude = true;

  po::options_description desc ( "OPTIONS" );
  desc.add_options()
      ( "help,h", "Print this help messages" )
      ( "input_image,i", po::value<string > ( &image_name )->required(),
        "Input image" )
      ( "pyramid_levels, p", po::value<unsigned int> ( &pyr_levels ),
        "Number of pyramid levels  [default: 4]" )
      ( "pyr_scale_factor,s", po::value<double> ( &pyr_scale_factor),
        "Pyramid scale factor [default: 2.0]" )
      ( "use_scharr", po::value<bool> ( &use_scharr),
        "Use Scharr operator to compute image derivetives  [default: true]" )
      ( "fast_magnitude", po::value<bool> ( &fast_magnitude),
        "Compute gradient magnitude in an approximated, fast way (i.e., using the absolute value) [default: true]" );

  po::variables_map vm;
  try
  {
    po::store ( po::parse_command_line ( argc, argv, desc ), vm );

    if ( vm.count ( "help" ) )
    {
      cout << "USAGE: "<<app_name<<" OPTIONS"
           << endl << endl<<desc;
      return 0;
    }

    po::notify ( vm );
  }
  catch ( boost::program_options::required_option& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }
  catch ( boost::program_options::error& e )
  {
    cerr << "ERROR: " << e.what() << endl << endl;
    return -1;
  }

  cout<<"Input_image : "<<image_name<<endl;
  cout<<"Pyramid levels : "<<pyr_levels<<endl;
  cout<<"Pyramid scale factor : "<<pyr_scale_factor<<endl;
  if( use_scharr )
    cout<<"Using Scharr operator to compute image derivatives"<<endl;
  else
    cout<<"Using Sobel operator to compute image derivatives"<<endl;
  if( fast_magnitude )
    cout<<"Computing gradient magnitude in an approximated, fast way"<<endl;

  cv::Mat input_img = cv::imread(image_name, cv::IMREAD_COLOR);
  cv_ext::ImageGradient im_grad( input_img, pyr_levels, pyr_scale_factor );
  im_grad.enableScharrOperator(use_scharr);
  im_grad.enableFastMagnitude(fast_magnitude);
  
  std::vector<std::string> scale_str;
  for( int i = 0; i < im_grad.numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << im_grad.getPyrScale(i);
    scale_str.push_back(strs.str());
  }
  
  for( int i = 0; i < im_grad.numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Intensities, scale : "<<scale_str.at(i);
    cv_ext::showImage( im_grad.getIntensities(i), strs.str(), true, 1 );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();

  for( int i = 0; i < im_grad.numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Gradient X, scale : "<<scale_str.at(i);
    cv_ext::showImage( im_grad.getGradientX(i), strs.str(), true, 1 );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();

  for( int i = 0; i < im_grad.numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Gradient Y, scale : "<<scale_str.at(i);
    cv_ext::showImage( im_grad.getGradientY(i), strs.str(), true, 1 );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();

  for( int i = 0; i < im_grad.numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Gradient directions, scale : "<<scale_str.at(i);
    cv_ext::showImage( im_grad.getGradientDirections(i), strs.str(), true, 1 );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();
  
  for( int i = 0; i < im_grad.numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Gradien magnitude, scale : "<<scale_str.at(i);
    cv_ext::showImage( im_grad.getGradientMagnitudes(i), strs.str(), true, 1 );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();

  for( int i = 0; i < im_grad.numPyrLevels(); i++)
  {
    std::ostringstream strs;
    strs << "Eigen orientation, scale : "<<scale_str.at(i);
    cv_ext::showImage( im_grad.getEigenDirections(i), strs.str(), true, 1 );
  }
  while(cv_ext::waitKeyboard() != 27);
  cv::destroyAllWindows();

  return 0;
}