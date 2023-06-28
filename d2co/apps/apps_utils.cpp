#include "apps_utils.h"

#include "cv_ext/cv_ext.h"

#include <boost/filesystem.hpp>
#include <boost/iterator/iterator_concepts.hpp>
#include <boost/tokenizer.hpp>
#include <boost/range/iterator_range.hpp>
#include <iostream>

using namespace boost;
using namespace boost::filesystem;
using namespace std;

bool readFileNamesFromFolder ( const string& input_folder_name, vector< string >& names )
{
  names.clear();
  if ( !input_folder_name.empty() )
  {
    path p ( input_folder_name );
    for ( auto& entry : make_iterator_range ( directory_iterator ( p ), {} ) )
      names.push_back ( entry.path().string() );
    std::sort ( names.begin(), names.end() );
    return true;
  }
  else
  {
    return false;
  }
}

void objectPoseControlsHelp()
{
  std::cout << "Object pose controls"<< std::endl;
  std::cout << "[j-l-k-i-u-o] translate the model along the X-Y-Z axis " << std::endl;
  std::cout << "[a-s-q-e-z-w] handle the rotation through axis-angle notation " << std::endl;  
}

void parseObjectPoseControls ( int key ,cv::Mat &r_vec, cv::Mat &t_vec,
                               double r_inc, double t_inc )
{
  switch( key )
  {
    case 'a':
    case 'A':
      r_vec.at<double>(1,0) += r_inc;
      break;
    case 's':
    case 'S':
      r_vec.at<double>(1,0) -= r_inc;
      break;
    case 'w':
    case 'W':
      r_vec.at<double>(0,0) += r_inc;
      break;
    case 'z':
    case 'Z':
      r_vec.at<double>(0,0) -= r_inc;
      break;
    case 'q':
    case 'Q':
      r_vec.at<double>(2,0) += r_inc;
      break;
    case 'e':
    case 'E':
      r_vec.at<double>(2,0) -= r_inc;
      break;
    case 'i':
    case 'I':
      t_vec.at<double>(1,0) -= t_inc;
      break;
    case 'm':
    case 'M':
      t_vec.at<double>(1,0) += t_inc;
      break;
    case 'j':
    case 'J':
      t_vec.at<double>(0,0) -= t_inc;
      break;
    case 'k':
    case 'L':
      t_vec.at<double>(0,0) += t_inc;
      break;
    case 'u':
    case 'U':
      t_vec.at<double>(2,0) -= t_inc;
      break;
    case 'o':
    case 'O':
      t_vec.at<double>(2,0) += t_inc;
      break;
    default:
      break;
  }
}
