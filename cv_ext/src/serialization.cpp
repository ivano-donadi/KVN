#include "cv_ext/serialization.h"
#include "cv_ext/conversions.h"

#include <stdexcept>
#include <fstream>

using namespace std;
using namespace cv;
using namespace cv_ext;

string cv_ext::generateYAMLFilename ( const string& filename )
{
  int pos1 = filename.length() - 4, pos2 = filename.length() - 5;
  string ext_str1 = filename.substr ((pos1 > 0)?pos1:0);
  string ext_str2 = filename.substr ((pos2 > 0)?pos2:0);

  if( ext_str1.compare(".yml") && ext_str2.compare(".yaml") &&
      ext_str1.compare(".YML") && ext_str2.compare(".YAML") )
    return filename + ".yml";
  else
    return filename;
}

void cv_ext::read3DTransf(const YAML::Node &in_node, cv::Mat &rotation, cv::Mat &translation, RotationFormat rf)
{
  int r_rows = in_node["rotation"]["rows"].as<int>();
  int r_cols = in_node["rotation"]["cols"].as<int>();
  int t_rows = in_node["translation"]["rows"].as<int>();
  int t_cols = in_node["translation"]["cols"].as<int>();
  YAML::Node r_data = in_node["rotation"]["data"];
  YAML::Node t_data = in_node["translation"]["data"];

  Mat_<double> r(r_rows, r_cols), t(t_rows, t_cols);
  // fill r
  for (int i=0; i<r_rows; ++i)
    for (int j=0; j<r_cols; ++j)
      r.at<double>(i,j) = r_data[r_cols*i + j].as<double>();
  // fill t
  for (int i=0; i<t_rows; ++i)
    for (int j=0; j<t_cols; ++j)
      t.at<double>(i,j) = t_data[t_cols*i + j].as<double>();

  switch( rf )
  {
    case ROT_FORMAT_MATRIX:

      if( r.rows == 3 && r.cols == 3 )
        rotation = r;
      else if( r.rows == 3 && r.cols == 1 )
        angleAxis2RotMat<double>(r, rotation);
      else if( r.rows == 1 && r.cols == 3 )
      {
        cv::transpose(r,r);
        angleAxis2RotMat<double>(r, rotation);
      }
      else
        throw runtime_error("read3DTransf() : invalid file format");

      break;

    case ROT_FORMAT_AXIS_ANGLE:

      if( r.rows == 3 && r.cols == 1 )
        rotation = r;
      else if( r.rows == 1 && r.cols == 3 )
        cv::transpose(r,rotation);
      else if( r.rows == 3 && r.cols == 3 )
        rotMat2AngleAxis<double>(r, rotation );
      else
        throw runtime_error("read3DTransf() : invalid file format");

      break;
  }

  if( t.rows == 3 && t.cols == 1 )
    translation = t;
  else if( t.rows == 1 && t.cols == 3 )
    cv::transpose(t,translation);
  else
    throw runtime_error("read3DTransf() : invalid file format");
}

bool cv_ext::read3DTransf( const string &filename, Mat &rotation,
                           Mat &translation, RotationFormat rf )
{
  string yml_filename = generateYAMLFilename(filename);

  try
  {
    YAML::Node in_node = YAML::LoadFile(yml_filename);
    read3DTransf( in_node, rotation, translation, rf );
  }
  catch(std::exception &e)
  {
    std::cerr << "read3DTransf() failed to read file" << std::endl;
    return false;
  }

  return true;
}

bool cv_ext::write3DTransf( const string &filename, const Mat &rotation,
                            const Mat &translation )
{
  string yml_filename = generateYAMLFilename(filename);

  try
  {
    YAML::Node out_node_root;
    write3DTransf(out_node_root, rotation, translation);

    std::ofstream out(yml_filename);
    out << out_node_root;
    out.close();
  }
  catch(std::exception &e)
  {
    std::cerr << "read3DTransf() failed to write file" << std::endl;
    return false;
  }

  return true;
}

void cv_ext::write3DTransf( YAML::Node &out_node, const cv::Mat &rotation,
                           const cv::Mat &translation)
{
  assert( ( rotation.rows == 3 && rotation.cols == 3 ) ||
          ( rotation.rows == 3 && rotation.cols == 1 ) ||
          ( rotation.rows == 1 && rotation.cols == 3 ) );

  assert( ( translation.rows == 3 && translation.cols == 1 ) ||
          ( translation.rows == 1 && translation.cols == 3 ) );

  YAML::Node r_node, t_node;
  r_node["rows"] = rotation.rows;
  r_node["cols"] = rotation.cols;
  r_node["dt"] = "d";
  for (int i=0; i<rotation.rows; ++i)
    for (int j=0; j<rotation.cols; ++j)
      r_node["data"].push_back(rotation.at<double>(i, j));
  t_node["rows"] = translation.rows;
  t_node["cols"] = translation.cols;
  t_node["dt"] = "d";
  for (int i=0; i<translation.rows; ++i)
    for (int j=0; j<translation.cols; ++j)
      t_node["data"].push_back(translation.at<double>(i, j));

  out_node["rotation"] = r_node;
  out_node["translation"] = t_node;
}