#pragma once

#include "cv_ext/cv_ext.h"
#include <string>
#include <boost/program_options.hpp>

struct ProgramOptionsBase
{
  explicit ProgramOptionsBase( const std::string &help ) : options_ (help){};

  const boost::program_options::options_description &getDescription(){ return options_; };
  virtual void print() const = 0;
  virtual bool checkData() const {return true; };

 protected:
  boost::program_options::options_description options_;
};

struct DefaultOptions : public ProgramOptionsBase
{
  explicit DefaultOptions( const std::string &help = "Default Options" ) : ProgramOptionsBase ( help )
  {
    addOptions();
  };

  void addOptions();
  void print() const override;

  bool helpRequired( const boost::program_options::variables_map &vm ) const;
  std::string cfgFilename( const boost::program_options::variables_map &vm ) const;

 private:
  std::string cfg_filename;
};

struct CADCameraOptions : public ProgramOptionsBase
{
  explicit CADCameraOptions( const std::string &help = "CAD Model/Camera Options" ) : ProgramOptionsBase ( help )
  {
    addOptions();
  };

  void addOptions();
  void print() const override;

  std::string model_filename, unit = std::string("m"), camera_filename;
  double scale_factor = 1.0;
};

struct SpaceSamplingOptions : public ProgramOptionsBase
{
  explicit SpaceSamplingOptions(const std::string &help = "Space Sampling Options" ) : ProgramOptionsBase (help )
  {
    addOptions();
  };

  void addOptions();
  void print() const override;
  bool checkData() const override;

  std::string vert_axis = std::string("z");

  int rotation_sampling_level = 2;

  double min_dist = 2, max_dist = 16, dist_step = 0.5;
  double min_height = 0, max_height = 0, height_step = 0.5;
  double min_soff = 0, max_soff = 0, soff_step = 0.5;
};


struct RoIOptions : public ProgramOptionsBase
{
  explicit RoIOptions(const std::string &help = "3D Bounding Box Options" ) : ProgramOptionsBase (help )
  {
    addOptions();
  };

  void addOptions();
  void print() const override;

  int top_boundary = -1, bottom_boundary = -1, left_boundary = -1, rigth_boundary = -1;
};


struct BBOffsetOptions : public ProgramOptionsBase
{
  explicit BBOffsetOptions(const std::string &help = "3D Bounding Box Offset Options" ) : ProgramOptionsBase (help )
  {
    addOptions();
  };

  void addOptions();
  void print() const override;

  double bb_xoff = 0, bb_yoff = 0, bb_zoff = 0,
         bb_woff = 0, bb_hoff = 0, bb_doff = 0;
};

struct SynLocOptions : public ProgramOptionsBase
{
  explicit SynLocOptions(const std::string &help = "Synergic Localization Options" ) : ProgramOptionsBase (help )
  {
    addOptions();
  };

  void addOptions();
  void print() const override;

  std::string pvnet_home = std::string("../pvnet"),
              templates_filename, pvnet_model, pvnet_inference_meta;
  double model_samplig_step = 0.1, score_threshold = 0.6;
  int obj_id = 0, matching_step = 4, num_matches = 5;
};

void objectPoseControlsHelp();
void parseObjectPoseControls ( int key ,cv::Mat &r_vec, cv::Mat &t_vec,
                               double r_inc = 0.1, double t_inc = 0.1 );
bool checkInterval( const double min, const double max, const double step, const std::string &name = "" );