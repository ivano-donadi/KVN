#pragma once

#include <string>

#include <hyro/core/Component.h>
#include <hyro/utils/ISpinner.h>

#include <hyro/msgs/All.h>
#include <opencv2/highgui.hpp>

#include <hyro/msgs/ar_object_detection/ObjectDetectionItem.h>
#include <hyro/media/VideoStreamer.h>

#include <mutex> 

#include "tm_object_localization.h"

namespace hyro
{

class ArObjectLocalizationComponent : public Component
{
public:

  using Component::Component;
  
  ArObjectLocalizationComponent (const URI & uri, std::string model_file, std::string template_file, std::string cam_calib_file, bool flag=false);

  virtual
  ~ArObjectLocalizationComponent () = default;

  // State transitions

  virtual Result
  reset () override;

  virtual Result
  init (const ComponentConfiguration & configuration) override;

  virtual Result
  start () override;

  virtual Result
  check () override;

  virtual Result
  update () override;
  
  void
  callback_det (std::shared_ptr<const ObjectDetection> && value);
  void
  callback_im (std::shared_ptr<const Image> && value);
  
  void
  callback_im_stream (std::shared_ptr<const ImageStream> && value);

private:
  
  void
  callback_im_base (std::shared_ptr<const Image> & value);

  static std::shared_ptr<HyroLogger> s_logger;

  std::shared_ptr<ChannelInput<ObjectDetection> > m_input_det;
  std::shared_ptr<ChannelInput<Image> > m_input_im;
  std::shared_ptr<ChannelInput<ImageStream> > m_input_im_stream;
  
  std::unique_ptr<hyro::VideoStreamer> video_decoder = nullptr;
  bool video_decoder_configured = false;
  
  std::unique_ptr<ISpinner> m_spinner_im;
  std::unique_ptr<ISpinner> m_spinner_det;
  
  bool stream_;
  
  ObjectDetection m_dets;
  
  std::mutex det_mtx;
  
  int m_idx;
  cv::Mat curr_im_;

  TMObjectLocalization obj_loc_;
};

} // namespace hyro
