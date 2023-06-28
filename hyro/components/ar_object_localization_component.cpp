#include <iostream>
#include <cassert>

#include <hyro/utils/SpinnerDefault.h>
#include <hyro/utils/SpinnerRated.h>
#include <hyro/ar_object_localization_component.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


namespace hyro
{
  
int getMaxConfIdx(ObjectDetection &msg)
{
  float c=0;
  int idx=-1;
  for (int i=0; i<msg.detections.size(); i++)
  {
    if(msg.detections[i].confidence>c)
    {
      c=msg.detections[i].confidence;
      idx=i;
    }
  }
  return idx;
}

void drawDetections(cv::Mat &im, ObjectDetection &dets)
{
  if (dets.detections.size()==0)
    return;
//   int idx=getMaxConfIdx(dets);
//   cv::Rect r(dets[idx].bb_tl_x, dets[idx].bb_tl_y, dets[idx].bb_w, dets[idx].bb_h);
//   cv::rectangle(im, r, cv::Scalar(0,0,255),2);
//   std::stringstream ss;
//   ss<<dets[idx].class_name<<" "<<dets[idx].confidence*100<<"\%";
//   cv::putText(im,ss.str(), cv::Point(dets[idx].bb_tl_x+3, dets[idx].bb_tl_y+25),cv::FONT_HERSHEY_PLAIN, 1.5,cv::Scalar(0,0,255));
  
  for(int i=0; i<dets.detections.size(); i++)
  {
    cv::Rect r(dets.detections[i].bb_tl_x, dets.detections[i].bb_tl_y, dets.detections[i].bb_w, dets.detections[i].bb_h);
    cv::rectangle(im, r, cv::Scalar(0,0,255),2);
    std::stringstream ss;
    ss<<dets.detections[i].class_name<<" "<<dets.detections[i].confidence*100<<"\%";
    cv::putText(im,ss.str(), cv::Point(dets.detections[i].bb_tl_x+3, dets.detections[i].bb_tl_y+25),cv::FONT_HERSHEY_PLAIN, 1.5,cv::Scalar(0,0,255));
  }
}

std::shared_ptr<HyroLogger> ArObjectLocalizationComponent::s_logger = HyroLoggerManager::CreateLogger("ArObjectLocalizationComponent");
ArObjectLocalizationComponent::ArObjectLocalizationComponent (const URI & uri, std::string model_file, std::string template_file, std::string cam_calib_file, bool flag) 
    : Component(uri)
{
  stream_=flag; 
  obj_loc_.initialize(cam_calib_file, model_file, template_file);
//   obj_loc_.setScaleFactor(3);
}

Result
ArObjectLocalizationComponent::reset ()
{
  if (stream_)
  {
    m_input_im_stream.reset();
  }
  else
    m_input_im.reset();
  m_input_det.reset();
  m_idx=0;

  return Result::RESULT_OK;
}

Result
ArObjectLocalizationComponent::init (const ComponentConfiguration & configuration)
{
  if (stream_)
  {
    m_input_im_stream = this->registerInput<hyro::ImageStream>("input_image"_uri, configuration);
  }
  else
    m_input_im = this->registerInput<Image>("input_image"_uri, configuration);
  m_input_det = this->registerInput<ObjectDetection>("input_detections"_uri, configuration);
  m_idx=0;
 return Result::RESULT_OK;
}

Result
ArObjectLocalizationComponent::start ()
{
  if(stream_)
  {
    m_spinner_im = SpinnerDefault::Create(m_input_im_stream, &ArObjectLocalizationComponent::callback_im_stream, this, this->suspensionToken());
  }
  else
    m_spinner_im = SpinnerDefault::Create(m_input_im, &ArObjectLocalizationComponent::callback_im, this, this->suspensionToken());
  m_spinner_det = SpinnerDefault::Create(m_input_det, &ArObjectLocalizationComponent::callback_det, this, this->suspensionToken());
  return Result::RESULT_OK;
}

Result
ArObjectLocalizationComponent::check ()
{
  return Result::RESULT_OK;
}

Result
ArObjectLocalizationComponent::update ()
{
  det_mtx.lock();
  if(curr_im_.cols>0&&curr_im_.rows>0)
  {
    cv::Mat dbg;
    cv::resize(curr_im_,dbg,cv::Size(800,600));
    obj_loc_.localize(dbg);
    
    cv::Mat disp=curr_im_.clone();
    drawDetections(disp, m_dets);
    det_mtx.unlock();
    
  //   s_logger->info(this, "Received: {}", m_idx);
//     cv::resize(disp,disp,cv::Size(1024,576));
//     cv::resize(disp,disp,cv::Size(),.75,.75);
//     cv::imshow("Object Localization viewer", disp);
//     cv::waitKey(10);
  }
  else
    det_mtx.unlock();
  return Result::RESULT_OK;
}

void
ArObjectLocalizationComponent::callback_im_base (std::shared_ptr<const Image> & value)
{
  m_idx ++;
  det_mtx.lock();
  curr_im_=value->image.clone();
  det_mtx.unlock();
}

void
ArObjectLocalizationComponent::callback_im (std::shared_ptr<const Image> && value)
{
  callback_im_base(value);
}

void
ArObjectLocalizationComponent::callback_im_stream (std::shared_ptr<const ImageStream> && encoded_frame)
{
  if(video_decoder == nullptr)
  {
    if (encoded_frame->codec == "ffvhuff")
      video_decoder = std::make_unique<hyro::LosslessVideoStreamer>();
    else if (encoded_frame->codec == "libx264")
      video_decoder = std::make_unique<hyro::H264VideoStreamer>();
    else if (encoded_frame->codec == "mpeg4")
      video_decoder = std::make_unique<hyro::MJpegVideoStreamer>();
    else
    {
      auto msg = fmt::format("Codec {} is not supported from ImageViewer", encoded_frame->codec);
      m_input_im_stream->disconnect();
      return;
    }
  }

  // Configure (just once) the decoder
  if (!video_decoder_configured)
  {
    video_decoder->allocateResourcesForReceive(encoded_frame->width, encoded_frame->height, 25);
    video_decoder->startReceiveFrame(encoded_frame->extra_data);

    video_decoder_configured = true;
  }

  // Decode the received frame
  std::shared_ptr<const hyro::Image> frame;
  frame = video_decoder->fromMessage(encoded_frame);
  callback_im_base(frame);  
}

void
ArObjectLocalizationComponent::callback_det (std::shared_ptr<const ObjectDetection> && value)
{
  det_mtx.lock();
  m_dets=*value;
//   std::cout<<m_dets.header<<std::endl<<m_dets.img_w<<" "<<m_dets.img_h<<std::endl;;
//   m_dets.clear();
//   m_dets.push_back(*value);
  det_mtx.unlock();
}

} // namespace hyro
