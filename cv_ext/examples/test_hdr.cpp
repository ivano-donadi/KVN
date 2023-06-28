#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

#include "cv_ext/cv_ext.h"

void loadExposureSeq ( std::string path, std::vector<cv::Mat>& images, 
                       std::vector<cv::Mat>& gl_images, std::vector<float>& times )
{
  images.clear();
  gl_images.clear();
  
  path = path + std::string ( "/" );
  std::ifstream list_file ( ( path + "list.txt" ).c_str());
  std::string name;
  float val;
  while ( list_file >> name >> val )
  {
    cv::Mat img = cv::imread ( path + name, cv::IMREAD_COLOR ),
            gl_img = cv::imread ( path + name, cv::IMREAD_GRAYSCALE );
    images.push_back ( img );
    gl_images.push_back ( gl_img );
    times.push_back ( 1.0/val );
  }
  list_file.close();
}


int main ( int argc, char**argv )
{
  std::vector<cv::Mat> images, gl_images;
  std::vector<float> times;
  
  if( argc > 2)
    return -1;
    
  loadExposureSeq ( argv[1], images, gl_images, times );
    
  cv_ext::BasicTimer timer;

  cv::Mat hdr, ldr;
  
  for( int i = 0; i < 2; i++)
  {
  
    cv_ext::DebevecHDR<uchar>debevec_hdr;
    debevec_hdr.enableParallelism ( true );

    debevec_hdr.setWeights(cv_ext::TRIANGLE_WEIGHT);
    
    timer.reset();

    debevec_hdr.merge ( images, times, hdr );
    
    std::cout<<"CVExt mergeDebevec() elapsed time : "<<timer.elapsedTimeMs() <<std::endl;
    
    cv_ext::GammaTonemap gamma_tonemap;
    gamma_tonemap.setGamma(5);
    
    timer.reset();
    
    gamma_tonemap.compute(hdr, ldr);
              
    std::cout<<"CVExt tonemapGamma() elapsed time : "<<timer.elapsedTimeMs() <<std::endl;

    cv_ext::showImage ( ldr,"CVExt Debevec merge with triangle weight and gamma tonemap", true, 10 );

    debevec_hdr.setWeights( cv_ext::GAUSSIAN_WEIGHT );
    
    timer.reset();

    debevec_hdr.merge ( images, times, hdr );
    
    std::cout<<"CVExt mergeDebevec() elapsed time : "<<timer.elapsedTimeMs() <<std::endl;
  
    timer.reset();
    
    gamma_tonemap.setGamma(2.2);
    gamma_tonemap.compute(hdr, ldr);

    std::cout<<"CVExt tonemapGamma() elapsed time : "<<timer.elapsedTimeMs() <<std::endl;

    cv_ext::showImage ( ldr,"CVExt Debevec merge with gaussian weight and gamma tonemap", true, 10 );
    
    cv_ext::MertensHDR<uchar> mertens_hdr;
    mertens_hdr.enableParallelism(true);
    mertens_hdr.setContrastExponent(1);
    mertens_hdr.setSaturationExponent(1);
    mertens_hdr.setExposednessExponent(1);
    
    timer.reset();
    
    mertens_hdr.merge(images, times, hdr);
              
    std::cout<<"CVExt Mertens elapsed time : "<<timer.elapsedTimeMs() <<std::endl;
    
    cv_ext::showImage ( hdr,"CVExt Mertens merge ", true, 10 );


#ifdef USE_OPENCV3
    timer.reset();
    
    cv::Ptr<cv::MergeMertens> merge_mertens = cv::createMergeMertens(1,1,1);
    merge_mertens->process(images, hdr);

    std::cout<<"OpenCV Merge Mertens elapsed time : "<<timer.elapsedTimeMs() <<std::endl;
    
    cv_ext::showImage ( hdr,"OpenCV Merge Mertens" );
#endif
   
    while( cv_ext::waitKeyboard() != 27 ) ;
    images = gl_images;
  }
  
  return 0;
}
