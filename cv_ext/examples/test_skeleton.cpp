#include <opencv2/opencv.hpp>

#include "cv_ext/cv_ext.h"

void plotGraph( cv::Mat &img, std::vector<cv::Point2f> &nodes, 
                              std::vector< std::vector<int> > &edges,
                              bool draw_edges = true, float scale = 1.0f )
{
  for( int i = 0; i < int(nodes.size()); i++ )
  {
    cv::Point p1(cvRound(scale*nodes[i].x), cvRound(scale*nodes[i].y));
    cv::circle(img, p1, 2, cv::Scalar(0,0,255), 2);
    if(draw_edges)
    {
      std::vector<int> &neighbors = edges[i];
      for( int j = 0; j < int(neighbors.size()); j++ )
      {     
        cv::Point p2(cvRound(scale*nodes[neighbors[j]].x), cvRound(scale*nodes[neighbors[j]].y));
        cv::line(img, p1, p2, cv::Scalar(255,0,0) );
      }
    }
  }
}

int main(int argc, char** argv)
{
  cv::Mat input_img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE), 
                      dilated_input_img, skeleton, graph, graph2X;
  
  cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT,cv::Size( 7,7),cv::Point(3,3));  
  cv::dilate(input_img ,dilated_input_img, element );
  
  cv_ext::BasicTimer timer;
  
  timer.reset();
  cv_ext::morphThinning( dilated_input_img, skeleton );
  std::cout<<"Time morphThinning : "<<(unsigned long)timer.elapsedTimeUs()<<std::endl;
  
  std::vector<cv::Point2f> nodes;
  std::vector< std::vector<int> > edges;
                                    
  timer.reset();
  cv_ext::graphExtraction( skeleton, nodes, edges, true, 3 );
  std::cout<<"Time graphExtraction : "<<(unsigned long)timer.elapsedTimeUs()<<std::endl;
 
  graph = cv::Mat(skeleton.size(), cv::DataType<cv::Vec3b>::type);
  cv::cvtColor(skeleton, graph, cv::COLOR_GRAY2BGR);
  plotGraph( graph, nodes, edges,true );
  cv::resize(graph, graph2X, cv::Size(2*graph.cols, 2*graph.rows));
  plotGraph( graph2X, nodes, edges,true, 2.0f );
  
  cv::imshow("src",input_img);
  cv::imshow("dilated",dilated_input_img);
  cv::imshow("skeleton",skeleton);
  cv::imshow("voronoi",graph);
  cv::imshow("voronoi2X",graph2X);
  
  while(cv_ext::waitKeyboard() != 27);
  
  return 0;
}
