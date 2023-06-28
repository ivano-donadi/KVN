#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

bool readFileNames ( const std::string &input_file_name, std::vector<std::string> &names );
bool readFileNames ( const std::string &input_file_name, std::vector< std::vector<std::string> > &names );
bool readFileNamesFromFolder( const std::string& input_folder_name, std::vector< std::string >& names );