#pragma once

#include <string>

class MeshConverter
{
 public:

  bool convert( const std::string &input_fn, const std::string &output_fn );

  void enforceASCIIOputput() { output_ = ASCII_OUT; };
  void enforceBinaryOutput() { output_ = BINARY_OUT; };
  void setScale( double  s ){ scale_ = s; };
  void enableVerboseOutput( bool enable ){ verbose_output_ = enable; };

 private:

  double  scale_ = 1.0;
  enum
  {
    UNCHANGED_OUT,
    BINARY_OUT,
    ASCII_OUT
  }

  output_ = UNCHANGED_OUT;

  bool verbose_output_ = true;
};