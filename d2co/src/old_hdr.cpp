#include <cfloat>
#include <cmath>
#include <assert.h>

#include "old_hdr.h"

static const int HDR_PIX_LEVELS = 256;
static const int HDR_MIN_WEIGHT = 1e-3;


// inline float debevec_sum(float weight, float /*average_weight*/, float t, float response)
// {
//      return weight * response/t;
// }
//
// inline float debevec_div(float weight, float /*average_weight*/, float /*t*/, float)
// {
//      return weight;
// }
//
// inline float debevec_out(float quotient)
// {
//      return quotient;
// }

// ---- Inner classes definition



HDRCreator::HDRCreator ()
{
  InitBasicVars();
}

HDRCreator::~HDRCreator()
{
}

void HDRCreator::InitBasicVars()
{
  e_WeightsType = HDRCreatorWeightsType_Triangular;
  e_ResponseType = HDRCreatorResponseType_Linear;
  e_MappingMethod = HDRCreatorMappingMethod_Log10; 
  //e_MappingMethod = HDRCreator::HDRCreatorMappingMethod_Gamma_2_6;
  hist_win_min = 0.0f;
  hist_win_max = 100.0f;
  min_response = 0;
  max_response = HDR_PIX_LEVELS;
}

cv::Mat HDRCreator::ComputeHDR ( std::vector<cv::Mat> &ldr_images, std::vector<float> &arrayofexptime )
{
  cv::Mat out( ldr_images[0].rows, ldr_images[0].cols, CV_32FC1 );
  const int pix_levels = HDR_PIX_LEVELS;
  float *w = new float[pix_levels];
  float *response = new float[pix_levels];

  switch ( e_WeightsType )
  {
  case HDRCreatorWeightsType_Triangular:
    weightsTriangle ( w, pix_levels );
    break;

  case HDRCreatorWeightsType_Gaussian:
    weightsGauss ( w, pix_levels, min_response, max_response );
    break;

  case HDRCreatorWeightsType_Plateau:
    exposureWeightsIcip06 ( w, pix_levels, min_response, max_response );
    break;
  }


  switch ( e_ResponseType )
  {
  case HDRCreatorResponseType_Linear:
    responseLinear ( response, pix_levels );
    break;
  case HDRCreatorResponseType_Gamma:
    responseGamma ( response, pix_levels );
    break;
  case HDRCreatorResponseType_Log10:
    responseLog10 ( response, pix_levels );
    break;
  }


  applyResponse ( ldr_images, arrayofexptime, out, response, w, pix_levels );



  delete[] w;
  delete[] response;
  return out;
}

cv::Mat HDRCreator::MapToLDR ( cv::Mat &hdrimage, double targetexposure )
{

  cv::Mat out( hdrimage.rows, hdrimage.cols, CV_8UC1 );

  fitToDynamicRange ( hdrimage );

  MappingFunc getMapping;

  switch ( e_MappingMethod )
  {
  case HDRCreatorMappingMethod_Linear:
    getMapping = &HDRCreator::getMappingLinear;
    break;
  case HDRCreatorMappingMethod_Gamma_1_4:
    getMapping = &HDRCreator::getMappingGamma_1_4;
    break;
  case HDRCreatorMappingMethod_Gamma_1_8:
    getMapping = &HDRCreator::getMappingGamma_1_8;
    break;
  case HDRCreatorMappingMethod_Gamma_2_2:
    getMapping = &HDRCreator::getMappingGamma_2_2;
    break;
  case HDRCreatorMappingMethod_Gamma_2_6:
    getMapping = &HDRCreator::getMappingGamma_2_6;
    break;
  case HDRCreatorMappingMethod_Log10:
    getMapping = &HDRCreator::getMappingLog10;
    break;
  default:
    getMapping = &HDRCreator::getMappingLinear;
    break;
  }


  uchar naninfcol = 0, negcol = 0;

  int rows = hdrimage.rows;
  int cols = hdrimage.cols;

  for ( int r = 0; r < rows; r++ )
  {
    float* hdr_row = hdrimage.ptr<float> ( r );
    uchar* out_row = out.ptr<uchar> ( r );
    for ( int c = 0; c < cols; c++ )
    {
      float v = hdr_row[c];

      if ( !finite ( v ) )
        out_row[c] = naninfcol;
      else if ( v < 0 )
        out_row[c] = negcol;
      else
        out_row[c] = ( this->*getMapping ) ( v );
    }
  }

  return out;
}
// ---- Calculation Methods


void HDRCreator::applyResponse ( std::vector<cv::Mat> &ldrimages, const std::vector<float> &arrayofexptime, cv::Mat &hdr_out,
                                 const float* response, const float* w, const int pix_levels )
{
  const int n_images = ldrimages.size();
  int rows = ldrimages[0].rows;
  int cols = ldrimages[0].cols;


  
  int min_pix_levels = 0;
  for ( int l = 0 ; l < pix_levels ; l++ )
  {
    if ( w[l] > 0 )
    {
      min_pix_levels = l;
      break;
    }
  }

  int max_pix_levels = pix_levels-1;
  for ( int l = pix_levels - 1 ; l >= 0 ; l-- )
  {
    if ( w[l]>0 )
    {
      max_pix_levels = l;
      break;
    }
  }



  // --- anti ghosting: for each image i, find images with
  // the immediately higher and lower exposure times
  int* i_lower = new int[n_images];
  int* i_upper = new int[n_images];
  for ( int i = 0; i < n_images ; i++ )
  {
    i_lower[i] = -1;
    i_upper[i] = -1;

    float ti = arrayofexptime[i];
    float ti_upper  = 1.0e8;
    float ti_lower  = 0;

    for ( int j=0 ; j<n_images ; j++ )
    {
      if ( i != j )
      {
        if ( arrayofexptime[j]>ti && arrayofexptime[j]<ti_upper )
        {
          ti_upper = arrayofexptime[j];
          i_upper[i] = j;
        }
        if ( arrayofexptime[j]<ti && arrayofexptime[j]>ti_lower )
        {
          ti_lower = arrayofexptime[j];
          i_lower[i] = j;
        }
      }
    }

    if ( i_lower[i] == -1 )
    {
      i_lower[i] = i;
    }
    if ( i_upper[i] == -1 )
    {
      i_upper[i]=i;
    }
  }


  for ( int r = 0; r < rows; r++ )
  {
    float *hdr_row = hdr_out.ptr<float> ( r );
    uchar **ldr_rows = new uchar*[n_images];
    for ( int i = 0; i < n_images; i++ )
      ldr_rows[i] = ldrimages[i].ptr<uchar> ( r );
    float out;

    for ( int c = 0; c < cols; c++ )
    {
      // all exposures for each pixel
      float sum = 0.0f, div = 0.0f, maxti = -1e6f, minti = 1e6f;

      //for all exposures
      for ( int i=0 ; i < n_images; ++i )
      {
        float ti = arrayofexptime[i];
        // --- anti saturation: observe minimum exposure time at which
        // saturated value is present, and maximum exp time at which
        // black value is present
        if ( ldr_rows[i][c]>max_pix_levels ) minti = minti<ti?minti:ti;
        if ( ldr_rows[i][c]<min_pix_levels ) maxti = minti>ti?minti:ti;


        // --- anti ghosting: monotonous increase in time should result
        // in monotonous increase in intensity; make forward and
        // backward check, ignore value if condition not satisfied
        int lower = ldr_rows[i_lower[i]][c];
        int upper = ldr_rows[i_upper[i]][c];
        if ( lower > ldr_rows[i][c] || upper < ldr_rows[i][c] )
          continue;

        sum += w[ldr_rows[i][c]] * response[ldr_rows[i][c]] / ti;
        div += w[ldr_rows[i][c]];
//                                      sum += (*sumf)(w[ldr_rows[i][c]], w[ldr_rows[i][c]], ti, response[ldr_rows[i][c]]);
//                                      div += (*divf)(w[ldr_rows[i][c]], w[ldr_rows[i][c]], ti, response[ldr_rows[i][c]]);
      }


      // --- anti saturation: if a meaningful representation of pixel
      // was not found, replace it with information from observed data
      if ( div == 0.0f && maxti > -1e6f )
      {
        sum = response[min_pix_levels]/maxti;
        div = 1.0f;
//                                      sum = (*sumf)(1.0f, 1.0f, maxti, response[min_pix_levels]);
//                                      div = (*divf)(1.0f, 1.0f, maxti, response[min_pix_levels]);
      }
      else if ( div == 0.0f && minti < +1e6f )
      {
        sum = response[max_pix_levels]/minti;
        div = 1.0f;
//                                      sum = (*sumf)(1.0f, 1.0f, minti, response[max_pix_levels]);
//                                      div = (*divf)(1.0f, 1.0f, minti, response[max_pix_levels]);
      }

      if ( div != 0.0f )
      {
        //out = (*outf)(sum/div);
        out = sum/div;
      }
      else
      {
        out = 0.0f;
      }

      hdr_row[c] = out;
    }
    delete[] ldr_rows;
  }

  delete[] i_lower;
  delete[] i_upper;

}

void HDRCreator::weightsTriangle ( float* w, int M )
{
  for ( int i=0; i<int ( float ( M ) /2.0f ); i++ )
  {
    w[i]=i/ ( float ( M ) /2.0f );
//        if (w[i]<0.06f)w[i]=0;
  }
  for ( int i=int ( float ( M ) /2.0f ); i<M; i++ )
  {
    w[i]= ( M-1-i ) / ( float ( M ) /2.0f );
//        if (w[i]<0.06f)w[i]=0;
  }
//   for( int m=0 ; m<M ; m++ )
//     if( m<Mmin || m>Mmax )
//       w[m] = 0.0f;
//     else
//     {
//      if ( m<int(Mmin+ (Mmax-Mmin)/2.0f +1) )
//              w[m]=(m-Mmin)/float(Mmin+(Mmax-Mmin)/2.0f);
//      else
//              w[m]=( -m+Mmin+((Mmax-Mmin)) )/float(Mmin+(Mmax-Mmin)/2.0f);
//     }

//        if (w[i]<0.06f)w[i]=0;
}

void HDRCreator::weightsGauss ( float* w, int M, int Mmin, int Mmax, float sigma )
{
  float mid = Mmin + ( Mmax-Mmin ) /2.0f - 0.5f;
  float mid2 = ( mid-Mmin ) * ( mid-Mmin );
  for ( int m=0 ; m<M ; m++ )
  {
    if ( m<Mmin || m>Mmax )
    {
      w[m] = 0.0f;
    }
    else
    {
      // gkrawczyk: that's not really a gaussian, but equation is
      // taken from Robertson02 paper.
      float weight = exp( -sigma * ( m - mid ) * ( m - mid ) / mid2 );

      if ( weight < HDR_MIN_WEIGHT )          // ignore very low weights
        w[m] = 0.0f;
      else
        w[m] = weight;
    }
  }
}

void HDRCreator::exposureWeightsIcip06 ( float* w, int M, int Mmin, int Mmax )
{
  for ( int m=0 ; m<M ; m++ )
    if ( m<Mmin || m>Mmax )
      w[m] = 0.0f;
    else
      w[m]=1.0f-pow ( ( ( 2.0f*float ( m-Mmin ) /float ( Mmax-Mmin ) ) - 1.0f ), 12.0f );
}


void HDRCreator::responseLinear ( float* response, int M )
{
  for ( int m=0 ; m<M ; m++ )
    response[m] = m / float ( M-1 ); // range is not important, values are normalized later
}

void HDRCreator::responseGamma ( float* response, int M )
{
  float norm = M / 4.0f;

  // response curve decided empirically
  for ( int m=0 ; m<M ; m++ )
    response[m] = powf ( m/norm, 1.7f ) + 1e-4;
}


void HDRCreator::responseLog10 ( float* response, int M )
{
  float mid = 0.5f * M;
  float norm = 0.0625f * M;

  for ( int m=0 ; m<M ; m++ )
    response[m] = powf ( 10.0f, float ( m - mid ) / norm );
}

inline int HDRCreator :: getMappingLinear ( float x )
{
  float final_value = ( x - this->min_response ) / ( this->max_response - this->min_response );
  return ( int ) ( clamp<float> ( final_value*255.f, 0.0f, 255.f ) + 0.5f );
}

inline int HDRCreator :: getMappingGamma_1_4 ( float x )
{
  float final_value = powf ( ( x - this->min_response ) / ( this->max_response - this->min_response ), 1.f/1.4f );
  return ( int ) ( clamp<float> ( final_value*255.f, 0.0f, 255.f ) + 0.5f );
}

inline int HDRCreator :: getMappingGamma_1_8 ( float x )
{
  float final_value = powf ( ( x - this->min_response ) / ( this->max_response - this->min_response ), 1.f/1.8f );
  return ( int ) ( clamp<float> ( final_value*255.f, 0.0f, 255.f ) + 0.5f );
}

inline int HDRCreator :: getMappingGamma_2_2 ( float x )
{
  float final_value = powf ( ( x - this->min_response ) / ( this->max_response - this->min_response ), 1.f/2.2f );
  return ( int ) ( clamp<float> ( final_value*255.f, 0.0f, 255.f ) + 0.5f );
}

inline int HDRCreator :: getMappingGamma_2_6 ( float x )
{
  float final_value = powf ( ( x - this->min_response ) / ( this->max_response - this->min_response ), 1.f/2.6f );
  return ( int ) ( clamp<float> ( final_value*255.f, 0.0f, 255.f ) + 0.5f );
}

inline int HDRCreator :: getMappingLog10 ( float x )
{
  float final_value = ( log10 ( x/hist_win_min ) ) / ( log10 ( hist_win_max/hist_win_min ) );
  return ( int ) ( clamp<float> ( final_value*255.f, 0.0f, 255.f ) + 0.5f );
}

void HDRCreator::fitToDynamicRange ( cv::Mat &hdrimage )
{
  float min = FLT_MAX;
  float max = -FLT_MAX;

  int rows = hdrimage.rows;
  int cols = hdrimage.cols;

  for ( int r = 0; r < rows; r++ )
  {
    float* hdr_row = hdrimage.ptr<float> ( r );
    for ( int c = 0; c < cols; c++ )
    {

      float v = hdr_row[c];
      if ( v > max )
        max = v;
      else if ( v < min )
        min = v;
    }
  }

  if ( min <= 0.000001 ) min = 0.000001; // If data contains negative values

  hist_win_min = log10 ( min );
  hist_win_max = log10 ( max );

  if ( hist_win_max - hist_win_min < 0.5 )
  {
    // Window too small
    float m = ( hist_win_min + hist_win_max ) /2.f;
    hist_win_max = m + 0.25;
    hist_win_min = m - 0.25;
  }

  hist_win_min = pow ( 10, hist_win_min );
  hist_win_max = pow ( 10, hist_win_max );
}

template<class T> T HDRCreator :: clamp ( T val, T min, T max )
{
  if ( val < min ) return min;
  if ( val > max ) return max;
  return val;
}
