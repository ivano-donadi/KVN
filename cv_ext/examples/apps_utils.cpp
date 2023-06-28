#include "apps_utils.h"

#include <fstream>
#include <algorithm>

#include <boost/filesystem.hpp>
#include <boost/iterator/iterator_concepts.hpp>
#include <boost/tokenizer.hpp>
#include <boost/range/iterator_range.hpp>

using namespace std;
using namespace boost;
using namespace boost::filesystem;

bool readFileNames ( const string& input_file_name, vector< string >& names )
{
  names.clear();
  std::ifstream f ( input_file_name.c_str() );

  if ( !f )
  {
    cout<<"Error opening file"<<endl;
    return false;
  }

  string line;

  while ( getline ( f, line ) )
  {
    if( !line.empty())
      names.push_back ( line );
  }

  f.close();

  return true;
}

bool readFileNames ( const string& input_file_name, vector< vector< string > >& names )
{
  names.clear();

  std::ifstream f ( input_file_name.c_str() );
  string s_line;

  if ( !f.is_open() )
  {
    cout<<"Error opening file"<<endl;
    return false;
  }

  bool first_line = true;
  int num_images_row = 0;

  char_separator<char> sep ( " " );

  string line;

  while ( getline ( f, line ) )
  {
    if( !line.empty())
    {
      names.push_back ( vector<string>() );
      vector<string> &image_names = names.back();

      tokenizer< char_separator<char> > tokens ( line, sep );
      for ( const auto& t : tokens )
      {
        image_names.push_back ( t );
      }

      if ( first_line )
      {
        first_line = false;
        num_images_row = image_names.size();
      }
      else
      {
        if ( int ( image_names.size() ) != num_images_row )
        {
          names.clear();
          cerr<<"Corrupted file"<<endl;
          return false;
        }
      }
    }
  }

  f.close();

  if ( first_line )
  {
    return false;
  }
  else
  {
    return true;
  }
}

bool readFileNamesFromFolder( const string& input_folder_name, vector< string >& names )
{
  names.clear();
  if (!input_folder_name.empty())
  {
    path p(input_folder_name);
    for(auto& entry : make_iterator_range(directory_iterator(p), {}))
      names.push_back(entry.path().string());
    std::sort(names.begin(), names.end());
    return true;
  }
  else
  {
    return false;
  }
}
