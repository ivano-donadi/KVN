#include "mesh_converter.h"

#include <iostream>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

using namespace OpenMesh;
typedef TriMesh_ArrayKernelT<>  Mesh;

#define MESH_CONVERTER_CHKOPT( Option ) \
  std::cout << "Provides " << #Option \
            << (opt.check(IO::Options:: Option)?": yes\n":": no\n")


bool MeshConverter::convert( const std::string &input_fn, const std::string &output_fn )
{
  Mesh  mesh;
  IO::Options opt;

  mesh.request_vertex_normals();
  mesh.request_face_normals();
  mesh.request_vertex_colors();
  mesh.request_face_colors();
  mesh.request_vertex_texcoords2D();

  opt += IO::Options::VertexNormal;
  opt += IO::Options::FaceNormal;
  opt += IO::Options::VertexColor;
  opt += IO::Options::FaceColor;
  opt += IO::Options::VertexTexCoord;

  if ( ! IO::read_mesh(mesh, input_fn, opt))
    return false;

  if( verbose_output_ )
  {
    // Show options
    std::cout << "File " << input_fn << std::endl;
    std::cout << "Is binary: "
              << (opt.check(IO::Options::Binary) ? " yes\n" : " no\n");

    // Mesh stats
    std::cout << "Num. vertices: " << mesh.n_vertices() << std::endl;
    std::cout << "Num. edges   : " << mesh.n_faces() << std::endl;
    std::cout << "Num. faces   : " << mesh.n_faces() << std::endl;

    MESH_CONVERTER_CHKOPT( VertexNormal    );
    MESH_CONVERTER_CHKOPT( VertexColor    );
    MESH_CONVERTER_CHKOPT( FaceNormal     );
    MESH_CONVERTER_CHKOPT( FaceColor      );
    MESH_CONVERTER_CHKOPT( VertexTexCoord );
  }

  // In case, rescale the mesh
  if( scale_ != 1.0 )
  {
    // (Linearly) iterate over all vertices
    for ( Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it!=mesh.vertices_end(); ++v_it )
    {
      OpenMesh::DefaultTraits::Point &p = mesh.point ( *v_it );
      p *= scale_;
    }
  }

  switch (output_)
  {
    case BINARY_OUT:
      opt += IO::Options::Binary;
      break;
    case ASCII_OUT:
      opt -= IO::Options::Binary;
      break;
    default:
      break;
  }

  if ( !IO::write_mesh( mesh, output_fn, opt ) )
    return false;
  else
    return true;
}

