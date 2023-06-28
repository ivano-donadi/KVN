#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <iostream>
#include <functional>

#include "cv_ext/macros.h"

#include <boost/concept_check.hpp>
#include <boost/thread/locks.hpp>

#include <OpenMesh/Core/IO/MeshIO.hh>
#include "raster_object_model3D.h"

// Include GLEW
#include <GL/glew.h>

// Include GLFW
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

// Include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>

#include "opengl_shaders.h"

using namespace glm;
using namespace std;
using namespace cv;


// TODO DEBUG CODE
//Mat g_contour_img;

// Useful for glm backward compatibility (older glm version use degrees in place of radians)
#ifndef GLM_FORCE_RADIANS
#define GLM_FORCE_RADIANS
#endif

#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
typedef OpenMesh::TriMesh_ArrayKernelT<>  Mesh;

static void checkFramebufferStatus()
{
  GLenum status = glCheckFramebufferStatus ( GL_FRAMEBUFFER );
  switch ( status )
  {
  case GL_FRAMEBUFFER_COMPLETE:
    // cerr<<"RasterObjectModel3D::GL_FRAMEBUFFER_COMPLETE"<<endl;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
    cerr<<"RasterObjectModel3D::GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER"<<endl;
    break;
  case GL_FRAMEBUFFER_UNSUPPORTED:
    cerr<<"RasterObjectModel3D::GL_FRAMEBUFFER_UNSUPPORTED"<<endl;
    break;
  case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
    cerr<<"RasterObjectModel3D::GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT "<<endl;
    break;
  default:
    break;
  }
}

// Custom specialization of std::hash for a vertex coordinate injected in namespace std
namespace std
{
template<> struct hash<OpenMesh::DefaultTraits::Point>
{
  std::size_t operator()(OpenMesh::DefaultTraits::Point const& p) const noexcept
  {
    std::size_t h1 = std::hash<float>{}(p[0]);
    std::size_t h2 = std::hash<float>{}(p[1]);
    std::size_t h3 = std::hash<float>{}(p[2]);

    // Taken from boost::hash_combine
    h1 ^= h2 + 0x9e3779b9 + (h1<<6) + (h1>>2);
    h1 ^= h3 + 0x9e3779b9 + (h1<<6) + (h1>>2);

    return h1;
  }
};
}

class RasterObjectModel3D::MeshModel
{
public:
  Mesh mesh;
  GLFWwindow* window = 0;
  GLuint fbo = 0, depth_rbo = 0, normal_rbo = 0, color_rbo = 0;
  vector<GLfloat> vertex_buffer_data;
  vector<GLfloat> color_buffer_data;
  vector<GLfloat> face_normal_buffer_data;
  GLuint vertex_buffer = 0, color_buffer = 0, face_normal_buffer = 0;
  GLuint shader_program_id = 0;
  glm::mat4 persp;

  void clear()
  {
    mesh.clear();

    if( vertex_buffer )
    {
      glDeleteBuffers ( 1, &vertex_buffer );
      vertex_buffer = 0;
    }

    if( color_buffer )
    {
      glDeleteBuffers ( 1, &color_buffer );
      color_buffer = 0;
    }

    if( face_normal_buffer )
    {
      glDeleteBuffers ( 1, &face_normal_buffer );
      face_normal_buffer = 0;
    }

    if( shader_program_id )
    {
      glDeleteProgram ( shader_program_id );
      shader_program_id = 0;
    }

    if( depth_rbo )
    {
      glDeleteRenderbuffers ( 1, &depth_rbo );
      depth_rbo = 0;
    }

    if( normal_rbo )
    {
      glDeleteRenderbuffers ( 1, &normal_rbo );
      normal_rbo = 0;
    }

    if( color_rbo )
    {
      glDeleteRenderbuffers ( 1, &color_rbo );
      color_rbo = 0;
    }
    if( fbo )
    {
      glDeleteFramebuffers ( 1, &fbo );
      fbo = 0;
    }

    vertex_buffer_data.clear();
    color_buffer_data.clear();
    face_normal_buffer_data.clear();
  }

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};


RasterObjectModel3D::RasterObjectModel3D()
{
  mesh_model_ptr_ = std::unique_ptr< MeshModel > ( new MeshModel () );
}

RasterObjectModel3D::~RasterObjectModel3D()
{
  if( raster_initiliazed_ )
  {
    mesh_model_ptr_->clear();
    glfwTerminate();
  }
}

bool RasterObjectModel3D::setModelFile ( const string& filename )
{
  mesh_model_ptr_->clear();

  OpenMesh::IO::Options opt;

  if( has_color_ )
  {
    if( vertex_color_ == cv::Scalar(-1) )
    {
      mesh_model_ptr_->mesh.request_vertex_colors();
      opt += OpenMesh::IO::Options::VertexColor;
    }
  }

  if ( !OpenMesh::IO::read_mesh ( mesh_model_ptr_->mesh, filename, opt ) )
  {
    cerr<<"RasterObjectModel3D::setModelFile() - Error opening file: "<<filename<<endl;
    return false;
  }

  if( !opt.vertex_has_color() && vertex_color_ == cv::Scalar(-1) )
    has_color_ = light_on_ = false;

  return true;
}

void RasterObjectModel3D::computeRaster()
{
  if( cam_model_.imgWidth() == 0 || cam_model_.imgHeight() == 0 )
  {
    cerr<<"RasterObjectModel3D::computeRaster() - invalid camera model (zero image width or image height)"<<endl;
    return;
  }

  model_mask_.create(cam_model_.imgWidth(), cam_model_.imgHeight(), DataType<uchar>::type);
  model_mask_.setTo(Scalar(0));
  mask_pts.clear();
  mask_pts.reserve(model_mask_.total());

  if( initOpenGL() )
  {    
    createShader();
    loadMesh();
    update();
    raster_initiliazed_ = true;
  }
}

bool RasterObjectModel3D::initOpenGL()
{
  // Initialise GLFW
  if ( !glfwInit() )
  {
    cerr<<"RasterObjectModel3D::initOpenGL() - Failed to initialize GLFW"<<endl;
    return false;
  }

  // Check these number
  glfwWindowHint ( GLFW_SAMPLES, 4 );
  glfwWindowHint ( GLFW_CONTEXT_VERSION_MAJOR, 2 );
  glfwWindowHint ( GLFW_CONTEXT_VERSION_MINOR, 1 );
  glfwWindowHint ( GLFW_VISIBLE, 0 );

  glm::mat4 view = glm::lookAt ( glm::vec3 ( 0, 0, 0 ),
                                 glm::vec3 ( 0, 0, 1 ),
                                 glm::vec3 ( 0, -1, 0 ) );

  float w = static_cast<float>(cam_model_.imgWidth()), h = static_cast<float>(cam_model_.imgHeight());
  float fx = static_cast<float>(cam_model_.fx()), fy = static_cast<float>(cam_model_.fy());
  float cx = static_cast<float>(cam_model_.cx()), cy = static_cast<float>(cam_model_.cy());

  float clip_a = (-render_z_far_ - render_z_near_)/(render_z_far_ - render_z_near_),
        clip_b = -2.0f*render_z_far_*render_z_near_/(render_z_far_ - render_z_near_);

  // OpenGL projection matrix originally taken by:
  // https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
  // We add here a 0.5 offset to the the principal point (cx,cy) in order to force a round() operation during
  // rasterization (in the default state, in OpenGL a point is rasterized by truncating its xw and yw coordinates)

  glm::mat4 persp;

  persp[0][0] = 2.0f*fx/w; persp[1][0] = 0.0f;        persp[2][0] = (w - 2.0f*(cx + 0.5f))/w; persp[3][0] = 0.0f;
  persp[0][1] = 0.0f;      persp[1][1] = -2.0f*fy/h;  persp[2][1] = (h - 2.0f*(cy + 0.5f))/h; persp[3][1] = 0.0f;
  persp[0][2] = 0.0f;      persp[1][2] = 0.0f;        persp[2][2] = clip_a;          persp[3][2] = clip_b;
  persp[0][3] = 0.0f;      persp[1][3] = 0.0f;        persp[2][3] = -1.0f;           persp[3][3] = 0.0f;


  mesh_model_ptr_->persp = persp*view;

  render_win_size_.width = cam_model_.imgWidth();
  render_win_size_.height = cam_model_.imgHeight();

  depth_buf_data_.resize( render_win_size_.width * render_win_size_.height  );
  normal_buf_data_.resize( render_win_size_.width * render_win_size_.height );
  color_buf_data_.resize( render_win_size_.width * render_win_size_.height );

  depth_buf_ = cv::Mat( Size(render_win_size_.width, render_win_size_.height), cv::DataType< float >::type,
                        depth_buf_data_.data(), Mat::AUTO_STEP);

  normal_buf_ = cv::Mat( Size(render_win_size_.width, render_win_size_.height), cv::DataType< cv::Vec3f >::type,
                         normal_buf_data_.data(), Mat::AUTO_STEP);

  color_buf_ = cv::Mat( Size(render_win_size_.width, render_win_size_.height), cv::DataType< cv::Vec3b >::type,
                        color_buf_data_.data(), Mat::AUTO_STEP);

  depth_transf_a_ = 2.0f * render_z_near_*render_z_far_;
  depth_transf_b_ = render_z_far_ + render_z_near_;
  depth_transf_c_ = render_z_far_ - render_z_near_;      

  // Create an hidden window
  mesh_model_ptr_->window = glfwCreateWindow ( render_win_size_.width, render_win_size_.height, "Render", NULL, NULL );
  
  if ( mesh_model_ptr_->window == NULL )
  {
    cerr<<"RasterObjectModel3D::initOpenGL() - Failed to open GLFW window"<<endl;
    mesh_model_ptr_->clear();
    glfwTerminate();
    return false;
  }

  glfwMakeContextCurrent(mesh_model_ptr_->window);

  // Initialize GLEW
  if ( glewInit() != GLEW_OK )
  {
    cerr<<"RasterObjectModel3D::initOpenGL() - Failed to initialize GLEW"<<endl;
    mesh_model_ptr_->clear();
    glfwTerminate();
    return false;
  }
  
  return true;
}

void RasterObjectModel3D::update()
{
//  cv_ext::BasicTimer t;

  glfwMakeContextCurrent(mesh_model_ptr_->window);

//   static uint64_t c_timer = 0, n_sample = 1;
//   cv_ext::BasicTimer timer;
  glBindFramebuffer(GL_FRAMEBUFFER, mesh_model_ptr_->fbo );
  glViewport(0,0,render_win_size_.width,render_win_size_.height);

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUseProgram ( mesh_model_ptr_->shader_program_id );
  
  glm::mat4 view = glm::translate ( glm::mat4 ( 1.0f ), glm::vec3 ( view_t_vec_( 0 ), view_t_vec_( 1 ), view_t_vec_( 2 ) ) );
  Eigen::Quaternionf view_rq ( view_r_mat_ );
  glm::quat q_view( view_rq.w(), view_rq.x(), view_rq.y(), view_rq.z() );
  // Compose the RT transformation
  view=view*glm::mat4_cast ( q_view );

  // TODO WARNING: Identity matrix for now
  glm::mat4 model = glm::mat4(1.0);

  glUniformMatrix4fv ( glGetUniformLocation ( mesh_model_ptr_->shader_program_id, "perspective" ), 1,
                       GL_FALSE, & ( mesh_model_ptr_->persp[0][0] ) );
  glUniformMatrix4fv(glGetUniformLocation( mesh_model_ptr_->shader_program_id, "view"), 1,
                     GL_FALSE, &view[0][0]);
  glUniformMatrix4fv(glGetUniformLocation( mesh_model_ptr_->shader_program_id, "model"), 1,
                     GL_FALSE, &model[0][0]);

  glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dist.k0" ), cam_model_.distK0() );
  glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dist.k1" ), cam_model_.distK1() );
  glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dist.k2" ), cam_model_.distK2() );
  glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dist.k3" ), cam_model_.distK3() );
  glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dist.k4" ), cam_model_.distK4() );
  glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dist.k5" ), cam_model_.distK5());
  glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dist.px" ), cam_model_.distPx() );
  glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dist.py" ), cam_model_.distPy() );


  // If true, color ha been changed
  if( prev_vertex_color_ != vertex_color_ )
  {
    prev_vertex_color_ = vertex_color_;
    Scalar v_c = vertex_color_/255.0;
    for(auto iter =  mesh_model_ptr_->color_buffer_data.begin();
             iter != mesh_model_ptr_->color_buffer_data.end(); )
    {
      *iter++ = v_c[0];
      *iter++ = v_c[1];
      *iter++ = v_c[2];
    }

    glNamedBufferSubData(	mesh_model_ptr_->color_buffer, 0,
                          sizeof ( GL_FLOAT ) *mesh_model_ptr_->color_buffer_data.size(),
                          mesh_model_ptr_->color_buffer_data.data() );
  }

  GLuint vertex_pos_id = 0, vertex_normal_id = 0, vertex_color_id = 0;
  
  // 1rst attribute buffer : vertices
  vertex_pos_id = glGetAttribLocation ( mesh_model_ptr_->shader_program_id, "vertex_pos" ),
  glEnableVertexAttribArray ( vertex_pos_id );
  glBindBuffer ( GL_ARRAY_BUFFER, mesh_model_ptr_->vertex_buffer );
  glVertexAttribPointer ( vertex_pos_id, 3, GL_FLOAT, GL_FALSE, 0, ( void* ) 0 );

  // 2nd attribute buffer : vertices normals
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  vertex_normal_id = glGetAttribLocation ( mesh_model_ptr_->shader_program_id, "vertex_normal" );
  glEnableVertexAttribArray( vertex_normal_id );
  glBindBuffer(GL_ARRAY_BUFFER, mesh_model_ptr_->face_normal_buffer);
  glVertexAttribPointer( vertex_normal_id, 3, GL_FLOAT, GL_FALSE, 0, (void*)0 );

  if( has_color_ )
  {
    // 3rd attribute buffer : vertices RGB colors
    glDrawBuffer(GL_COLOR_ATTACHMENT1);
    vertex_color_id = glGetAttribLocation ( mesh_model_ptr_->shader_program_id, "vertex_color" );
    glEnableVertexAttribArray( vertex_color_id );
    glBindBuffer(GL_ARRAY_BUFFER, mesh_model_ptr_->color_buffer );
    glVertexAttribPointer( vertex_color_id, 3, GL_FLOAT, GL_FALSE, 0, (void*)0 );

    if( light_on_ )
    {
      glm::vec3 light_pos = glm::vec3(point_light_pos_.x, point_light_pos_.y, point_light_pos_.z),
                light_dir = glm::vec3(light_dir_.x, light_dir_.y, light_dir_.z);
      // Set material properties
      glUniform1i( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "material.diffuse" ), 0 );
      glUniform1i( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "material.specular" ), 1 );
      glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "material.shininess" ), 32.0f );

      GLint view_pos_id = glGetUniformLocation( mesh_model_ptr_->shader_program_id, "view_pos" );
      glUniform3f( view_pos_id, view_t_vec_( 0 ), view_t_vec_( 1 ), view_t_vec_( 2 ) );

      // Directional lights
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dir_lights[0].direction" ), light_dir.x, light_dir.y, light_dir.z );
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dir_lights[0].ambient" ), 0.05f, 0.05f, 0.05f );
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dir_lights[0].diffuse" ), 0.4f, 0.4f, 0.4f );
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "dir_lights[0].specular" ), 0.5f, 0.5f, 0.5f );

      // Point light 1
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[0].position" ), light_pos.x, light_pos.y, light_pos.z);
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[0].ambient" ), 0.05f, 0.05f, 0.05f );
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[0].diffuse" ), 0.8f, 0.8f, 0.8f );
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[0].specular" ), 1.0f, 1.0f, 1.0f );
      glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[0].constant" ), 1.0f );
      glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[0].linear" ), 0.09f );
      glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[0].quadratic" ), 0.032f );

      // Point light 2
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[1].position" ), -light_pos.x, -light_pos.y, -light_pos.z );
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[1].ambient" ), 0.05f, 0.05f, 0.05f );
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[1].diffuse" ), 0.8f, 0.8f, 0.8f );
      glUniform3f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[1].specular" ), 1.0f, 1.0f, 1.0f );
      glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[1].constant" ), 1.0f );
      glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[1].linear" ), 0.09f );
      glUniform1f( glGetUniformLocation( mesh_model_ptr_->shader_program_id, "point_lights[1].quadratic" ), 0.032f );
    }
  }


  // Draw the triangles
  GLuint attachments[2] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
  glDrawBuffers(2, attachments);

  glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
  glDrawArrays ( GL_TRIANGLES, 0, mesh_model_ptr_->vertex_buffer_data.size() );

  glDisableVertexAttribArray ( vertex_pos_id );
  glDisableVertexAttribArray ( vertex_normal_id );
  glDisableVertexAttribArray ( vertex_color_id );

  glReadPixels ( 0, 0, render_win_size_.width, render_win_size_.height,
                 GL_DEPTH_COMPONENT, GL_FLOAT, ( GLvoid * ) depth_buf_data_.data() );

  // To read out a specific buffer, you have to specify it
  glReadBuffer(GL_COLOR_ATTACHMENT0);
  glReadPixels ( 0, 0, render_win_size_.width, render_win_size_.height,
                 GL_BGR, GL_FLOAT, ( GLvoid * ) normal_buf_data_.data() );

  // TODO debug code
//  cv_ext::showImage(depth_buf_, "depth_buffer", true, 1);
//  cv_ext::showImage(normal_buf_, "normal_buffer", true, 1);

  if( has_color_ )
  {
    // To read out a specific buffer, you have to specify it
    glReadBuffer(GL_COLOR_ATTACHMENT1);
    glReadPixels ( 0, 0, render_win_size_.width, render_win_size_.height,
                   GL_BGR, GL_UNSIGNED_BYTE, ( GLubyte * ) color_buf_data_.data() );

//    cv_ext::showImage(color_buf_, "color_buffer", true, 1);
  }
  
  glBindFramebuffer(GL_FRAMEBUFFER, 0  );

//  cout<<"OpenGL stuff time: "<<t.elapsedTimeMs()<<endl;
//  t.reset();

  Mesh &mesh = mesh_model_ptr_->mesh;

  vis_pts_.clear();
  vis_d_pts_.clear();
  vis_segs_.clear();
  vis_d_segs_.clear();

  vis_pts_p_ = &vis_pts_;
  vis_d_pts_p_ = &vis_d_pts_;
  vis_segs_p_ = &vis_segs_;
  vis_d_segs_p_ = &vis_d_segs_;

  clearMask();

  // Iterate over all faces

  Eigen::Matrix4f rt_mat( Eigen::Matrix4f::Identity() );
  rt_mat.block<3,3> ( 0,0 ) = view_r_mat_;
  rt_mat.col ( 3 ).head ( 3 ) = view_t_vec_;

  for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
  {
    Mesh::FaceHalfedgeIter fh_it = mesh.fh_iter ( *f_it );
    Mesh::Normal face_normal = mesh.calc_face_normal ( *f_it );

    Eigen::Vector3f rotated_face_normal_eigen = view_r_mat_*Eigen::Map < Eigen::Vector3f >( face_normal.data() );

//    Eigen::Vector3d rotated_face_normal_eigen ( ( double ) face_normal[0], ( double ) face_normal[1], ( double ) face_normal[2] );
//    rotated_face_normal_eigen=RT.block<3,3> ( 0,0 ) *rotated_face_normal_eigen;
    //Vec3f rotated_face_normal((float)rotated_face_normal_eigen(0),(float)rotated_face_normal_eigen(1),(float)rotated_face_normal_eigen(2));

    for ( ; fh_it.is_valid(); ++fh_it )
    {
      const OpenMesh::VertexHandle from_ph = mesh.from_vertex_handle ( *fh_it ),
                                   to_ph = mesh.to_vertex_handle ( *fh_it );

      const OpenMesh::DefaultTraits::Point &from_p = mesh.point ( from_ph ),
                                            &to_p = mesh.point ( to_ph );

      Point3f p0 ( from_p[0], from_p[1], from_p[2] ),
              p1 ( to_p[0], to_p[1], to_p[2] );
              
      Mesh::HalfedgeHandle opposite_heh = mesh.opposite_halfedge_handle ( *fh_it );
      Mesh::FaceHandle opposite_fh = mesh.face_handle ( opposite_heh );
      
      if( !opposite_fh.is_valid() )
      {
        addVisibleSegment(p0, p1);
      }
      else
      {
        Mesh::Normal near_face_normal = mesh.calc_face_normal ( opposite_fh );
        OpenMesh::Vec3f normals_diff = near_face_normal - face_normal;
        float norm_dist = float ( normals_diff[0]*normals_diff[0] +
                                  normals_diff[1]*normals_diff[1] +
                                  normals_diff[2]*normals_diff[2] );

        if ( norm_dist > normal_epsilon2_ )
        {
          addVisibleSegment(p0, p1);
        }
        else
        {
          Eigen::Vector3f
              pfrom = view_r_mat_*Eigen::Map< Eigen::Vector3f >( reinterpret_cast<float *>(&p0) ) + view_t_vec_,
              pto = view_r_mat_*Eigen::Map< Eigen::Vector3f >( reinterpret_cast<float *>(&p1) ) + view_t_vec_;
          Eigen::Vector3f center ( ( pfrom+pto )/2 );
          float dot=center.dot ( rotated_face_normal_eigen );

          Eigen::Vector3f rotated_near_face_normal_eigen =
              view_r_mat_*Eigen::Map< Eigen::Vector3f >( near_face_normal.data() );
          float dot_near=center.dot ( rotated_near_face_normal_eigen );

//          Eigen::Vector4d pfrom ( p0.x, p0.y, p0.z, 1.0f );
//          Eigen::Vector4d pto ( double(p1.x), double(p1.y), double(p1.z), 1.0f );
//          pfrom=RT*pfrom;
//          pto=RT*pto;
//          pfrom /= pfrom ( 3 );
//          pto /= pto ( 3 );
//          Eigen::Vector3d centr ( ( pfrom+pto ).head ( 3 ) /2 );
//          double dot=centr.dot ( rotated_face_normal_eigen );

//          Eigen::Vector3d rotated_near_face_normal_eigen ( ( double ) near_face_normal[0], ( double ) near_face_normal[1], ( double ) near_face_normal[2] );
//          rotated_near_face_normal_eigen=RT.block<3,3> ( 0,0 ) *rotated_near_face_normal_eigen;
//          double dot_near=centr.dot ( rotated_near_face_normal_eigen );


          if ( ( dot*dot_near ) < 0.0f )
          {
            addVisibleSegment(p0, p1);
          }
        }
      }
    }
  }


//  // TODO DEBUG CODE
//  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
//  g_contour_img.create( Size(cols, rows) , DataType<uchar>::type );
//  g_contour_img.setTo( cv::Scalar(0) );
//
//  vector<Vec4f> raster_segs;
//  this->projectRasterSegments( raster_segs );
//  cv_ext::drawSegments( g_contour_img, raster_segs );
//
//  std::vector<std::vector<cv::Point> > contours;
//  vector<Vec4i> hierarchy;
//  cv::findContours( g_contour_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//  cv::drawContours(g_contour_img, contours, -1, cv::Scalar(255),-1);
////  cv_ext::showImage(g_contour_img, "ccc", true, 10);

  raster_updated_ = true;
//  cout<<"Projection time: "<<t.elapsedTimeMs()<<endl;
//  cv_ext::showImage(color_buf_, "color_buffer", true, 1);
//   cout<<"update() time: "<<(c_timer += timer.elapsedTimeUs())/n_sample++<<endl;
}

const vector< cv::Point3f >& RasterObjectModel3D::getPoints(bool only_visible_points) const
{
  if(only_visible_points)
    return *vis_pts_p_;
  else
    return pts_;
}

const vector< cv::Point3f >& RasterObjectModel3D::getDPoints(bool only_visible_points) const
{
  if(only_visible_points)
    return *vis_d_pts_p_;
  else
    return d_pts_;
}

const vector< cv::Vec6f >& RasterObjectModel3D::getSegments(bool only_visible_segments) const
{
  if(only_visible_segments)
    return *vis_segs_p_;
  else
    return segs_;
}

const vector< cv::Point3f >& RasterObjectModel3D::getDSegments(bool only_visible_segments) const
{
  if(only_visible_segments)
    return *vis_d_segs_p_;
  else
    return d_segs_;
}

void RasterObjectModel3D::createShader()
{
  glGenFramebuffers ( 1, &mesh_model_ptr_->fbo );
  glBindFramebuffer ( GL_FRAMEBUFFER, mesh_model_ptr_->fbo );

  // Render buffer as depth buffer
  glGenRenderbuffers ( 1, &mesh_model_ptr_->depth_rbo );
  glBindRenderbuffer ( GL_RENDERBUFFER, mesh_model_ptr_->depth_rbo );
  glRenderbufferStorage ( GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F, render_win_size_.width, render_win_size_.height );
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  // Attach render buffer to the fbo as depth buffer
  glFramebufferRenderbuffer ( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER,  mesh_model_ptr_->depth_rbo );

  // Render buffer as "normal" RGB buffer
  glGenRenderbuffers ( 1, &mesh_model_ptr_->normal_rbo );
  glBindRenderbuffer(GL_RENDERBUFFER, mesh_model_ptr_->normal_rbo );
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, render_win_size_.width, render_win_size_.height );
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  // Attach render buffer to the fbo as normal buffer
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, mesh_model_ptr_->normal_rbo );

  if( has_color_ )
  {
    // Render buffer as color RGB buffer
    glGenRenderbuffers(1, &mesh_model_ptr_->color_rbo );
    glBindRenderbuffer(GL_RENDERBUFFER, mesh_model_ptr_->color_rbo );
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA, render_win_size_.width, render_win_size_.height );
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    // Attach render buffer to the fbo as color buffer
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_RENDERBUFFER, mesh_model_ptr_->color_rbo );
  }

  checkFramebufferStatus();

  // Enable depth test
  glEnable ( GL_DEPTH_TEST );
  // Accept fragment if it closer to the camera than the former one
  glDepthFunc ( GL_LESS );

  GLuint vertex_shader_id = glCreateShader ( GL_VERTEX_SHADER ), fragment_shader_id = 0;
  string shading_version ( ( char* ) glGetString ( GL_SHADING_LANGUAGE_VERSION ) );
  cout<<"RasterObjectModel3D::createShader() - Shader_version: "<<shading_version<<endl;

  string sh_ver="";
  for ( int i=0; i < int(shading_version.size()); i++ )
    if ( isdigit ( shading_version[i] ) ) sh_ver.push_back ( shading_version[i] );

  string vertex_shader_code;

  if( has_color_ )
  {
    if( light_on_ )
      vertex_shader_code = string( SHADED_OBJECT_VERTEX_SHADER_CODE ( sh_ver ) );  
    else
      vertex_shader_code = string( COLORED_OBJECT_VERTEX_SHADER_CODE ( sh_ver ) );    
  }  
  else
    vertex_shader_code = string( OBJECT_VERTEX_SHADER_CODE ( sh_ver ) );

  GLint result = GL_FALSE;

  // Compile Vertex Shader
  char const *vertex_source_ptr = vertex_shader_code.c_str();
  glShaderSource ( vertex_shader_id, 1, &vertex_source_ptr , NULL );
  glCompileShader ( vertex_shader_id );

  // Check Vertex Shader
  int info_log_length;
  glGetShaderiv ( vertex_shader_id, GL_COMPILE_STATUS, &result );
  if( result == GL_FALSE )
    cerr<<"RasterObjectModel3D::createShader() : vertex shader compile operation failed"<<endl;

  glGetShaderiv ( vertex_shader_id, GL_INFO_LOG_LENGTH, &info_log_length );
  if ( info_log_length > 1 )
  {
    vector<char> vertex_shader_error_message ( info_log_length + 1 );
    glGetShaderInfoLog ( vertex_shader_id, info_log_length, NULL, &vertex_shader_error_message[0] );
    cout<<"RasterObjectModel3D::createShader() - ShaderInfoLog Vertex Shader: "<<&vertex_shader_error_message[0]<<endl;
  }

  // Link the program
  mesh_model_ptr_->shader_program_id = glCreateProgram();

  glAttachShader ( mesh_model_ptr_->shader_program_id, vertex_shader_id );

  if( has_color_ )
  {
    fragment_shader_id = glCreateShader ( GL_FRAGMENT_SHADER );
    string fragment_shader_code;
    if( light_on_ )
      fragment_shader_code = string( SHADED_OBJECT_FRAGMENT_SHADER_CODE( sh_ver ) );
    else
      fragment_shader_code = string( COLORED_OBJECT_FRAGMENT_SHADER_CODE( sh_ver ) );

    // Compile Fragment Shader
    char const * fragment_source_ptr = fragment_shader_code.c_str();
    glShaderSource ( fragment_shader_id, 1, &fragment_source_ptr , NULL );
    glCompileShader ( fragment_shader_id );

    // Check Fragment Shader
    glGetShaderiv(fragment_shader_id, GL_COMPILE_STATUS, &result);
    if( result == GL_FALSE )
      cerr<<"RasterObjectModel3D::createShader() : fragment shader compile operation failed"<<endl;

    glGetShaderiv(fragment_shader_id, GL_INFO_LOG_LENGTH, &info_log_length);
    if ( info_log_length > 1 )
    {
      vector<char> fragment_shader_error_message(info_log_length + 1);
      glGetShaderInfoLog(fragment_shader_id, info_log_length, NULL, &fragment_shader_error_message[0]);
      cout<<"RasterObjectModel3D::createShader() - ShaderInfoLog Fragment Shader: "<<&fragment_shader_error_message[0]<<endl;
    }
    glAttachShader(mesh_model_ptr_->shader_program_id, fragment_shader_id);
  }

  glLinkProgram ( mesh_model_ptr_->shader_program_id );

  // Check the program
  glGetProgramiv(mesh_model_ptr_->shader_program_id, GL_LINK_STATUS, &result);
  if( result == GL_FALSE )
    cerr<<"RasterObjectModel3D::createShader() : shader link operation failed"<<endl;

  glGetProgramiv(mesh_model_ptr_->shader_program_id, GL_INFO_LOG_LENGTH, &info_log_length);
  if ( info_log_length > 1 ){
    vector<char> program_error_message(info_log_length + 1);
    glGetProgramInfoLog(mesh_model_ptr_->shader_program_id, info_log_length, NULL, &program_error_message[0]);
    cout<<"RasterObjectModel3D::createShader() - ShaderInfoLog Program: "<<&program_error_message[0]<<endl;
  }

  glDeleteShader ( vertex_shader_id );
  glDeleteShader ( fragment_shader_id );

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RasterObjectModel3D::loadMesh()
{
  mesh_model_ptr_->vertex_buffer_data.clear();
  mesh_model_ptr_->color_buffer_data.clear();
  mesh_model_ptr_->face_normal_buffer_data.clear();

  Mesh &mesh = mesh_model_ptr_->mesh;

  pts_.clear();
  d_pts_.clear();
  segs_.clear();
  d_segs_.clear();

  float multiplier = 1.0f/unit_meas_;

  OpenMesh::DefaultTraits::Point p_min( numeric_limits<float>::max(),
                                        numeric_limits<float>::max(),
                                        numeric_limits<float>::max()),
                                 p_max( numeric_limits<float>::lowest(),
                                        numeric_limits<float>::lowest(),
                                        numeric_limits<float>::lowest() );

  std::vector< OpenMesh::VertexHandle > unique_handles;
  std::unordered_map< OpenMesh::DefaultTraits::Point, int > points_map;
  std::unordered_map< OpenMesh::VertexHandle, int > handles_map;

  unique_handles.reserve( mesh.n_vertices() );

  std::cout<<"Checking for duplicate vertices..."<<std::endl;

  for (Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it != mesh.vertices_end(); ++v_it )
  {

    if( *v_it == Mesh::InvalidVertexHandle )
      continue;

    OpenMesh::DefaultTraits::Point &p = mesh.point(*v_it);

    // Check if the vertex has been already included in duplicates-free vertex set
    auto pm_iter = points_map.find (p);

    // If not, include the new vertex in the duplicates-free vertex set
    if ( pm_iter == points_map.end() )
    {
      points_map[p] = unique_handles.size();
      handles_map[*v_it] = unique_handles.size();
      unique_handles.push_back(*v_it);
    }
    // Otherwise, get the index of the coincident point in the duplicates-free vertex set
    else
    {
      handles_map[*v_it] = points_map[p];
    }
  }

  if( unique_handles.size() != mesh.n_vertices() )
  {
    std::cout<<"Merging coincident vertices "<<std::endl;
    Mesh clean_mesh;
    bool set_color = has_color_ && vertex_color_ == cv::Scalar(-1);
    for( auto &h : unique_handles )
    {
      OpenMesh::VertexHandle new_handle = clean_mesh.add_vertex( mesh.point ( h ) );
      if( set_color )
        clean_mesh.set_color( new_handle, mesh.color( h ) );

      h = new_handle;
    }

    std::vector<Mesh::VertexHandle> face_vhandles;
    face_vhandles.reserve(3);
    for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
    {
      Mesh::FaceVertexIter fv_it = mesh.fv_iter(*f_it);
      face_vhandles.clear();
      bool add_face = true;
      for( int i = 0; i < 3; i++, fv_it++ )
      {
        auto h = handles_map.find(*fv_it);
        if ( h  != handles_map.end() )
        {
          face_vhandles.push_back(  unique_handles[h->second] );
          // add_face &= face_vhandles.back() != Mesh::InvalidVertexHandle; // Useful? Check
        }
        else
        {
          add_face = false;
          break;
        }
      }

      if( add_face ) clean_mesh.add_face(face_vhandles);
    }
    // Replace the original mesh
    mesh = clean_mesh;
  }
    // (Linearly) iterate over all vertices
  for ( Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it!=mesh.vertices_end(); ++v_it )
  {
    OpenMesh::DefaultTraits::Point &p = mesh.point ( *v_it );
    p *= multiplier;

    if ( p[0] > p_max[0] ) p_max[0] = p[0];
    else if ( p[0] < p_min[0] ) p_min[0] = p[0];
    if ( p[1] > p_max[1] ) p_max[1] = p[1];
    else if ( p[1] < p_min[1] ) p_min[1] = p[1];
    if ( p[2] > p_max[2] ) p_max[2] = p[2];
    else if ( p[2] < p_min[2] ) p_min[2] = p[2];
  }

  if ( centroid_orig_offset_ == CENTROID_ORIG_OFFSET )
  {
    OpenMesh::DefaultTraits::Point mean(0,0,0);
    int n_vtx = 0;
    for ( Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it!=mesh.vertices_end(); ++v_it )
    {
      OpenMesh::DefaultTraits::Point &p = mesh.point ( *v_it );
      mean += p;
      n_vtx++;
    }
    mean[0] /= float ( n_vtx );
    mean[1] /= float ( n_vtx );
    mean[2] /= float ( n_vtx );

    orig_offset_ = -Point3f ( mean[0], mean[1], mean[2] );
  }
  else if ( centroid_orig_offset_ == BOUNDING_BOX_CENTER_ORIG_OFFSET )
  {
    orig_offset_ = -Point3f (p_min[0] + ( p_max[0] - p_min[0] ) /2.0f,
                             p_min[1] + ( p_max[1] - p_min[1] ) /2.0f,
                             p_min[2] + ( p_max[2] - p_min[2] ) /2.0f );
  }

  vertices_.clear();
  vertices_.reserve( mesh.n_vertices() );
  OpenMesh::DefaultTraits::Point orig_offset ( orig_offset_.x, orig_offset_.y, orig_offset_.z );
  for ( Mesh::VertexIter v_it = mesh.vertices_sbegin(); v_it!=mesh.vertices_end(); ++v_it )
  {
    OpenMesh::DefaultTraits::Point &p = mesh.point ( *v_it );
    p += orig_offset;
    vertices_.emplace_back( p[0], p[1], p[2] );
  }

  p_min += orig_offset;
  p_max += orig_offset;

  bbox_ = orig_bbox_ = cv_ext::Box3f( cv::Point3f(p_min[0], p_min[1], p_min[2]),
                                      cv::Point3f(p_max[0], p_max[1], p_max[2]) );

  // Iterate over all faces
  for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
  {
    Mesh::FaceVertexIter fv_it = mesh.fv_iter ( *f_it );
    Point3f p0, p1, p2;

    OpenMesh::DefaultTraits::Point &v0 = mesh.point ( *fv_it++ );

    mesh_model_ptr_->vertex_buffer_data.push_back ( v0[0] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v0[1] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v0[2] );

    p0.x = v0[0];
    p0.y = v0[1];
    p0.z = v0[2];

    OpenMesh::DefaultTraits::Point &v1 = mesh.point ( *fv_it++ );

    mesh_model_ptr_->vertex_buffer_data.push_back ( v1[0] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v1[1] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v1[2] );

    p1.x = v1[0];
    p1.y = v1[1];
    p1.z = v1[2];

    OpenMesh::DefaultTraits::Point &v2 = mesh.point ( *fv_it );

    mesh_model_ptr_->vertex_buffer_data.push_back ( v2[0] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v2[1] );
    mesh_model_ptr_->vertex_buffer_data.push_back ( v2[2] );

    p2.x = v2[0];
    p2.y = v2[1];
    p2.z = v2[2];

    addSegment(p0, p1);
    addSegment(p1, p2);
    addSegment(p2, p0);

  }

  glDeleteBuffers ( 1, & ( mesh_model_ptr_->vertex_buffer ) );
  glGenBuffers ( 1, & ( mesh_model_ptr_->vertex_buffer ) );
  glBindBuffer ( GL_ARRAY_BUFFER, mesh_model_ptr_->vertex_buffer );
  glBufferData ( GL_ARRAY_BUFFER, sizeof ( GL_FLOAT ) *mesh_model_ptr_->vertex_buffer_data.size(),
                 mesh_model_ptr_->vertex_buffer_data.data(), GL_STATIC_DRAW );

  for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
  {
    Mesh::Normal face_normal = mesh.calc_face_normal ( *f_it );

    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[0]) );
    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[1]) );
    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[2]) );

    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[0]) );
    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[1]) );
    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[2]) );

    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[0]) );
    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[1]) );
    mesh_model_ptr_->face_normal_buffer_data.push_back ( GLfloat(face_normal[2]) );
  }

  glDeleteBuffers ( 1, & ( mesh_model_ptr_->face_normal_buffer ) );
  glGenBuffers ( 1, & ( mesh_model_ptr_->face_normal_buffer ) );
  glBindBuffer ( GL_ARRAY_BUFFER, mesh_model_ptr_->face_normal_buffer );
  glBufferData (GL_ARRAY_BUFFER, sizeof ( GL_FLOAT ) *mesh_model_ptr_->face_normal_buffer_data.size(),
                mesh_model_ptr_->face_normal_buffer_data.data(), GL_STATIC_DRAW );

  if( has_color_ )
  {
    GLenum color_buffer_usage;

    if( vertex_color_ == cv::Scalar(-1) )
    {
      color_buffer_usage = GL_STATIC_DRAW;
      for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
      {
        Mesh::FaceVertexIter fv_it = mesh.fv_iter ( *f_it );

        const OpenMesh::DefaultTraits::Color &c0 = mesh.color( *fv_it++ );

        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c0[0])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c0[1])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c0[2])/255.0f );

        const OpenMesh::DefaultTraits::Color &c1 = mesh.color( *fv_it++ );

        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c1[0])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c1[1])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c1[2])/255.0f );

        const OpenMesh::DefaultTraits::Color &c2 = mesh.color( *fv_it++ );

        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c2[0])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c2[1])/255.0f );
        mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(c2[2])/255.0f );

      }
    }
    else
    {
      color_buffer_usage = GL_DYNAMIC_DRAW;
      Scalar v_c = vertex_color_/255.0;
      for ( Mesh::FaceIter f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it )
      {
        for( int i = 0; i < 3; i++ )
        {
          mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(v_c[0]));
          mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(v_c[1]));
          mesh_model_ptr_->color_buffer_data.push_back ( GLfloat(v_c[2]));
        }
      }
    }

    glDeleteBuffers ( 1, & ( mesh_model_ptr_->color_buffer ) );
    glGenBuffers ( 1, & ( mesh_model_ptr_->color_buffer ) );
    glBindBuffer ( GL_ARRAY_BUFFER, mesh_model_ptr_->color_buffer );
    glBufferData ( GL_ARRAY_BUFFER, sizeof ( GL_FLOAT ) *mesh_model_ptr_->color_buffer_data.size(),
                  mesh_model_ptr_->color_buffer_data.data(), color_buffer_usage );
  }
}

void RasterObjectModel3D::clearMask()
{
  for(auto &p : mask_pts )
    model_mask_.at<uchar>(p.y,p.x) = 0;
  mask_pts.clear();
}

void RasterObjectModel3D::addSegment (Point3f& p0, Point3f& p1 )
{
  Point3f dir = p1 - p0;
  float len = std::sqrt ( dir.dot ( dir ) );
  if ( !len )
    return;

  int n_steps = len/step_;
  Point3f dp ( p0 + epsilon_*dir );
  if ( len >= min_seg_len_ )
  {
    segs_.push_back ( Vec6f ( p0.x, p0.y, p0.z, p1.x, p1.y, p1.z ) );
    d_segs_.push_back ( dp );
  }

  if ( !n_steps )
  {
    // Push at least the first point
    pts_.push_back ( p0 );
    d_pts_.push_back ( dp );
  }
  else
  {
    float dist_step = 1.0/n_steps;

    for ( int i = 0; i <= n_steps; i++ )
    {
      pts_.push_back ( p0 + ( i*dist_step ) *dir );
      d_pts_.push_back ( p0 + ( i*dist_step + epsilon_ ) *dir );
    }
  }
}

void RasterObjectModel3D::addVisibleSegment (Point3f& p0, Point3f& p1 )
{

//  // TODO Debug code!!
//  float q[4] = { ( float ) rq_view_.w(), ( float ) rq_view_.x(), ( float ) rq_view_.y(), ( float ) rq_view_.z() };
//  float t[3] = { ( float ) t_view_ ( 0 ), ( float ) t_view_ ( 1 ), ( float ) t_view_ ( 2 ) };
//
//  const float scene_pt0[3] = { p0.x, p0.y, p0.z }, scene_pt1[3] = { p1.x, p1.y, p1.z };
//  float proj_pt0[2], proj_pt1[2], depth0, depth1;
//
//  cam_model_.quatRTProject( q, t, scene_pt0, proj_pt0, depth0 );
//  cam_model_.quatRTProject( q, t, scene_pt1, proj_pt1, depth1 );
//
//  //  projectPoint ( p, proj_p, depth );
//  int x = cvRound(proj_pt0[0]), y = cvRound(proj_pt0[1]);
////  int x = proj_pt0[0], y = proj_pt0[1];
//  cv::Vec3b color = cv::Vec3b(255,255,255);
//
////  if(proj_pt0[0] - static_cast<int>(proj_pt0[0]) > .5 && proj_pt0[1] - static_cast<int>(proj_pt0[1]) > .5)
////  {
////    if( proj_pt0[0] - static_cast<int>(proj_pt0[0]) > proj_pt0[1] - static_cast<int>(proj_pt0[1]) )
////      x++;
////    else
////      y++;
////    color[0] = 0;
////    color[2] = 0;
////  }
////  else if(proj_pt0[0] - static_cast<int>(proj_pt0[0]) > .5)
////  {
////    x++;
////    color[1] = 0;
////  }
////  else if(proj_pt0[1] - static_cast<int>(proj_pt0[1]) > .5)
////  {
////    y++;
////    color[2] = 0;
////  }
//
//  // TODO Check why some points go outside the image
//  if( x < 0 || y < 0 || x >= cam_model_.imgWidth() || y >= cam_model_.imgHeight() )
//  {
//
//  }
//  else
//    color_buf_.at<cv::Vec3b>(y,x) = color;
//  x = cvRound(proj_pt1[0]), y = cvRound(proj_pt1[1]);
////  x = proj_pt1[0], y = proj_pt1[1];
//
//  color = cv::Vec3b(255,255,255);
//
////  if(proj_pt1[0] - static_cast<int>(proj_pt1[0]) > .5 && proj_pt1[1] - static_cast<int>(proj_pt1[1]) > .5)
////  {
////    if( proj_pt1[0] - static_cast<int>(proj_pt1[0]) > proj_pt1[1] - static_cast<int>(proj_pt1[1]) )
////      x++;
////    else
////      y++;
////    color[1] = 0;
////    color[2] = 0;
////  }
////  if(proj_pt1[0] - static_cast<int>(proj_pt1[0]) > .5)
////  {
////    x++;
////    color[1] = 0;
////  }
////  if(proj_pt1[1] - static_cast<int>(proj_pt1[1]) > .5)
////  {
////    y++;
////    color[2] = 0;
////  }
//
//  // Check why some points go outside the image
//  if( x < 0 || y < 0 || x >= cam_model_.imgWidth() || y >= cam_model_.imgHeight() )
//  {
//
//  }
//  else
//    color_buf_.at<cv::Vec3b>(y,x) = color;
//  // TODO End debug code!!


  // Unnormalized direction vector
  Point3f dir = p1 - p0;
  float len = std::sqrt ( dir.dot ( dir ) );
  if ( !len )
    return;

  int init_i = vis_pts_.size();
  bool single_point = false;
  // sub_segs is used to collect unoccluded sub-segments along a segment
  vector< Point3f > sub_segs;

  int n_steps = len/step_;
  if ( !n_steps )
  {
    // Try to push at least one point
    if ( checkPointOcclusion ( p0 ) )
    {
      vis_pts_.push_back ( p0 );
      vis_d_pts_.push_back ( p0 + epsilon_*dir );
      single_point = true;
    }
  }
  else
  {
    float dist_step = 1.0/n_steps;
    bool start_point = true;
    for ( int i = 0; i <= n_steps; i++ )
    {
      Point3f p = p0 + ( i*dist_step ) * dir;
      if ( checkPointOcclusion ( p ) )
      {
        // First point of an unoccluded sub-segment
        if ( start_point )
        {
          sub_segs.push_back ( p );
          start_point = false;
        }
        vis_pts_.push_back ( p );
        vis_d_pts_.push_back ( p+ epsilon_*dir );
      }
      else
      {
        // Last point of an unoccluded sub-segment
        if ( !start_point )
        {
          Point3f end_p = p0 + ( ( i-1 ) *dist_step ) *dir;
          sub_segs.push_back ( end_p );
          start_point = true;
        }
      }
    }
    // In case, "close" the last subsegment
    if ( !start_point )
      sub_segs.push_back ( p1 );
  }

  if ( single_point )
  {
    if ( len >= min_seg_len_ && checkPointOcclusion ( p1 ) )
    {
      vis_segs_.push_back ( Vec6f ( p0.x, p0.y, p0.z, p1.x, p1.y, p1.z ) );
      vis_d_segs_.push_back ( p0 + epsilon_*dir );
    }
  }
  else if ( init_i < int(vis_pts_.size()) )
  {
    // Iterate for each subsegment
    for ( int i = 0; i < int(sub_segs.size()); i += 2 )
    {
      Point3f s_seg = sub_segs[i], e_seg = sub_segs[i+1];
      Point3f diff = s_seg - e_seg;
      len = std::sqrt ( diff.dot ( diff ) );
      if ( len >= min_seg_len_ )
      {
        vis_segs_.push_back ( Vec6f ( s_seg.x, s_seg.y, s_seg.z, e_seg.x, e_seg.y, e_seg.z ) );
        vis_d_segs_.push_back ( s_seg + epsilon_*dir );
      }
    }
  }
}

inline float RasterObjectModel3D::glDepthBuf2Depth(float d)
{
  return depth_transf_a_ / ( depth_transf_b_ - ( 2.0f * d - 1.0f ) * depth_transf_c_ );
}

Mat RasterObjectModel3D::getRenderedModel( const Mat &background_img  )
{
  cv_ext_assert( background_img.empty() ||
                 ( background_img.type() == DataType<Vec3b>::type &&
                   background_img.cols == cam_model_.imgWidth() &&
                   background_img.rows == cam_model_.imgHeight() ) );

  if( !raster_updated_ )
    update();

  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
  Mat render_img( Size(cols, rows) , DataType<Vec3b>::type );
  if(background_img.empty())
    render_img.setTo(Scalar(0,0,0));
  else
    render_img = background_img.clone();

  for( int r = 0; r < rows; r++ )
  {
    const float *depth_p = depth_buf_.ptr<float>(r);
    const Vec3b *color_p = color_buf_.ptr<Vec3b>(r);
//    const uchar *ppp = g_contour_img.ptr<uchar>(r);
    Vec3b *rend_p = render_img.ptr<Vec3b>(r);

    for( int c = 0; c < cols; c++, depth_p++, color_p++, rend_p++/*, ppp++*/ )
    {
      if( *depth_p < 1.0f /*&& *ppp*/ )
        *rend_p = *color_p;
    }
  }
  return render_img;
}

Mat RasterObjectModel3D::getRenderedModel( const Scalar &background_color )
{
  if( !raster_updated_ )
    update();

  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
  Mat render_img( Size(cols, rows) , DataType<Vec3b>::type,
                  Scalar(background_color[2],background_color[1],background_color[0]) );

  for( int r = 0; r < rows; r++ )
  {
    const float *depth_p = depth_buf_.ptr<float>(r);
    const Vec3b *color_p = color_buf_.ptr<Vec3b>(r);
//    const uchar *ppp = g_contour_img.ptr<uchar>(r);
    Vec3b *rend_p = render_img.ptr<Vec3b>(r);

    for( int c = 0; c < cols; c++, depth_p++, color_p++, rend_p++/*, ppp++*/ )
    {
      if( *depth_p < 1.0f /*&& *ppp*/ )
        *rend_p = *color_p;
    }
  }
  return render_img;
}

Mat RasterObjectModel3D::getModelDepthMap()
{
  if( !raster_updated_ )
    update();

  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
  Mat real_depth_img( rows, cols, DataType<float>::type);

  for( int r = 0; r < rows; r++ )
  {
    const float *depth_p = depth_buf_.ptr<float>(r);
    float *real_depth_p = real_depth_img.ptr<float>(r);

    for( int c = 0; c < cols; c++, depth_p++, real_depth_p++ )
    {
      float d = *depth_p;
      if( d < 1.0f )
        *real_depth_p = glDepthBuf2Depth( d );
      else
        *real_depth_p = -1;
    }
  }
  return real_depth_img;
}

Mat RasterObjectModel3D::getMask()
{
  if( !raster_updated_ )
    update();

  int rows = cam_model_.imgHeight(), cols = cam_model_.imgWidth();
  Mat mask( Size(cols, rows) , DataType<uchar>::type, Scalar(0));



  for( int r = 0; r < rows; r++ )
  {
    const float *depth_p = depth_buf_.ptr<float>(r);
    uchar *mask_p = mask.ptr<uchar>(r);

    for( int c = 0; c < cols; c++, depth_p++, mask_p++ )
    {
      float d = *depth_p;
      if( d < 1.0f )
        *mask_p = 255;
    }
  }
  return mask;
}

// TODO Optimize this!
bool RasterObjectModel3D::checkPointOcclusion ( Point3f& p )
{
  if( custom_bbox_ )
  {
    if( !bbox_.contains(p))
      return false;
  }

  float depth;
  float proj_pt[2];

  cam_model_.rTProject( view_r_mat_, view_t_vec_, reinterpret_cast<float *>(&p), proj_pt, depth );

  int x = cvRound(proj_pt[0]), y = cvRound(proj_pt[1]);

  // TODO Check why some points go outside the image
  if( x < 1 || y < 1 || x >= cam_model_.imgWidth() - 1 || y >= cam_model_.imgHeight() - 1 )
    return false;

  // Already added point
  if( model_mask_.at<uchar>(y,x) )
    return false;

//  std::cout<<proj_pt[0]<<" "<<proj_pt[1]<<std::endl;

  float closest_depth0 = glDepthBuf2Depth( depth_buf_.at<float>(y,x) );
  float closest_depth1 = glDepthBuf2Depth( depth_buf_.at<float>(y + 1,x) );
  float closest_depth2 = glDepthBuf2Depth( depth_buf_.at<float>(y - 1,x) );
  float closest_depth3 = glDepthBuf2Depth( depth_buf_.at<float>(y,x + 1) );
  float closest_depth4 = glDepthBuf2Depth( depth_buf_.at<float>(y,x - 1) );

  if ( depth <= closest_depth0 + depth_buffer_epsilon_ ||
       depth <= closest_depth1 + depth_buffer_epsilon_ ||
       depth <= closest_depth2 + depth_buffer_epsilon_ ||
       depth <= closest_depth3 + depth_buffer_epsilon_ ||
       depth <= closest_depth4 + depth_buffer_epsilon_ )
  {
    // color_debug_img.at<Vec3b>(proj_p.y, proj_p.x) = Vec3b(0,0,255);
    model_mask_.at<uchar>(y,x) = 255;
    mask_pts.emplace_back(x,y);
    return true;
  }
  else
    return false;
}
