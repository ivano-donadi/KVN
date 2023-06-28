#include "pvnet_wrapper.h"

#include "pyhelper.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <map>

class PVNetWrapper::Impl
{
 public:
  Impl( const std::string &python_root ) : py_env(python_root){};

  ~Impl() = default;

  bool init()
  {
    CPyObject py_module = PyImport_ImportModule("tools.inference");

    if(!py_module)
    {
      std::cerr<<"Python module not imported"<<std::endl;
      PyErr_Print();
      return false;
    }

    CPyObject dict = PyModule_GetDict(py_module);
    if(!dict)
    {
      std::cerr<<"Failed to get the Python module dictionary"<<std::endl;
      PyErr_Print();
      return false;
    }

    CPyObject py_class = PyDict_GetItemString(dict, "MultiInference");
    if(!py_class || !PyCallable_Check(py_class))
    {
      std::cerr<<"Failed to get the Python class"<<std::endl;
      PyErr_Print();
      return false;
    }

    py_object = PyObject_CallObject(py_class, nullptr);

    if(!py_object)
    {
      std::cerr<<"Cannot instantiate the Python class"<<std::endl;
      PyErr_Print();
      return false;
    }

    import_array1(false);

    initialized = true;
    return true;
  }
  bool initialized = false;
  CPyObject py_object;
  int num_objects = 0;
  std::map<int, int> id_map;
 private:
  CPyInstance py_env;
};

PVNetWrapper::PVNetWrapper(const std::string &python_root) :
pimpl_(new PVNetWrapper::Impl(python_root)){}

PVNetWrapper::~PVNetWrapper() = default;

bool PVNetWrapper::registerObject(int obj_id, const std::string &model_fn, const std::string &inference_meta_fn)
{
  if(!pimpl_->initialized && !pimpl_->init() )
    return false;

  PyObject_CallMethod(pimpl_->py_object, const_cast<char*>("addModel"), const_cast<char*>("ss"),
                      model_fn.c_str(), inference_meta_fn.c_str());
  pimpl_->id_map[obj_id] = pimpl_->num_objects++;

  return true;
}
bool PVNetWrapper::localize(cv::Mat &img, int obj_id, cv::Mat_<double> &r_mat, cv::Mat_<double> &t_vec)
{
  npy_intp dims[3] = { img.rows, img.cols, 3 };
  uchar *py_array_data = img.ptr<uchar>();
  PyObject* py_array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, (void *)py_array_data);

  int idx = pimpl_->id_map[obj_id];
  CPyObject pvnet_pos = PyObject_CallMethod(pimpl_->py_object, const_cast<char*>("__call__"),
                                            const_cast<char*>("Oi"), py_array, idx);
  Py_DECREF(py_array);

  PyArrayObject *np_ret = reinterpret_cast<PyArrayObject*>(pvnet_pos.getObject());
  double *c_out = reinterpret_cast<double*>(PyArray_DATA(np_ret));
  cv::Mat_<double> r_vec(3,1);
  r_mat.create(3,3);
  t_vec.create(3,1);

  int  i = 0;
  for (int r = 0; r < 3; r++)
  {
    r_mat(r,0) = c_out[i++];
    r_mat(r,1) = c_out[i++];
    r_mat(r,2) = c_out[i++];
    t_vec(r,0) = c_out[i++];
  }
  //    std::cout<<r_mat<<std::endl;
  //    std::cout<<t_vec<<std::endl;
  return true;
}
