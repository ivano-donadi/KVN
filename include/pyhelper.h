#pragma once

/* Taken from:
 *
 * https://www.codeproject.com/Articles/820116/Embedding-Python-program-in-a-C-Cplusplus-code
 */

#include <Python.h>

class CPyInstance
{
 public:
  CPyInstance( const std::string &python_root )
  {
    Py_Initialize();
    PyObject *sysmodule, *syspath;
    sysmodule = PyImport_ImportModule("sys");
    syspath = PyObject_GetAttrString(sysmodule, "path");
    PyList_Append(syspath, PyUnicode_FromString( python_root.c_str() ));
    PyList_Append(syspath, PyUnicode_FromString( "/home/ivano/.local/lib/python3.8/site-packages"));
    Py_DECREF(syspath);
    Py_DECREF(sysmodule);
  }

  ~CPyInstance()
  {
    Py_Finalize();
  }

};


class CPyObject
{
 private:
  PyObject *p;
 public:
  CPyObject() : p(NULL)
  {}

  CPyObject(PyObject* _p) : p(_p)
  {}


  ~CPyObject()
  {
    Release();
  }

  PyObject* getObject()
  {
    return p;
  }

  PyObject* setObject(PyObject* _p)
  {
    return (p=_p);
  }

  PyObject* AddRef()
  {
    if(p)
    {
      Py_INCREF(p);
    }
    return p;
  }

  void Release()
  {
    if(p)
    {
      Py_DECREF(p);
    }

    p= NULL;
  }

  PyObject* operator ->()
  {
    return p;
  }

  bool is()
  {
    return p ? true : false;
  }

  operator PyObject*()
  {
    return p;
  }

  PyObject* operator = (PyObject* pp)
  {
    p = pp;
    return p;
  }

  operator bool()
  {
    return p ? true : false;
  }
};