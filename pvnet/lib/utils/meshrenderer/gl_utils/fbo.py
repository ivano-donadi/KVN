# -*- coding: utf-8 -*-
import numpy as np

from OpenGL.GL import *
import ctypes
from lib.utils.meshrenderer.gl_utils.renderbuffer import Renderbuffer, RenderbufferMultisample
from lib.utils.meshrenderer.gl_utils.texture import Texture, TextureMultisample

class Framebuffer(object):

    def __init__(self, attachements):
        self.__id = np.empty(1, dtype=np.uint32)
        glCreateFramebuffers(len(self.__id), self.__id)
        for k in attachements.keys():
            attachement = attachements[k]
            if isinstance(attachement, Renderbuffer) or isinstance(attachement, RenderbufferMultisample):
                glNamedFramebufferRenderbuffer(ctypes.c_uint(self.id), k, GL_RENDERBUFFER, ctypes.c_uint(attachement.id))
            elif isinstance(attachement, Texture) or isinstance(attachement, TextureMultisample):
                glNamedFramebufferTexture(ctypes.c_uint(self.id), k, ctypes.c_uint(attachement.id), ctypes.c_int(0))
            else:
                raise ValueError('Unknown frambuffer attachement class: {0}'.format(attachement))

        if glCheckNamedFramebufferStatus(ctypes.c_uint(self.id), GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Framebuffer not complete.')
        self.__attachements = attachements

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.id)

    def delete(self):
        glDeleteFramebuffers(1, self.id)
        for k in self.__attachements.keys():
            self.__attachements[k].delete()

    @property
    def id(self):
        return self.__id[0]