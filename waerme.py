#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import vbo 
import pygame
import sys, time
import numpy
import pyopencl as cl

class Timer:
    def __init__(self):
        self.total_time = 0
        self.n_runs = 0
    
    def start(self):
        self.start_time = time.clock()
    def stop(self):
        self.total_time += time.clock() - self.start_time
        self.n_runs += 1
    
    def average(self):
        return self.total_time/self.n_runs

try:
    from PIL.Image import open as open_image
except ImportError, err:
    from Image import open as open_image

if __name__ == "__main__":
    
    try:
        inital_texture = open_image(sys.argv[1])
    except IndexError:
        print "Usage:",sys.argv[0],"INITAL_TEXTURE"
        sys.exit(-1)
        
    try:
        ix, iy, image_data = inital_texture.size[0], inital_texture.size[1], inital_texture.tostring("raw", "RGBA", 0, -1)
    except SystemError:
        ix, iy, image_data = inital_texture.size[0], inital_texture.size[1], inital_texture.tostring("raw", "RGBX", 0, -1)
    width,height = inital_texture.size
    
    temperature_field_a = numpy.ndarray((width*height,1), dtype=numpy.float32)
    temperature_field_b = numpy.ndarray((width*height,1), dtype=numpy.float32)
    temperature_gradient = numpy.ndarray((width*height,2), dtype=numpy.float32)
    for x in xrange(ix):
        for y in xrange(iy):
            temperature_field_a[x*height + y] = temperature_field_b[x*height + y] = float(ord(image_data[x*(height*4) + y*4]))/255.0*1000.0
    
    pygame.init()
    surface = pygame.display.set_mode(inital_texture.size, pygame.OPENGL|pygame.DOUBLEBUF, 16)
    pygame.display.set_caption("OpenCL Wärmeleitungsgleichung")
    
    glEnable(GL_TEXTURE_2D)
    status_texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, status_texture)
    glPixelStorei(GL_UNPACK_ALIGNMENT,1)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)
    
    pos = numpy.array((
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),)
    )
    geometry_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    texcoord_vbo = vbo.VBO(data=pos, usage=GL_DYNAMIC_DRAW, target=GL_ARRAY_BUFFER)
    
    glClearColor(1.0, 0.0, 0.0, 1.0)
    
    platform = cl.get_platforms()[0]
    
    from pyopencl.tools import get_gl_sharing_context_properties
    import sys
    if sys.platform == "darwin":
        ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                devices=[])
    else:
        # Some OSs prefer clCreateContextFromType, some prefer
        # clCreateContext. Try both.
        try:
            ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, platform)]
                + get_gl_sharing_context_properties())
        except:
            ctx = cl.Context(properties=[
                (cl.context_properties.PLATFORM, platform)]
                + get_gl_sharing_context_properties(),
                devices = [platform.get_devices()[0]])
                
    
    prog = cl.Program(ctx, open("waerme.cl","r").read()).build()
    queue = cl.CommandQueue(ctx)
    
    temperature_fields = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=temperature_field_a), cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=temperature_field_b)
    gradient_field = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=temperature_gradient)
    heat_tex_dev = cl.GLTexture(ctx, cl.mem_flags.WRITE_ONLY, GL_TEXTURE_2D, 0, status_texture, 2)
    queue.finish()
    
    loop_time_list = []
    cl_time_list = []
    
    time_total = Timer()
    time_acquire = Timer()
    time_release = Timer()
    time_gradient = Timer()
    time_heat = Timer()
    time_finish = Timer()
    
    quit = False
    texture_a_is_target = True
    while not quit:
        loop_start = time.clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit = True
        
        time_total.start()
        time_gradient.start()
#        prog.calc_gradient(queue, (width,height), None,  temperature_fields[0], gradient_field, numpy.array((width,height),dtype=numpy.int32))
        time_gradient.stop()
        
        time_acquire.start()
        cl.enqueue_acquire_gl_objects(queue, [heat_tex_dev])
        time_acquire.stop()
        
        time_heat.start()
        prog.solve_heat_equation(queue, (width,height), None,  temperature_fields[0], temperature_fields[1], gradient_field, numpy.array((width,height),dtype=numpy.int32), heat_tex_dev)
        time_heat.stop()
        
        time_release.start()
        cl.enqueue_release_gl_objects(queue, [heat_tex_dev])
        time_release.stop()
        
        time_finish.start()
        queue.finish()
        time_finish.stop()
        time_total.stop()
        temperature_fields = temperature_fields[1], temperature_fields[0]
        
                
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glViewport(0, 0, width, height)
        glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, status_texture)
        
        """
        texcoord_vbo.bind()
        glTexCoordPointer(2, GL_FLOAT, 0, texcoord_vbo)
        geometry_vbo.bind()
        glVertexPointer(2, GL_FLOAT, 0, geometry_vbo)
        
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        #draw the VBOs
        glDrawArrays(GL_TRIANGLES, 0, 6)

        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        """
        
        glBegin(GL_TRIANGLES)
        glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(0.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(0.0, 1.0)
        glEnd()
        
        pygame.display.flip()
        loop_time_list.append(time.clock() - loop_start)
    print "Average loop time [ms]", sum(loop_time_list)/len(loop_time_list)*1e3
    print "Average time_total", time_total.average()*1e3
    print "Average time_acquire", time_acquire.average()*1e3
    print "Average time_release", time_release.average()*1e3
    print "Average time_gradient", time_gradient.average()*1e3
    print "Average time_heat", time_heat.average()*1e3
    print "Average time_finish", time_finish.average()*1e3
