from silx.opencl.processing import OpenclProcessing
from threading import Semaphore
import os
import numpy
import pyopencl
from pyopencl import array as clarray
from PIL import Image


os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"


class Deflicker(OpenclProcessing):
    """This class is used for deflickering timelapse video frame per frame
    """

    def __init__(self, image_list, ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, memory=None, profile=False):
        """constructor
        
        :param: image_list: list of frames as JPEG files
        """
        self.sem = Semaphore()
        OpenclProcessing.__init__(self, ctx=None, devicetype="all", platformid=None, deviceid=None,
                                  block_size=None, memory=None, profile=False)
        self.image_list = image_list
        self.nimg = len(image_list)
        assert self.nimg
        raw = numpy.asarray(Image.open(image_list[0]))
        shape = raw.shape
        if len(shape) == 3:
            self.ncol = shape[2]
            self.shape = shape[:2]
        else:
            self.ncol = 1
            self.shape = shape
        self.dtype = raw.dtype


    def measure(self, imgage, previous=None):
        """measure the difference on light between an image and the previous one
        :param:
        :param:
        :return:  """


def sigma_clip_cl(data, cutof=3):
    if isinstance(data, gpuarray.Array):
        delta_d = data
    else:
        delta_d.set(data)
    size = data.size
    res, evt = rk2(delta_d, return_event=True)
    mm = res.get()
    first_size = current_size = mm["s1"]
    last_size = current_size + 1
    while last_size > current_size:
        evt2 = mean2std(queue, (size,), None, delta_d.data, res.data, delta2_d.data, numpy.int32(size))
        res2, evt3 = rk2(delta2_d, return_event=True, wait_for=[evt2])
        evt4 = sigmaclip(queue, (size,), None, delta_d.data, res.data, res2.data, numpy.float32(cutof), numpy.int32(size))
        m = mm["s0"] / mm["s1"]
        v = res2.get()
        s = numpy.sqrt(v["s0"] / (v["s1"] - 1.0))
        last_size = current_size
        res, evt = rk2(delta_d, return_event=True)
        mm = res.get()
        current_size = mm["s1"]
        # print(current_size)

    # print(first_size, current_size)
    return m, s


def sigma_clip_np(data, cutof=3):
    "Pure numpy implementation of"
    ldata = data.copy()
    first_size = current_size = numpy.isfinite(ldata).sum()
    last_size = current_size + 1

    while last_size > current_size:
        last_size = current_size
        m = numpy.nanmean(ldata)
        s = numpy.nanstd(ldata)
        ldata[abs(ldata - m) > cutof * s] = numpy.nan
        current_size = numpy.isfinite(ldata).sum()
        print(current_size)
    print(first_size, current_size)
    return m, s
