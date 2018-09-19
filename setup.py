!/usr/bin/python
# coding: utf8
# /*##########################################################################
#
# Copyright (c) 2015-2018 Jerome Kieffer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ###########################################################################*/

__authors__ = ["Jérôme Kieffer"]
__date__ = "23/04/2018"
__license__ = "MIT"


import sys
import os
import platform
import shutil
import logging
import glob
# io import has to be here also to fix a bug on Debian 7 with python2.7
# Without this, the system io module is not loaded from numpy.distutils.
# The silx.io module seems to be loaded instead.
import io

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("timelapse.setup")


from distutils.command.clean import clean as Clean
from distutils.command.build import build as _build
try:
    from setuptools import Command
    from setuptools.command.build_py import build_py as _build_py
    from setuptools.command.build_ext import build_ext
    from setuptools.command.sdist import sdist
    logger.info("Use setuptools")
except ImportError:
    try:
        from numpy.distutils.core import Command
    except ImportError:
        from distutils.core import Command
    from distutils.command.build_py import build_py as _build_py
    from distutils.command.build_ext import build_ext
    from distutils.command.sdist import sdist
    logger.info("Use distutils")

try:
    import sphinx
    import sphinx.util.console
    sphinx.util.console.color_terminal = lambda: False
    from sphinx.setup_command import BuildDoc
except ImportError:
    sphinx = None


PROJECT = "timelapse"

if "LANG" not in os.environ and sys.platform == "darwin" and sys.version_info[0] > 2:
    print("""WARNING: the LANG environment variable is not defined,
an utf-8 LANG is mandatory to use setup.py, you may face unexpected UnicodeError.
export LANG=en_US.utf-8
export LC_ALL=en_US.utf-8
""")


def get_version():
    """Returns current version number from version.py file"""
    import version
    return version.strictversion


def get_readme():
    """Returns content of README.rst file"""
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(dirname, "README.rst")
    with io.open(filename, "r", encoding="utf-8") as fp:
        long_description = fp.read()
    return long_description


classifiers = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Environment :: MacOS X",
               "Environment :: Win32 (MS Windows)",
               "Environment :: X11 Applications :: Qt",
               "Intended Audience :: Education",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Natural Language :: English",
               "Operating System :: MacOS",
               "Operating System :: Microsoft :: Windows",
               "Operating System :: POSIX",
               "Programming Language :: Cython",
               "Programming Language :: Python :: 2.7",
               "Programming Language :: Python :: 3.4",
               "Programming Language :: Python :: 3.5",
               "Programming Language :: Python :: 3.6",
               "Programming Language :: Python :: 3.7",
               "Programming Language :: Python :: Implementation :: CPython",
               "Topic :: Scientific/Engineering :: Physics",
               "Topic :: Software Development :: Libraries :: Python Modules",
               ]


# ########## #
# version.py #
# ########## #

class build_py(_build_py):
    """
    Enhanced build_py which copies version.py to <PROJECT>._version.py
    """
    def find_package_modules(self, package, package_dir):
        modules = _build_py.find_package_modules(self, package, package_dir)
        if package == PROJECT:
            modules.append((PROJECT, '_version', 'version.py'))
        return modules

# ##### #
# setup #
# ##### #

def get_project_configuration(dry_run):
    """Returns project arguments for setup"""
    # Use installed numpy version as minimal required version
    # This is useful for wheels to advertise the numpy version they were built with
    if dry_run:
        numpy_requested_version = ""
    else:
        from numpy.version import version as numpy_version
        numpy_requested_version = ">=%s" % numpy_version
        logger.info("Install requires: numpy %s", numpy_requested_version)

    install_requires = [
        # for most of the computation
        "numpy%s" % numpy_requested_version,
        # for the script launcher
        "setuptools"]

    setup_requires = ["setuptools", "numpy"]

    # extras requirements: target 'full' to install all dependencies at once
    full_requires = [
        # io
        'h5py',
        'fabio>=0.7',
        # opencl
        'pyopencl',
        'Mako',
        # gui
        'qtconsole',
        'matplotlib>=1.2.0',
        'PyOpenGL',
        'python-dateutil',
        'PyQt5',
        # extra
        'scipy',
        'Pillow']

    extras_require = {
        'full': full_requires,
    }

    package_data = {
        # Resources files for silx
        'silx.resources': [
            'gui/logo/*.png',
            'gui/logo/*.svg',
            'gui/icons/*.png',
            'gui/icons/*.svg',
            'gui/icons/*.mng',
            'gui/icons/*.gif',
            'gui/icons/*/*.png',
            'opencl/*.cl',
            'opencl/image/*.cl',
            'opencl/sift/*.cl',
            'opencl/codec/*.cl',
            'gui/colormaps/*.npy'],
        'silx.examples': ['*.png'],
    }

    entry_points = {
        'console_scripts': ['silx = silx.__main__:main'],
        # 'gui_scripts': [],
    }

    cmdclass = dict(
        build=Build,
        build_py=build_py,
        test=PyTest,
        build_doc=BuildDocCommand,
        test_doc=TestDocCommand,
        build_ext=BuildExt,
        build_man=BuildMan,
        clean=CleanCommand,
        sdist=SourceDistWithCython,
        debian_src=sdist_debian)

    if dry_run:
        # DRY_RUN implies actions which do not require NumPy
        #
        # And they are required to succeed without Numpy for example when
        # pip is used to install silx when Numpy is not yet present in
        # the system.
        setup_kwargs = {}
    else:
        config = configuration()
        setup_kwargs = config.todict()

    setup_kwargs.update(name=PROJECT,
                        version=get_version(),
                        url="http://github.com/kif/timelapse",
                        author="data analysis unit",
                        author_email="timelapse@terre-adelie.org",
                        classifiers=classifiers,
                        description="Software library for smoothing timelapse video",
                        long_description=get_readme(),
                        install_requires=install_requires,
                        setup_requires=setup_requires,
                        extras_require=extras_require,
                        cmdclass=cmdclass,
                        package_data=package_data,
                        zip_safe=False,
                        entry_points=entry_points,
                        )
    return setup_kwargs


def setup_package():
    """Run setup(**kwargs)

    Depending on the command, it either runs the complete setup which depends on numpy,
    or a *dry run* setup with no dependency on numpy.
    """

    # Check if action requires build/install
    dry_run = len(sys.argv) == 1 or (len(sys.argv) >= 2 and (
        '--help' in sys.argv[1:] or
        sys.argv[1] in ('--help-commands', 'egg_info', '--version',
                        'clean', '--name')))

    if dry_run:
        # DRY_RUN implies actions which do not require dependencies, like NumPy
        try:
            from setuptools import setup
            logger.info("Use setuptools.setup")
        except ImportError:
            from distutils.core import setup
            logger.info("Use distutils.core.setup")
    else:
        try:
            from setuptools import setup
        except ImportError:
            from numpy.distutils.core import setup
            logger.info("Use numpy.distutils.setup")

    setup_kwargs = get_project_configuration(dry_run)
    setup(**setup_kwargs)


if __name__ == "__main__":
    setup_package()
