import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages

exec(open('dynamic_watershed/version.py').read()) # loads __version__
description = """ Post-processing function used in 'Segmentation of 
                  Nuclei in Histopathology Images by deep regression 
                  of the distance map'. """ 


setup(name='dynamic_watershed',
      version=__version__,
      author='Peter Jack Naylor',
      author_email="peter.jack.naylor@gmail.com",
      description=description,
      long_description=open('README.md').read(),
      long_description_content_type="text/markdown",
      url="https://github.com/PeterJackNaylor/dynamic_watershed.git",
      license='see LICENSE.txt',
      keywords="",
      packages=[],
      classifiers=(
          "Programming Language :: Python",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ),
)
