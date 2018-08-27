dynamic_watershed
=================

Package description
--------------

We implement the splitting algorithm for splitting nuclei nucleas described in in 'Nuclei segmentation in histopathology images using deep neural networks'. This algorithm is essentially a dynamic watershed.
The main function is named: `post_process`.


Installation
--------------

dynamic_watershed can be installed by unzipping the source code in one directory and using this command: ::

    python setup.py install

You can also install it directly from the Python Package Index with this command (not working yet): :: 

    pip install dynamic_watershed

Example
--------------
```python
>>> from dynamic_watershed import post_process
>>> from skimage.io import imread
>>> probability_image = imread('example.png')
>>> p1, p2 = 7, 0.5
>>> result_segmentation = post_process(probability_image, p1, thresh=p2)
```

Licence
--------

See file LICENCE.txt in this folder.


Contribute
-----------
dynamic_watershed is an open-source software. Everyone is welcome to contribute !


Cite
-----------

If you use this work please cite our paper.

BibTeX: 
```
  @inproceedings{naylor2017nuclei,
    title={Nuclei segmentation in histopathology images using deep neural networks},
    author={Naylor, Peter and La{\'e}, Marick and Reyal, Fabien and Walter, Thomas},
    booktitle={Biomedical Imaging (ISBI 2017), 2017 IEEE 14th International Symposium on},
    pages={933--936},
    year={2017},
    organization={IEEE}
    }
```

