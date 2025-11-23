from distutils.core import setup
from Cython.Build import cythonize
import numpy as np
setup(name='Fast oligo tools',ext_modules=cythonize(r"U:\Lab\MERFISH_and_MERSCOPE\Human_Basal_ganglia_Gene_Probe_design\package_and_utils\LibraryDesign3\C_Tools/seqint.pyx"), include_dirs=[np.get_include()])
#setup(name='Fast oligo tools',ext_modules=cythonize(r"C_Tools\seqint.pyx"), include_dirs=[np.get_include()])
