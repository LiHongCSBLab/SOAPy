.. highlight:: shell

============
Installation
============


Install by PyPi
---------------

**Step 1:**

Prepare conda environment for SOAPy:
::

	conda create -n SOAPy_st python=3.9
	conda activate SOAPy_st

**Step 2:**

Install SOAPy using `pip`:
::

	pip install SOAPy_st


Install by github
-----------------

Download the file from github:
::

    cd SOAPy_st
    python setup.py build
    python setup.py install


Requirements of SOAPy
-----------------

Those will be installed automatically when using pip.

::

    anndata==0.9.1
    ctxcore==0.2.0
    esda==2.4.3
    geopandas==0.14.3
    libpysal==4.8.1
    networkx==2.8.6
    numba==0.60.0
    opencv-python==4.8.1.78
    pyscenic==0.12.1
    s-dbw==0.4.0
    shapely==2.0.3
    scanpy==1.10.3
    scikit-image==0.19.3
    scikit-learn==1.1.2
    scikit-misc==0.3.1
    scipy==1.13.1
    seaborn==0.13.2
    statsmodels==0.13.2
    tensorly==0.8.1
    torch==1.12.0
    torch-cluster==1.6.3
    torch_geometric==1.6.1
    torch-scatter==2.1.2
    torch-sparse==0.6.18
    torch-spline-conv==1.2.2
    torchaudio==0.12.0
    torchvision==0.13.0
    matplotlib
    numpy
    pandas
    tqdm


Install rpy2 for spatial domain (optional)
-----------------

install r-base by conda:
::

    conda install -c conda-forge r-base

In the R console, run the following command to install the mclust package:
::

    install.packages("mclust")

Exit the r console after installation, install rpy2 by pip:
::

    pip install rpy2

