# ChemSchematicResolver
**ChemSchematicResolver** is a toolkit for the automatic resolution of chemical schematic diagrams and their labels. You can find out how it works on the [website](http://www.chemschematicresolver.org) , and try out the online demo [here](http://www.chemschematicresolver.org/demo)

## Features

- Extraction of generic R-Group structures
- Automatic detection and download of schematic chemical diagrams from scientific articles
- HTML and XML document format support from RSC and Elsevier
- High-throughput capabilities
- Direct extraction from image files
- PNG, GIF, JPEG, TIFF image format support

## Installation

Installation of ChemSchematicResolver is achieved using [conda](https://docs.conda.io/en/latest).

First, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), which contains a complete Python distribution alongside the conda package manager.

Next, go to the command line terminal and create a working environment by typing

    conda create --name <my_env> python=3.6
    
Once this is created, enter this environment with the command

    conda activate <my_env>
    
There are two ways to continue the installation - via the anaconda cloud, and from source. 

### Option 1 - Installation via anaconda

*Please note that the following option will not work until release.*

We recommend the installation of ChemSchematicResolver through the anaconda cloud.

Simply type:

    conda install -c edbeard chemschematicresolver
    
This command installs ChemSchematicResolver and all it's dependencies from the author's channel.
This includes [pyosra](https://github.com/edbeard/pyosra), the Python wrapper for the OSRA toolkit, and [ChemDataExtractor-CSR](https://github.com/edbeard/chemdataextractor-csr), the bespoke version of ChemDataExtractor containing diagram parsers.

*This method of installation is currently supported on linux machines only*

### Option 2 - Installation from source

*Please note that all following links will not work until release.*

We strongly recommend installation via the conda cloud whenever possible, as all the dependencies are automatically handled.
 
If this cannot be done, users are invited to compile the code from source. This is easiest to do through [conda build](https://docs.conda.io/projects/conda-build/en/latest/), by building and installing using the recipes [here](https://github.com/edbeard/conda_recipes). 

The following packages will need to be built from a recipe, in the order below:

1. **Pyosra**: [[recipe](https://github.com/edbeard/conda_recipes/tree/master/pyosra), [source code](https://github.com/edbeard/pyosra)]

2. **ChemDataExtracor-CSR**: [[recipe](https://github.com/edbeard/conda_recipes/tree/master/cde-csr/recipes/chemdataextractor), [source code](https://github.com/edbeard/chemdataextractor-csr)]

3. **ChemSchematicResolver**: [[recipe](https://github.com/edbeard/conda_recipes/tree/master/csr), [source code](https://github.com/edbeard/ChemSchematicResolver)]

For each, enter the directory and run:

    conda build .
    
to create a compressed tarball file, which contains the instructions for installing the code *(Please note that this can take up to 30 minutes to build)*.
 
Move all compressed tarballs to a single directory, enter the directory and run:

    conda index .

This changes the directory to a format emulating a conda channel. To install all code and dependencies, then simply run

    conda install -c <path/to/tarballs> chemschematicresolver
    
And you should have everything installed!


# Getting Started

This section gives a introduction on how to get started with ChemSchematicResolver. This assumes you already have
ChemSchematicResolver and all dependencies installed.

## Extract Image
It's simplest to run ChemSchematicResolver on an image file.

Open a python terminal and import the library with: 

    >>> import chemschematicresolver as csr
    
Then run:

    >>> result = csr.extract_image('<path/to/image/file>')
    
to perform the extraction. 

This runs ChemSchematicResolver on the image and returns a list of tuples to `output`. Each tuple consists of a SMILES string and a list of label candidates, where each tuple identifies a unique structure. For example:

    >>> print(result)
    [(['1a'], 'C1CCCCC1'), (['1b'], 'CC1CCCCC1')]

## Extract Document

To automatically extract the structures and labels of diagrams from a HTML or XML article, use the `extract_document` method instead:
 
    >>> result = csr.extract_document('<path/to/document/file>')
    
If the user has permissions to access the full article, this function will download all relevant images locally to a directory called *csr*, and extract from them automatically. The *csr* directory with then be deleted.

The tool currently supports HTML documents from the [Royal Society of Chemistry](https://www.rsc.org/) and [Springer](https://www.springer.com), as well as XML files obtained using the [Elsevier Developers Portal](https://dev.elsevier.com/index.html) .

ChemSchematicResolver will return the complete chemical records from the document extracted with [ChemDataExtractor](http://chemdataextractor.org/), enriched with extracted structure and raw label. For example:

    >>> print(result)
    {'labels': ['1a'], 'roles': ['compound'], 'melting_points': [{'value': '5', 'units': '°C'}], 'diagram': { 'smiles': 'C1CCCCC1', 'label': '1a' } }

Alternatively, if you just want the structures and labels extracted from the images without the ChemDataExtractor output, run:

    >>> result = csr.extract_document('<path/to/document/file>', extract_all=False)
    
which, for the above example, will return:

    >>> print(output)
    [(['1a'], 'C1CCCCC1')]
