
from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'This module enables users to perform molecular dynamics simulations utilizing GROMACS. \
               It is build to enhance the setup and analysis of several simulations, including free energy simulations. \
               Mdp and job files are generated via jinja2 templates and provided yaml parameter files.'

# Setting up
setup(
    name="pyGROMACS",
    version=VERSION,
    description=DESCRIPTION,
    url='https://github.com/samirdarouich/pyGROMACS.git',
    author="Samir Darouich",
    author_email="samir.darouich@itt.uni-stuttgart.de",
    license_files = ('LICENSE'),
    packages=find_packages(),
    install_requires=['numpy',
                      'jinja2',
                      'scipy',
                      'pandas',
                      'alchemlyb',
                      'PyYAML'
                      ],

    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Users",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)