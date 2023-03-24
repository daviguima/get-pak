import setuptools

__version__ = '0.0.1'
__package__ = 'getpak'
short_description = 'Sentinel-2 and 3 raster and vector manipulation and validation tools.'

setuptools.setup(
    name=__package__,
    version=__version__,
    url='https://github.com/daviguima/get-pak',
    packages=setuptools.find_packages(),
    license='MIT',
    author='David Guimaraes',
    author_email='dvdgmf@gmail.com',
    description=short_description,
    entry_points={
        'console_scripts': ['getpak=main:main'],
    },
    install_requires=[
        'scikit_learn',
        'matplotlib',
        'numpy',
        'pandas'
        ]
    )