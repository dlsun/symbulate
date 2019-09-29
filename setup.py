from setuptools import setup, find_packages

setup(
    name="symbulate",
    version="0.5.4",

    description="A symbolic algebra for specifying simulations.",

    url="https://github.com/dlsun/symbulate",

    author="Dennis Sun",
    author_email="dsun09@calpoly.edu",

    license="GPLv3",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
    
    keywords='probability simulation',

    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ]
)
