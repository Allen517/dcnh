from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md')) as f:
    long_description = f.read()


setup(
    name='learning sparse network embedding for identity user retrieval',
    version='1.0.0.dev',
    url='https://github.com/Allen517/sne',
    description='',
    long_description=long_description,
    keywords='',
    author='',
    maintainer='King Wang',
    maintainer_email='wangyongqing.casia@gmail.com',
    license='BSD',
    packages=find_packages(exclude=('tests', 'tests.*')),
    package_data={
    },
    entry_points={
    },
    classifiers=[
        'Framework :: SNE',
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Web Environment',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: Chinese (Simplified)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Softwamax_shinglere Development :: Libraries :: Application Frameworks',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'keras=2.1.1',
        'numpy',
    ],
)
