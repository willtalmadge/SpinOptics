from setuptools import setup

setup(name='spin-optics',
      version='0.1',
      description='Data analysis library used by the Spin Optics group in the University of Utah physics department',
      url='https://github.com/willtalmadge/SpinOptics',
      author='William Talmadge',
      author_email='willtalmadge@gmail.com',
      license='MIT',
      packages=[
          'data_wrangling', 'fitting', 'models', 'plotting', 'unit_converters'
      ],
      install_requires=[
          'numpy',
          'scipy',
          'pandas'
      ],
      zip_safe=False)