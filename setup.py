from setuptools import setup

setup(name='spin-optics',
      version='0.1',
      description='Data analysis library used by the Spin Optics group in the University of Utah physics department',
      url='https://github.com/willtalmadge/SpinOptics',
      author='William Talmadge',
      author_email='willtalmadge@gmail.com',
      license='MIT',
      packages=[
          'spin_optics.data_wrangling',
          'spin_optics.fitting',
          'spin_optics.models',
          'spin_optics.plotting',
          'spin_optics.unit_converters'
      ],
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib'
      ],
      zip_safe=False)