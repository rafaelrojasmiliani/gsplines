try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='gsplines',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy >= 1.17.2',
        'sympy >= 1.4',
        'scipy >= 1.3.1',
    ],
    packages=[
        'gsplines',
    ],
)
