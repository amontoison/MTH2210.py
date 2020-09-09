from setuptools import setup, find_packages

setup(name='MTH2210',
    version='0.1',
    description='Codes Python pour le cours MTH2210 de Polytechnique Montréal.',
    author='Pierre-Yves Bouchet',
    url='https://github.com/amontoison/MTH2210.py',
    packages=find_packages(),#['MTH2210'],
    install_requires=['numpy']
 )
