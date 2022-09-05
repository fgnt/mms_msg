from setuptools import setup
from setuptools import find_packages

setup(
    name='mms_msg',
    version='0.0.0',
    packages=find_packages(exclude=('tests', 'notebooks')),
    url='',
    license='',
    author='Thilo von Neumann',
    author_email='vonneumann@nt.upb.de',
    description='MMS-MSG: Multipurpose Multi Speaker Mixture Signal Generator',
    install_requires=[
        'paderbox @ git+http://github.com/fgnt/paderbox',
        'padertorch @ git+http://github.com/fgnt/padertorch',
        'lazy_dataset @ git+http://github.com/fgnt/lazy_dataset',
        'tqdm',
        'cached_property',
        'numpy',
        'scipy',
        'click',
    ],
)
