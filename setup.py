from setuptools import setup

setup(
    name='mms_msg',
    version='0.0.0',
    packages=['mms_msg'],
    url='',
    license='',
    author='Thilo von Neumann',
    author_email='vonneumann@nt.upb.de',
    description='MMS-MSG: Multipurpose Multi Speaker Mixture Signal Generator',
    install_requires=[
        'paderbox @ git+http://github.com/fgnt/paderbox',
        'padertorch @ git+http://github.com/fgnt/padertorch',
        'lazy_dataset',
        'tqdm',
        'cached_property',
        'numpy',
        'scipy',
    ],
)
