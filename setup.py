from setuptools import setup

config = {
    'include_package_data': True,
    'description': 'Deep RegulAtory GenOmic Neural Networks (DragoNN)',
    'download_url': 'https://github.com/kundajelab/dragonn',
    'version': '0.1.0',
    'packages': ['dragonn', 'dragonn.synthetic'],
    'package_data': {'dragonn.synthetic': ['motifs.txt.gz']},
    'setup_requires': [],
    'install_requires': ['numpy>=1.9', 'keras==0.3.2', 'deeplift', 'shapely', 'matplotlib',
                         'sklearn', 'pyprg', 'pydot_ng', 'future'],
    'dependency_links': ['https://github.com/kundajelab/deeplift/tarball/master#egg=deeplift-0.2'],
    'scripts': [],
    'entry_points': {'console_scripts': ['dragonn = dragonn.__main__:main']},
    'name': 'dragonn'
}

if __name__== '__main__':
    setup(**config)
