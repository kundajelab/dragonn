from setuptools import setup

config = {
    'include_package_data': True,
    'description': 'Deep RegulAtory GenOmic Neural Networks (DragoNN)',
    'download_url': 'https://github.com/kundajelab/dragonn',
    'version': '0.1.1',
    'packages': ['dragonn'],
    'setup_requires': [],
    'install_requires': ['numpy>=1.9', 'keras==0.3.2', 'deeplift', 'shapely', 'simdna', 'matplotlib',
                         'sklearn', 'pyprg', 'pydot_ng'],
    'dependency_links': ["https://github.com/kundajelab/deeplift/tarball/v0.2-alpha#egg=deeplift-0.2",
                         "https://github.com/kundajelab/simdna/tarball/0.1#egg=simdna-0.1"],
    'scripts': [],
    'entry_points': {'console_scripts': ['dragonn = dragonn.__main__:main']},
    'name': 'dragonn'
}

if __name__== '__main__':
    setup(**config)
