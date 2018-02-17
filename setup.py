from setuptools import setup

config = {
    'include_package_data': True,
    'description': 'Deep RegulAtory GenOmic Neural Networks (DragoNN)',
    'download_url': 'https://github.com/kundajelab/dragonn',
    'version': '0.1.3',
    'packages': ['dragonn'],
    'setup_requires': [],
    'install_requires': ['numpy>=1.9','tensorflow-gpu==1.0.1', 'keras==1.2.2', 'deeplift==0.4.0-tensorflow', 'shapely', 'simdna==0.3', 'matplotlib<=1.5.3',
                         'scikit-learn', 'pydot_ng==1.0.0', 'h5py'],
    'dependency_links': ["https://github.com/kundajelab/deeplift/tarball/v0.4.0-tensorflow#egg=deeplift-0.4.0-tensorflow",
                         "https://github.com/kundajelab/simdna/tarball/0.3#egg=simdna-0.3"],
    'scripts': [],
    'entry_points': {'console_scripts': ['dragonn = dragonn.__main__:main']},
    'name': 'dragonn'
}

if __name__== '__main__':
    setup(**config)
