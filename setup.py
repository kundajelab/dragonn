from setuptools import setup,find_packages

config = {
    'include_package_data': True,
    'description': 'Deep RegulAtory GenOmic Neural Networks (DragoNN)',
    'version': '0.2.2',
    'packages': ['dragonn'],
    'setup_requires': [],
    'install_requires': ['numpy>=1.15', 'keras>=2.2.0','tensorflow>=1.6','deeplift>=0.6.8.1', 'shapely', 'matplotlib',
                         'scikit-learn', 'pydot_ng==1.0.0', 'h5py','concise','seqdataloader>=0.117','simdna>=0.3'],
    'extras_requires':{'tensorflow with gpu':['tensorflow-gpu>=1.7']},
    'dependency_links': [],
    'scripts': [],
    'entry_points': {'console_scripts': ['dragonn = dragonn.__main__:main']},
    'name': 'dragonn'
}

if __name__== '__main__':
    setup(**config)
