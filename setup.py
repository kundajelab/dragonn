from setuptools import setup,find_packages

config = {
    'include_package_data': True,
    'description': 'Deep RegulAtory GenOmic Neural Networks (DragoNN)',
    'version': '0.4.1',
    'packages': find_packages(),
    'setup_requires': [],
    'install_requires': ['deeplift>=0.6.9.0', 'shapely', 'matplotlib', 'plotnine','scikit-learn>=0.20.0', 'pydot_ng==1.0.0', 'h5py','seqdataloader>=0.124','simdna_dragonn','abstention', 'logomaker'],
    'dependency_links': [],
    'scripts': [],
    'name': 'dragonn'
}

if __name__== '__main__':
    setup(**config)
