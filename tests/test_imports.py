from shapely.wkt import loads as load_wkt

def test_shapely():
    # Example call from Shapely's test_affinity.py
    # Don't want to fully test Shapley, but ensure it is available
    load_wkt('LINESTRING(2.4 4.1, 2.4 3, 3 3)')
