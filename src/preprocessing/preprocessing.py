from data_acquisition.generate_data import generate_data, generate_edges
import numpy as np
import json


def preprocessing(nr_of_cities):    
    cities_data = []

    for i in range(nr_of_cities):

        city = generate_data(f"city_{i+1}", i)
        
        city["agglomeration"] = (
            np.log(city["population"]) / 15 * 0.50 +
            (city["gdp_per_capita"] - 20000) / 60000 * 0.2 +
            0.1 * city["education_score"] + 
            0.1 * city["infrastructure_score"] +
            0.1 *city["location_score"] 
        )

        cities_data.append(city)

    edges_data = generate_edges(cities_data, nr_of_cities)
    
    return cities_data, edges_data


def preprocessing_read_from_json(file_path):
    # file_path = "data/aigenerated.json"
    cities_data = []

    edges_data = []

    with open(file_path, 'r') as file:
        data = json.load(file)

        for city in data['cities']:
            cities_data.append(city)
   

        for edge in data['edges']:
            edges_data.append(edge)
   
    
    print(cities_data, edges_data)
    return cities_data, edges_data


        