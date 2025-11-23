from data_acquisition.generate_data import generate_data, generate_edges
import numpy as np


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

        