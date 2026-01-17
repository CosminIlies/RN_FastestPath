from data_acquisition.generate_data import generate_data, generate_edges, apply_neighborhood_influence, apply_economic_clustering
import numpy as np
import json


def preprocessing(nr_of_cities, apply_influence=True):    
    cities_data = []

    for i in range(nr_of_cities):
        city = generate_data(f"city_{i+1}", i)
        cities_data.append(city)

    # Generate initial edges based on original city features
    edges_data = generate_edges(cities_data, nr_of_cities)
    
    # Apply neighborhood influence to make cities more realistic
    if apply_influence:
        print("Applying neighborhood influence to city features...")
        
        # First apply general neighborhood influence
        cities_data = apply_neighborhood_influence(
            cities_data, edges_data, influence_strength=0.5
        )
        
        # Then apply economic clustering effects
        cities_data = apply_economic_clustering(
            cities_data, edges_data, clustering_factor=0.6
        )
        
        # Regenerate edges after influence (cities may have changed)
        edges_data = generate_edges(cities_data, nr_of_cities)
        print(f"Neighborhood influence applied. Final edges: {len(edges_data)}")
    
    # Calculate agglomeration scores after all influences are applied
    for city in cities_data:

        agglomeration = (
            np.log(city["population"]) / 15 * 0.10 +
            (city["gdp_per_capita"] - 20000) / 60000 * 0.3 +
            0.2 * city["education_score"] + 
            0.2 * city["infrastructure_score"] +
            0.2 * city["location_score"] 
        )

        # agglomeration = (
        #     0.33 * city["education_score"] + 
        #     0.33 * city["infrastructure_score"] +
        #     0.33 * city["location_score"] 
        # )
        
        # Clamp agglomeration to [0,1] range to ensure valid values
        city["agglomeration"] = max(0.0, min(1.0, agglomeration))
    
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


        