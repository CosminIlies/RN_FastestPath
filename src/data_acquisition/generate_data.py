import numpy as np

def generate_data(city_name, seed = 1):
    np.random.seed(seed)
    city_coef = np.random.uniform(0.1, 1)

    min_val = min(city_coef - 0.2, city_coef)
    max_val = max(city_coef, city_coef + 0.2)
    

    population_mean = 8.5 + city_coef * 3 
    population = int(np.random.lognormal(mean=population_mean, sigma=1.5))
    # population = int(np.random.uniform(10000, 500000))
    area = np.random.uniform(50, 500) 
    gdp_per_capita = np.random.uniform(20000, 80000) 
    education_score = np.random.uniform(min_val, max_val)
    infrastructure_score = np.random.uniform(min_val, max_val)
    location_score = np.random.uniform(min_val, max_val)
    x = np.random.uniform(0, 1000)
    y = np.random.uniform(0, 1000)


    return {"city_name":city_name, "population":population, "area":area, "gdp_per_capita":gdp_per_capita, "education_score":education_score, "infrastructure_score":infrastructure_score, "location_score":location_score, "x":x, "y":y}


def generate_edges(cities, num_cities):

    edges = []


    for i in range(num_cities):
        # edges.append((i, i, 1))
        for j in range(i + 1, num_cities):
            city1, city2 = cities[i], cities[j]
            

            geo_dist = np.sqrt((city1['x'] - city2['x'])**2 + (city1['y'] - city2['y'])**2)
            econ_similarity = 1 - abs(city1['gdp_per_capita'] - city2['gdp_per_capita']) / 60000

            if geo_dist < 75 or (geo_dist < 25 and econ_similarity > 0.7):
                weight = geo_dist 
                edges.append((i, j, weight))
                # edges.append((j, i))
                
    return edges


def apply_neighborhood_influence(cities, edges, influence_strength=0.3):

    neighbors = {i: [] for i in range(len(cities))}
    
    for edge in edges:
        city1_idx, city2_idx = edge[0], edge[1]
        weight = edge[2] if len(edge) > 2 else 1.0
        neighbors[city1_idx].append((city2_idx, weight))
        neighbors[city2_idx].append((city1_idx, weight))
    

    influenceable_features = [
        'gdp_per_capita', 'education_score', 
        'infrastructure_score', 'location_score'
    ]
    

    influenced_cities = []
    for city in cities:
        influenced_cities.append(city.copy())
    

    for city_idx, city in enumerate(influenced_cities):
        city_neighbors = neighbors[city_idx]
        
        if not city_neighbors:
            continue

        for feature in influenceable_features:
            if feature not in city:
                continue
                
            neighbor_influence = 0.0
            total_weight = 0.0
            
            for neighbor_idx, weight in city_neighbors:
                neighbor_city = cities[neighbor_idx] 
                if feature in neighbor_city:
                    influence_weight = 1.0 / (weight + 1.0)
                    neighbor_influence += neighbor_city[feature] * influence_weight
                    total_weight += influence_weight
            
            if total_weight > 0:

                avg_neighbor_value = neighbor_influence / total_weight
                
                original_value = city[feature]
                influenced_value = (
                    original_value * (1 - influence_strength) + 
                    avg_neighbor_value * influence_strength
                )
                
                if feature == 'gdp_per_capita':
                    influenced_value = max(15000, min(100000, influenced_value))
                elif feature in ['education_score', 'infrastructure_score', 'location_score']:
                    influenced_value = max(0.1, min(1.0, influenced_value))
                
                city[feature] = influenced_value
    
    return influenced_cities


def apply_economic_clustering(cities, edges, clustering_factor=0.4):

    influenced_cities = []
    for city in cities:
        influenced_cities.append(city.copy())
    
    neighbors = {i: [] for i in range(len(cities))}
    for edge in edges:
        city1_idx, city2_idx = edge[0], edge[1]
        weight = edge[2] if len(edge) > 2 else 1.0
        neighbors[city1_idx].append((city2_idx, weight))
        neighbors[city2_idx].append((city1_idx, weight))
    
    for city_idx, city in enumerate(influenced_cities):
        city_neighbors = neighbors[city_idx]
        
        if not city_neighbors:
            continue
            
        current_gdp = city['gdp_per_capita']
        similar_neighbors = []
        
        for neighbor_idx, weight in city_neighbors:
            neighbor_gdp = cities[neighbor_idx]['gdp_per_capita']
            gdp_similarity = 1 - abs(current_gdp - neighbor_gdp) / max(current_gdp, neighbor_gdp)
            
            if gdp_similarity > 0.8: 
                distance_weight = 1.0 / (weight + 1.0)
                similar_neighbors.append((neighbor_idx, distance_weight))
        
        if similar_neighbors:

            total_weight = sum(weight for _, weight in similar_neighbors)
            
            for feature in ['education_score', 'infrastructure_score']:
                neighbor_avg = sum(
                    cities[neighbor_idx][feature] * weight 
                    for neighbor_idx, weight in similar_neighbors
                ) / total_weight
                
                original_value = city[feature]
                clustered_value = (
                    original_value * (1 - clustering_factor) + 
                    neighbor_avg * clustering_factor
                )
                
                city[feature] = max(0.1, min(1.0, clustered_value))
    
    return influenced_cities