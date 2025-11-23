import numpy as np

def generate_data(city_name, seed = 1):
    np.random.seed(seed)
    
    population = int(np.random.lognormal(mean=10, sigma=1.5))
    area = np.random.uniform(50, 500) 
    gdp_per_capita = np.random.uniform(20000, 80000) 
    education_score = np.random.uniform(0.1, 1.0)
    infrastructure_score = np.random.uniform(0.1, 1.0)
    location_score = np.random.uniform(0.1, 1.0)
    x = np.random.uniform(0, 1000)
    y = np.random.uniform(0, 1000)


    return {"city_name":city_name, "population":population, "area":area, "gdp_per_capita":gdp_per_capita, "education_score":education_score, "infrastructure_score":infrastructure_score, "location_score":location_score, "x":x, "y":y}


def generate_edges(cities, num_cities):

    edges = []

    for i in range(num_cities):
        for j in range(i + 1, num_cities):
            city1, city2 = cities[i], cities[j]
            

            geo_dist = np.sqrt((city1['x'] - city2['x'])**2 + (city1['y'] - city2['y'])**2)
        

            if geo_dist < 300:
                weight = geo_dist 
                edges.append((i, j, weight))
                # edges.append((j, i))
                
    return edges