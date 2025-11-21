import numpy as np

def generate_data(city_name):

    population = int(np.random.lognormal(mean=10, sigma=1.5))
    area = np.random.uniform(50, 500) 
    gdp_per_capita = np.random.uniform(20000, 80000) 
    education_score = np.random.uniform(0.1, 1.0)
    infrastructure_score = np.random.uniform(0.1, 1.0)
    location_score = np.random.uniform(0.1, 1.0)
    x = np.random.uniform(0, 1000)
    y = np.random.uniform(0, 1000)


    return {"city_name":city_name, "population":population, "area":area, "gdp_per_capita":gdp_per_capita, "education_score":education_score, "infrastructure_score":infrastructure_score, "location_score":location_score, "x":x, "y":y}
