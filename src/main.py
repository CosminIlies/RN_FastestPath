from preprocessing.preprocessing import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.saving_data import save_to_json, split_and_save_data


def statistics(cities, edges):
    
    city_names = [city['city_name'] for city in cities]
    x_coords = [city['x'] for city in cities]
    y_coords = [city['y'] for city in cities]
    agglomerations = [city['agglomeration'] for city in cities]
    
    # Chart 1: Plot cities
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    scatter = ax1.scatter(x_coords, y_coords, c=agglomerations, s=100, alpha=0.6, cmap='viridis')
    for edge in edges:
        city1_idx = edge[0]
        city2_idx = edge[1]
        ax1.plot([x_coords[city1_idx], x_coords[city2_idx]], 
                [y_coords[city1_idx], y_coords[city2_idx]], 
                'gray', alpha=0.4, linewidth=0.5)
        
    plt.colorbar(scatter, ax=ax1, label='Agglomeration')
    for i, name in enumerate(city_names):
        ax1.annotate(name, (x_coords[i], y_coords[i]), fontsize=8)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('City Locations')
    ax1.grid(True, alpha=0.3)
    

    fig2, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2, figsize=(14, 18))
    
    # Chart 2: Population vs Agglomeration
    populations = [city['population'] for city in cities]
    ax2.scatter(agglomerations, populations, s=100, alpha=0.6, color='coral')
    ax2.set_xlabel('Agglomeration')
    ax2.set_ylabel('Population')
    ax2.set_title('Population vs Agglomeration')
    ax2.grid(True, alpha=0.3)

    # Chart 3: GDP per Capita vs Agglomeration
    gdp_per_capita = [city['gdp_per_capita'] for city in cities]
    ax3.scatter(agglomerations, gdp_per_capita, s=100, alpha=0.6, color='green')
    ax3.set_xlabel('Agglomeration')
    ax3.set_ylabel('GDP per Capita')
    ax3.set_title('GDP per Capita vs Agglomeration')
    ax3.grid(True, alpha=0.3)

    # Chart 4: Education vs Agglomeration
    education = [city['education_score'] for city in cities]
    ax4.scatter(agglomerations, education, s=100, alpha=0.6, color='blue')
    ax4.set_xlabel('Agglomeration')
    ax4.set_ylabel('Education')
    ax4.set_title('Education vs Agglomeration')
    ax4.grid(True, alpha=0.3)

    # Chart 5: Infrastructure Score vs Agglomeration
    infrastructure = [city['infrastructure_score'] for city in cities]
    ax5.scatter(agglomerations, infrastructure, s=100, alpha=0.6, color='purple')
    ax5.set_xlabel('Agglomeration')
    ax5.set_ylabel('Infrastructure Score')
    ax5.set_title('Infrastructure Score vs Agglomeration')
    ax5.grid(True, alpha=0.3)

    # Chart 6: Location Score vs Agglomeration
    fig3, ax6 = plt.subplots(figsize=(10, 8))
    location = [city['location_score'] for city in cities]
    ax6.scatter(agglomerations, location, s=100, alpha=0.6, color='orange')
    ax6.set_xlabel('Agglomeration')
    ax6.set_ylabel('Location Score')
    ax6.set_title('Location Score vs Agglomeration')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    nr_of_cities = 30

    cities, edges = preprocessing(nr_of_cities)
    

    save_to_json(cities, edges, file_path='data/processed', filename='processed.json')
    
    file_paths = split_and_save_data(cities, edges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    statistics(cities, edges)

