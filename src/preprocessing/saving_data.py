import json
from datetime import datetime
import numpy as np
import os
import random
from math import floor

def save_to_json(cities, edges, file_path=None, filename=None):
    

    if file_path is None:
        current_dir = os.path.dirname(__file__) 
        src_dir = os.path.dirname(current_dir)  
        project_root = os.path.dirname(src_dir)  
        file_path = os.path.join(project_root, 'data', 'processed')

    os.makedirs(file_path, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cities_data_{timestamp}.json"
    
    cities_data = []
    for city in cities:
        city_dict = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in city.items()
        }
        cities_data.append(city_dict)

    
    complete_data = {
        'cities': cities_data,
        'edges': edges
    }
    
    # Construct full file path
    full_path = os.path.join(file_path, filename)
    
    with open(full_path, 'w') as f:
        json.dump(complete_data, f, indent=2)
    
    print(f" Cities data saved to: {full_path}")
    return full_path


def split_and_save_data(cities, edges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):


    random.seed(random_seed)
    np.random.seed(random_seed)
    
    num_cities = len(cities)
    

    train_size = floor(num_cities * train_ratio)
    val_size = floor(num_cities * val_ratio)
    test_size = num_cities - train_size - val_size
    

    indices = list(range(num_cities))
    random.shuffle(indices)

    train_indices = set(indices[:train_size])
    val_indices = set(indices[train_size:train_size + val_size])
    test_indices = set(indices[train_size + val_size:])

    train_cities = [cities[i] for i in sorted(train_indices)]
    val_cities = [cities[i] for i in sorted(val_indices)]
    test_cities = [cities[i] for i in sorted(test_indices)]
    

    train_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(train_indices))}
    val_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(val_indices))}
    test_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(test_indices))}
    
    train_edges = []
    val_edges = []
    test_edges = []
    
    for edge in edges:
        city1_idx, city2_idx = edge[0], edge[1]
        
        if city1_idx in train_indices and city2_idx in train_indices:
            new_edge = (train_mapping[city1_idx], train_mapping[city2_idx], edge[2])
            train_edges.append(new_edge)
        elif city1_idx in val_indices and city2_idx in val_indices:
            new_edge = (val_mapping[city1_idx], val_mapping[city2_idx], edge[2])
            val_edges.append(new_edge)
        elif city1_idx in test_indices and city2_idx in test_indices:
            new_edge = (test_mapping[city1_idx], test_mapping[city2_idx], edge[2])
            test_edges.append(new_edge)
    
    current_dir = os.path.dirname(__file__) 
    src_dir = os.path.dirname(current_dir)   
    project_root = os.path.dirname(src_dir) 
    

    file_paths = {}
    
    train_path = os.path.join(project_root, 'data', 'train')
    file_paths['train'] = save_to_json(train_cities, train_edges, train_path, 'train_data.json')
    
    val_path = os.path.join(project_root, 'data', 'validation')
    file_paths['validation'] = save_to_json(val_cities, val_edges, val_path, 'validation_data.json')

    test_path = os.path.join(project_root, 'data', 'test')
    file_paths['test'] = save_to_json(test_cities, test_edges, test_path, 'test_data.json')
    
    print(f"\nData splited into:")
    print(f"   Train: {len(train_cities)} cities, {len(train_edges)} edges")
    print(f"   Validation: {len(val_cities)} cities, {len(val_edges)} edges")
    print(f"   Test: {len(test_cities)} cities, {len(test_edges)} edges")
    
    return file_paths
