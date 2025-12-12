from preprocessing.preprocessing import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.saving_data import save_to_json, split_and_save_data
from enum import Enum
from neural_network.city_agglomeration_gnn import CityAgglomerationGNN
import torch
import torch.nn.functional as F
from torch.optim import Adam
import json
from torch_geometric.data import Data

class State(Enum):
    """Define the possible states"""
    INIT = "initialization"
    GENERATE_DATA = "generating_data"
    SAVE_DATA = "saving_data"
    VISUALIZE = "visualizing"

    CREATE_MODEL = "creating_model"
    TRAIN_MODEL = "train_model"
    PREDICT = "prediction"

    IDLE = "idle"

    COMPLETE = "complete"
    ERROR = "error"


class DataProcessingStateMachine:
    def __init__(self, nr_of_cities=30):
        self.current_state = State.INIT
        self.nr_of_cities = nr_of_cities
        self.cities = None
        self.edges = None
        self.file_paths = None
        self.error_message = None
        self.model = None
        self.optimizer = None
        
        # Define valid transitions
        self.transitions = {
            State.INIT: [State.GENERATE_DATA, State.ERROR],
            State.GENERATE_DATA: [State.SAVE_DATA, State.ERROR],
            State.SAVE_DATA: [State.CREATE_MODEL, State.IDLE, State.COMPLETE, State.ERROR],

            State.CREATE_MODEL : [State.TRAIN_MODEL, State.ERROR],
            State.TRAIN_MODEL : [State.IDLE, State.PREDICT, State.ERROR],
            State.PREDICT : [State.IDLE, State.ERROR],
            State.IDLE : [State.COMPLETE, State.VISUALIZE, State.ERROR],

            State.VISUALIZE: [State.IDLE, State.ERROR],

            State.COMPLETE: [],
            State.ERROR: [State.INIT] 
        }
    
    def can_transition_to(self, new_state):
        return new_state in self.transitions[self.current_state]
    
    def transition_to(self, new_state):
        if self.can_transition_to(new_state):
            print(f"State transition: {self.current_state.value} → {new_state.value}")
            self.current_state = new_state
            return True
        else:
            print(f"Invalid transition from {self.current_state.value} to {new_state.value}")
            return False
    
    def execute_current_state(self):
        print(f"Executing state: {self.current_state.value}")
        
        try:
            if self.current_state == State.INIT:
                self._init_state()
            elif self.current_state == State.GENERATE_DATA:
                self._generate_data_state()
            elif self.current_state == State.SAVE_DATA:
                self._save_data_state()


            elif self.current_state == State.CREATE_MODEL:
                self._create_model_state()
            elif self.current_state == State.TRAIN_MODEL:
                self._train_model_state()
            elif self.current_state == State.PREDICT:
                self._predict_state()

            elif self.current_state == State.IDLE:
                self._idle_state()

            elif self.current_state == State.VISUALIZE:
                self._visualize_state()

            elif self.current_state == State.ERROR:
                self._error_state()
            elif self.current_state == State.COMPLETE:
                self._complete_state()
                
        except Exception as e:
            self.error_message = str(e)
            print(f"Error in state {self.current_state.value}: {e}")
            self.transition_to(State.ERROR)
    



    def _init_state(self):
        print(f"Initializing data processing for {self.nr_of_cities} cities")
        self.transition_to(State.GENERATE_DATA)
    
    def _generate_data_state(self):
        print("Generating city and edge data...")
        self.cities, self.edges = preprocessing(self.nr_of_cities)
        print(f"Generated {len(self.cities)} cities and {len(self.edges)} edges")
        self.transition_to(State.SAVE_DATA)
    
    def _save_data_state(self):
        print("Saving data...")
        # Save complete dataset
        save_to_json(self.cities, self.edges, file_path='data/processed', filename='processed.json')
        
        # Split and save train/validation/test sets
        self.file_paths = split_and_save_data(self.cities, self.edges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        print("Data saved successfully")
        self.transition_to(State.CREATE_MODEL)
    
    def _visualize_state(self):
        print("Creating visualizations...")
        statistics(self.cities, self.edges)
        print("Visualizations completed")
        self.transition_to(State.IDLE)
    
    def _create_model_state(self):
        print("Creating neural network model...")
        # Initialize model with 7 input features
        self.model = CityAgglomerationGNN(input_dim=7, hidden_dim=64, output_dim=1)
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        print("Model created successfully")
        self.transition_to(State.TRAIN_MODEL)
    
    def _train_model_state(self):
        print("Training neural network model...")
        
        # Load training data
        try:
            with open(self.file_paths['train'], 'r') as f:
                train_data = json.load(f)
            
            cities_data = train_data['cities']
            edges_data = train_data['edges']
            
            # Prepare features and labels
            feature_keys = ['x', 'y', 'population', 'gdp_per_capita', 'education_score', 'infrastructure_score', 'location_score']
            
            node_features = []
            node_labels = []
            
            for city in cities_data:
                features = [city.get(key, 0) for key in feature_keys]
                node_features.append(features)
                node_labels.append(city.get('agglomeration', 0))
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(node_labels, dtype=torch.float).view(-1, 1)
            
            # Prepare edge indices
            edge_indices = []
            for edge in edges_data:
                if len(edge) >= 2:
                    city1_idx, city2_idx = edge[0], edge[1]
                    edge_indices.extend([[city1_idx, city2_idx], [city2_idx, city1_idx]])
            
            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            
            # Training loop
            self.model.train()
            num_epochs = 100
            
            for epoch in range(num_epochs):
                self.optimizer.zero_grad()
                
                # Forward pass
                out = self.model(x, edge_index)
                loss = F.mse_loss(out, y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                if (epoch + 1) % 20 == 0:
                    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
            
            print("Model training completed")
            self.transition_to(State.IDLE)
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.error_message = str(e)
            self.transition_to(State.ERROR)
    
    def _idle_state(self):
        print("System is in idle state")
        self.transition_to(State.COMPLETE)
    
    def _predict_state(self):
        print("Making predictions with trained model...")
        # TODO: Implement actual prediction logic
        # Use the trained model to make predictions on test data
        print("Predictions completed")
        self.transition_to(State.IDLE)
    
    def _complete_state(self):
        print("Data processing workflow completed successfully!")
    
    def _error_state(self):
        print(f"In error state: {self.error_message}")
        print("Use reset() to restart the workflow")

    def reset(self):
        print("Resetting state machine...")
        self.current_state = State.INIT
        self.cities = None
        self.edges = None
        self.file_paths = None
        self.error_message = None
        self.model = None
        self.optimizer = None
    
    def run_workflow(self):
        print("Starting data processing workflow...")
        
        while self.current_state not in [State.COMPLETE, State.ERROR]:
            self.execute_current_state()
        
        if self.current_state == State.COMPLETE:
            print("Workflow completed successfully!")
        else:
            print("Workflow ended with error")
    
    def get_status(self):
        return {
            'current_state': self.current_state.value,
            'cities_loaded': self.cities is not None,
            'edges_loaded': self.edges is not None,
            'files_saved': self.file_paths is not None,
            'model_created': self.model is not None,
            'error': self.error_message
        }


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
    print("=== Using State Machine ===")
    state_machine = DataProcessingStateMachine(nr_of_cities=1000)
    state_machine.run_workflow()
    
    print(f"\nFinal Status: {state_machine.get_status()}")
    

    # nr_of_cities = 30
    # cities, edges = preprocessing(nr_of_cities)
    # save_to_json(cities, edges, file_path='data/processed', filename='processed.json')
    # file_paths = split_and_save_data(cities, edges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    # statistics(cities, edges)

