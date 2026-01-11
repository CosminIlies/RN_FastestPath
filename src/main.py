from preprocessing.preprocessing import preprocessing, preprocessing_read_from_json
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
from flask import Flask, request, jsonify, render_template
import threading
import os
from datetime import datetime

# Get the parent directory of src
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)



class State(Enum):
    """Define the possible states"""
    INIT = "initialization"
    GENERATE_DATA = "generating_data"
    SAVE_DATA = "saving_data"
    VISUALIZE = "visualizing"

    SAVE_MODEL = "saving_model"
    LOAD_MODEL = "loading_model"

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
        self.original_cities = None  # Store original city data for comparison
        self.edges = None
        self.file_paths = None
        self.error_message = None
        self.model = None
        self.optimizer = None
        self.predictions = None
        self.predictions = None
        
        # Define valid transitions
        self.transitions = {
            State.INIT: [State.GENERATE_DATA, State.LOAD_MODEL, State.ERROR],
            State.GENERATE_DATA: [State.SAVE_DATA, State.ERROR],
            State.SAVE_DATA: [State.CREATE_MODEL, State.IDLE, State.COMPLETE, State.ERROR],

            State.SAVE_MODEL: [State.IDLE, State.ERROR],
            State.LOAD_MODEL: [State.IDLE, State.ERROR],

            State.CREATE_MODEL : [State.TRAIN_MODEL, State.ERROR],
            State.TRAIN_MODEL : [State.IDLE, State.SAVE_MODEL, State.PREDICT, State.ERROR],
            State.PREDICT : [State.IDLE, State.ERROR],
            State.IDLE : [State.COMPLETE, State.PREDICT, State.VISUALIZE, State.ERROR],

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
        # print(f"Executing state: {self.current_state.value}")
        
        try:
            if self.current_state == State.INIT:
                self._init_state()
            elif self.current_state == State.GENERATE_DATA:
                self._generate_data_state()
            elif self.current_state == State.SAVE_DATA:
                self._save_data_state()


            elif self.current_state == State.SAVE_MODEL:
                self._save_model()
            elif self.current_state == State.LOAD_MODEL:
                self._load_model()

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
        print(f"Initializing for {self.nr_of_cities} cities")

        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        model_path = os.path.join(models_dir, 'city_agglomeration_gnn.pt')
        
        if os.path.exists(model_path):
            self.transition_to(State.LOAD_MODEL)
        else:
            self.transition_to(State.GENERATE_DATA)
    
    def _generate_data_state(self):
        print("Generating city and edge data...")
        self.cities, self.edges = preprocessing(self.nr_of_cities)
        
        # Store original city data for comparison
        import copy
        self.original_cities = copy.deepcopy(self.cities)
        
        # Clear any existing predictions when generating new data
        for city in self.cities:
            city.pop('predicted_agglomeration', None)
            
        print(f"Generated {len(self.cities)} cities and {len(self.edges)} edges")
        self.transition_to(State.SAVE_DATA)
    
    def _save_data_state(self):
        print("Saving data...")
        # Save complete dataset
        save_to_json(self.cities, self.edges, file_path='data/processed', filename='processed.json')
        save_to_json(self.cities, self.edges, file_path='data/generated', filename='generated.json')
        
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
            num_epochs = 1000
            
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
            
            # Save the trained model
            # self._save_model()
            
            self.transition_to(State.SAVE_MODEL)
            
        except Exception as e:
            print(f"Training failed: {e}")
            self.error_message = str(e)
            self.transition_to(State.ERROR)


    def _save_model(self):
        try:
            # Create models directory if it doesn't exist
            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
            os.makedirs(models_dir, exist_ok=True)
            
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"city_agglomeration_gnn.pt"
            model_path = os.path.join(models_dir, model_filename)
            
            # Save model state dict, optimizer state dict, and model configuration
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'model_config': {
                    'input_dim': 7,
                    'hidden_dim': 64,
                    'output_dim': 1
                },
                'nr_of_cities': self.nr_of_cities,
                'timestamp': timestamp
            }, model_path)
            
            print(f"Model saved successfully to: {model_path}")
            self.transition_to(State.IDLE)
            
        except Exception as e:
            print(f"Failed to save model: {e}")
            self.transition_to(State.ERROR)

    def _load_model(self):
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        model_path = os.path.join(models_dir, 'city_agglomeration_gnn.pt')
        try:
            # Load the saved model
            checkpoint = torch.load(model_path)
            
            # Extract model configuration
            model_config = checkpoint['model_config']
            
            # Initialize model with saved configuration
            self.model = CityAgglomerationGNN(
                input_dim=model_config['input_dim'],
                hidden_dim=model_config['hidden_dim'],
                output_dim=model_config['output_dim']
            )
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Initialize optimizer
            self.optimizer = Adam(self.model.parameters(), lr=0.01)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Set model to evaluation mode
            self.model.eval()
            
            print(f"Model loaded successfully from: {model_path}")
            print(f"Model was trained on {checkpoint.get('nr_of_cities', 'unknown')} cities")
            print(f"Model timestamp: {checkpoint.get('timestamp', 'unknown')}")
            
            # Try to load existing processed data to display in the UI
            self._load_existing_processed_data()
            
            self.transition_to(State.IDLE)

            
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.transition_to(State.ERROR)

    def _load_existing_processed_data(self):
        """Try to load existing processed data to display in the UI when model is loaded"""
        try:
            # Try multiple possible file locations
            possible_files = [
                'data/processed/processed.json',
                'data/generated/generated.json',
                'data/aigenerated.json'
            ]
            
            for processed_file in possible_files:
                if os.path.exists(processed_file):
                    print(f"Loading existing data from: {processed_file}")
                    self.cities, self.edges = preprocessing_read_from_json(processed_file)
                    
                    # Store original city data for comparison
                    import copy
                    self.original_cities = copy.deepcopy(self.cities)
                    
                    print(f"Loaded {len(self.cities)} cities and {len(self.edges)} edges from processed data")
                    return True
            
            print("No existing processed data found - you can generate new data or make predictions with existing model")
            return False
                
        except Exception as e:
            print(f"Warning: Could not load existing processed data: {e}")
            print("You can generate new data or the model is still ready for predictions")
            return False

    
    def _idle_state(self):
        pass 
        # print("System is in idle state")
    
    def _predict_state(self):
        print("Making predictions with trained model...")
        
        try:
            # Check if model exists
            if self.model is None:
                raise Exception("Model not initialized. Please train the model first.")
            
            # Load test data for predictions
            if self.file_paths and 'test' in self.file_paths:
                with open(self.file_paths['test'], 'r') as f:
                    test_data = json.load(f)
            else:
                # Use validation data if test data not available
                if self.file_paths and 'validation' in self.file_paths:
                    with open(self.file_paths['validation'], 'r') as f:
                        test_data = json.load(f)
                else:
                    # Use current cities and edges as fallback
                    test_data = {'cities': self.cities, 'edges': self.edges}
            
            cities_data = test_data['cities']
            edges_data = test_data['edges']
            
            # Prepare features for prediction
            feature_keys = ['x', 'y', 'population', 'gdp_per_capita', 'education_score', 'infrastructure_score', 'location_score']
            
            node_features = []
            actual_labels = []
            
            for city in cities_data:
                features = [city.get(key, 0) for key in feature_keys]
                node_features.append(features)
                actual_labels.append(city.get('agglomeration', 0))
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            y_actual = torch.tensor(actual_labels, dtype=torch.float).view(-1, 1)
            
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
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(x, edge_index)
                
                # Calculate prediction metrics
                mse_loss = F.mse_loss(predictions, y_actual)
                mae_loss = F.l1_loss(predictions, y_actual)
                
                # Store predictions and metrics
                self.predictions = {
                    'predicted_values': predictions.squeeze().tolist(),
                    'actual_values': actual_labels,
                    'mse_loss': mse_loss.item(),
                    'mae_loss': mae_loss.item(),
                    'num_predictions': len(actual_labels)
                }
                
                # Update cities with predicted agglomeration values for visualization
                predicted_values = predictions.squeeze().tolist()
                for i, city in enumerate(self.cities):
                    if i < len(predicted_values):
                        city['predicted_agglomeration'] = predicted_values[i]
                
                print(f"Predictions completed successfully!")
                print(f"Number of predictions: {self.predictions['num_predictions']}")
                print(f"MSE Loss: {self.predictions['mse_loss']:.4f}")
                print(f"MAE Loss: {self.predictions['mae_loss']:.4f}")
                
                # Print some sample predictions
                print("\nSample predictions (first 5):")
                for i in range(min(5, len(self.predictions['predicted_values']))):
                    predicted = self.predictions['predicted_values'][i]
                    actual = self.predictions['actual_values'][i]
                    print(f"  City {i}: Predicted={predicted:.3f}, Actual={actual:.3f}")
            
            self.transition_to(State.IDLE)
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            self.error_message = str(e)
            self.transition_to(State.ERROR)
    
    def _complete_state(self):
        print("Data processing workflow completed successfully!")
    
    def _error_state(self):
        print(f"In error state: {self.error_message}")
        print("Use reset() to restart the workflow")

    def reset(self):
        print("Resetting state machine...")
        self.current_state = State.INIT
        
        # Clear predicted agglomeration values from cities if they exist
        if self.cities:
            for city in self.cities:
                city.pop('predicted_agglomeration', None)
        
        self.cities = None
        self.original_cities = None
        self.edges = None
        self.file_paths = None
        self.error_message = None
        self.model = None
        self.optimizer = None
        self.predictions = None
    
    def run_workflow(self):
        print("Starting data processing workflow...")
        
        while self.current_state not in [State.COMPLETE, State.ERROR, State.IDLE]:
            self.execute_current_state()
        
        if self.current_state == State.COMPLETE:
            print("Workflow completed successfully!")
        elif self.current_state == State.IDLE:
            print("Workflow is ready - model and data loaded")
        else:
            print("Workflow ended with error")
    
    def get_status(self):
        return {
            'current_state': self.current_state.value,
            'cities_loaded': self.cities is not None,
            'edges_loaded': self.edges is not None,
            'files_saved': self.file_paths is not None,
            'model_created': self.model is not None,
            'predictions_available': self.predictions is not None,
            'error': self.error_message,
            'predictions_summary': {
                'num_predictions': len(self.predictions['predicted_values']) if self.predictions else 0,
                'mse_loss': self.predictions['mse_loss'] if self.predictions else None,
                'mae_loss': self.predictions['mae_loss'] if self.predictions else None
            } if self.predictions else None
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



state_machine = DataProcessingStateMachine(nr_of_cities=30)



@app.route("/")
def hello_world():
    return render_template('dashboard.html')

@app.route("/api/status")
def get_status():
    """API endpoint to get current status"""
    return jsonify(state_machine.get_status())

@app.route("/api/control/<action>", methods=["POST"])
def control_state_machine(action):
    """Control the state machine"""
    try:
        if action == "generate":
            state_machine.reset()
            state_machine.transition_to(State.GENERATE_DATA)
            state_machine.execute_current_state()
        elif action == "train":
            if state_machine.current_state == State.IDLE:
                state_machine.transition_to(State.TRAIN_MODEL)
                state_machine.execute_current_state()
        elif action == "predict":
            if state_machine.current_state == State.IDLE:
                state_machine.transition_to(State.PREDICT)
                state_machine.execute_current_state()
        elif action == "visualize":
            if state_machine.current_state == State.IDLE:
                state_machine.transition_to(State.VISUALIZE)
                state_machine.execute_current_state()
        elif action == "reset":
            state_machine.reset()
        else:
            return jsonify({"error": "Invalid action"}), 400
            
        return jsonify(state_machine.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/model")
def get_model():
    """
    GET /api/model endpoint that returns the whole graph structure
    """
    try:
        # Check if we have cities and edges data
        if state_machine.cities is None or state_machine.edges is None:
            return jsonify({
                "error": "No graph data available. Please generate data first.",
                "status": state_machine.current_state.value
            }), 400
        
        # Prepare graph structure for response
        graph_data = {
            "nodes": state_machine.cities,
            "edges": state_machine.edges,
            "metadata": {
                "num_nodes": len(state_machine.cities),
                "num_edges": len(state_machine.edges),
                "state": state_machine.current_state.value,
                "model_trained": state_machine.model is not None
            }
        }
        
        return jsonify(graph_data)
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to retrieve graph: {str(e)}"
        }), 500

@app.route("/api/model", methods=["PATCH"])
def patch_model():
    """
    PATCH /api/model endpoint that predicts agglomeration if parameters change
    """
    try:
        # Check if model exists
        if state_machine.model is None:
            return jsonify({
                "error": "Model not available. Please train the model first.",
                "status": state_machine.current_state.value
            }), 400
        
        # Check if cities data exists
        if state_machine.cities is None:
            return jsonify({
                "error": "No city data available.",
                "status": state_machine.current_state.value
            }), 400
        
        # Get request body
        data = request.get_json()
        if not data:
            return jsonify({"error": "No parameter changes provided in request body"}), 400
        
        # Handle reset request
        if data.get('reset_to_original', False) and state_machine.original_cities:
            affected_cities = data.get('affected_cities', [])
            for city_idx in affected_cities:
                if city_idx < len(state_machine.cities) and city_idx < len(state_machine.original_cities):
                    # Restore original values but keep predicted_agglomeration if it exists
                    predicted_value = state_machine.cities[city_idx].get('predicted_agglomeration')
                    import copy
                    state_machine.cities[city_idx] = copy.deepcopy(state_machine.original_cities[city_idx])
                    if predicted_value is not None:
                        state_machine.cities[city_idx]['predicted_agglomeration'] = predicted_value
            
            # Return current data
            graph_data = {
                "nodes": state_machine.cities,
                "edges": state_machine.edges,
                "metadata": {
                    "num_nodes": len(state_machine.cities),
                    "num_edges": len(state_machine.edges),
                    "state": state_machine.current_state.value,
                    "model_trained": state_machine.model is not None,
                    "reset_applied": True,
                    "affected_cities": affected_cities
                }
            }
            return jsonify(graph_data)
        
        # Create modified cities with parameter changes
        modified_cities = []
        affected_cities = data.get('affected_cities', list(range(len(state_machine.cities))))
        parameter_changes = {k: v for k, v in data.items() if k not in ['affected_cities', 'individual_values']}
        individual_values = data.get('individual_values', False)
        
        for i, city in enumerate(state_machine.cities):
            modified_city = city.copy()
            
            # Apply parameter changes only to affected cities
            if i in affected_cities:
                for param, new_value in parameter_changes.items():
                    if param in modified_city:
                        if individual_values:
                            # Use absolute values directly
                            modified_city[param] = new_value
                        else:
                            # Use multipliers (old behavior)
                            if param in ['population', 'gdp_per_capita']:
                                modified_city[param] *= new_value
                            else:
                                modified_city[param] = new_value
                        
                        # Ensure values stay within reasonable bounds
                        if param in ['education_score', 'infrastructure_score', 'location_score']:
                            modified_city[param] = max(0.1, min(1.0, float(modified_city[param])))
                        elif param == 'population':
                            modified_city[param] = max(1000, int(modified_city[param]))
                        elif param == 'gdp_per_capita':
                            modified_city[param] = max(10000, float(modified_city[param]))
            
            modified_cities.append(modified_city)
        
        # Prepare features for prediction with modified parameters
        feature_keys = ['x', 'y', 'population', 'gdp_per_capita', 'education_score', 'infrastructure_score', 'location_score']
        
        node_features = []
        for city in modified_cities:
            features = [city.get(key, 0) for key in feature_keys]
            node_features.append(features)
        
        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Prepare edge indices
        edge_indices = []
        for edge in state_machine.edges:
            if len(edge) >= 2:
                city1_idx, city2_idx = edge[0], edge[1]
                edge_indices.extend([[city1_idx, city2_idx], [city2_idx, city1_idx]])
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Make predictions with the trained model
        state_machine.model.eval()
        with torch.no_grad():
            predicted_agglomerations = state_machine.model(x, edge_index)
            predicted_values = predicted_agglomerations.squeeze().tolist()
        
        # Update cities with predicted agglomeration values
        for i, city in enumerate(modified_cities):
            city['predicted_agglomeration'] = predicted_values[i]
        
        # Update the actual state machine cities data with the modifications
        # This ensures the changes persist and are visible in tooltips
        if individual_values and affected_cities:
            for city_idx in affected_cities:
                if city_idx < len(state_machine.cities):
                    # Update the actual city data with the new parameter values
                    for param, new_value in parameter_changes.items():
                        if param in state_machine.cities[city_idx]:
                            state_machine.cities[city_idx][param] = modified_cities[city_idx][param]
                    
                    # Also update with the predicted agglomeration
                    state_machine.cities[city_idx]['predicted_agglomeration'] = predicted_values[city_idx]
        
        # Prepare response in same format as GET /model
        graph_data = {
            "nodes": modified_cities,
            "edges": state_machine.edges,
            "metadata": {
                "num_nodes": len(modified_cities),
                "num_edges": len(state_machine.edges),
                "state": state_machine.current_state.value,
                "model_trained": state_machine.model is not None,
                "prediction_applied": True,
                "parameter_changes": parameter_changes,
                "affected_cities": affected_cities,
                "total_affected": len(affected_cities)
            }
        }
        
        return jsonify(graph_data)
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to predict agglomeration: {str(e)}"
        }), 500
    



def start_state_machine():
    state_machine.run_workflow()

def start_web_server():
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    

    states_thread = threading.Thread(target=start_state_machine)
    web_thread = threading.Thread(target=start_web_server)

    states_thread.start()
    web_thread.start()

    states_thread.join()
    web_thread.join()
    
    
    # print(f"\nFinal Status: {state_machine.get_status()}")
    
    
    # nr_of_cities = 30
    # cities, edges = preprocessing(nr_of_cities)
    # save_to_json(cities, edges, file_path='data/processed', filename='processed.json')
    # file_paths = split_and_save_data(cities, edges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    # statistics(cities, edges)

