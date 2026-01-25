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
from sklearn.metrics import accuracy_score, r2_score
import csv
import time

# Get the parent directory of src
template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

learning_rate = 0.001
dynamic_learing_rate = True
num_epochs = 3500


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
    def __init__(self, nr_of_cities=30, split_data=True, use_lr_scheduler=True):
        self.current_state = State.INIT
        self.nr_of_cities = nr_of_cities
        self.split_data = split_data
        self.use_lr_scheduler = use_lr_scheduler
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
        print("Generating city and edge data with neighborhood influence...")
        self.cities, self.edges = preprocessing(self.nr_of_cities, apply_influence=True)
        
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
        
        # Conditionally split data based on parameter
        if self.split_data:
            print("Splitting data into train/validation/test sets...")
            self.file_paths = split_and_save_data(self.cities, self.edges, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
            print("Data split and saved successfully")
        else:
            print("Data splitting disabled - using full dataset")
            # Create file_paths pointing to the full dataset for training
            self.file_paths = {
                'train': 'data/processed/processed.json',
                'validation': 'data/processed/processed.json',
                'test': 'data/processed/processed.json'
            }
        
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
        self.model = CityAgglomerationGNN(input_dim=5, hidden_dim=128, output_dim=1)  # Increased from 64 to 128
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5) 
        
        # Conditionally add learning rate scheduler
        if self.use_lr_scheduler:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=200)
            print("Model created with ReduceLROnPlateau scheduler")
        else:
            self.scheduler = None
            print("Model created without learning rate scheduler")
        print("Model created successfully")
        self.transition_to(State.TRAIN_MODEL)
    
    def _train_model_state(self):
        print("Training neural network model...")
        
        # Record training start time
        training_start_time = time.time()
        
        # Load training data
        try:
            with open(self.file_paths['train'], 'r') as f:
                train_data = json.load(f)
            
            cities_data = train_data['cities']
            edges_data = train_data['edges']
            
            # Prepare features and labels
            feature_keys = [ 'population', 'gdp_per_capita', 'education_score', 'infrastructure_score', 'location_score']
            
            node_features = []
            node_labels = []
            
            for city in cities_data:
                features = [city.get(key, 0) for key in feature_keys]
                node_features.append(features)
                node_labels.append(city.get('agglomeration', 0))
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(node_labels, dtype=torch.float).view(-1, 1)
            
            # CRITICAL: Clamp target labels to [0,1] range for binary_cross_entropy
            y = torch.clamp(y, min=0.0, max=1.0)
            print(f"Target agglomeration stats: Min: {y.min():.3f}, Max: {y.max():.3f}, Mean: {y.mean():.3f}")
            
            # CRITICAL: Normalize features to prevent prediction collapse
            print(f"Feature stats before normalization:")
            print(f"  Min: {x.min(dim=0)[0]}")
            print(f"  Max: {x.max(dim=0)[0]}")
            print(f"  Mean: {x.mean(dim=0)}")
            
            # Standard normalization (zero mean, unit variance)
            x_mean = x.mean(dim=0, keepdim=True)
            x_std = x.std(dim=0, keepdim=True) + 1e-8  # Add small value to avoid division by zero
            x = (x - x_mean) / x_std
            
            # Store normalization parameters for consistent predictions
            self.feature_mean = x_mean
            self.feature_std = x_std
            
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
            
            # Record epoch timing
            epoch_start_time = time.time()
            
            # Setup CSV logging
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
            os.makedirs(results_dir, exist_ok=True)
            csv_filename = f"training_history.csv"
            csv_path = os.path.join(results_dir, csv_filename)
            
            # Create CSV file with headers
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['epoch', 'loss', 'mae', 'learning_rate']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            
            print(f"Training metrics will be logged to: {csv_path}")
            

            for epoch in range(num_epochs):
                self.optimizer.zero_grad()
                
                # Forward pass
                out = self.model(x, edge_index)
                
                # Debug: Check if model outputs are in valid range
                if (epoch + 1) % 100 == 0:
                    print(f"Model output stats - Min: {out.min().item():.4f}, Max: {out.max().item():.4f}")
                
                # Ensure model outputs are in [0,1] range (safety clamp)
                out = torch.clamp(out, min=1e-7, max=1-1e-7)  # Avoid exact 0 and 1 for BCE
                
                # Use binary cross entropy loss for outputs between 0 and 1
                # Use Huber loss for better robustness to outliers
                loss = F.smooth_l1_loss(out, y)
                # loss = F.mse_loss(out, y)
                
                # Backward pass
                loss.backward()
                # Add gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update learning rate scheduler if enabled
                if self.use_lr_scheduler and hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step(loss)
                
                # Early stopping and evaluation every 50 epochs
                if (epoch + 1) % 50 == 0:
                    with torch.no_grad():
                        self.model.eval()
                        predictions = self.model(x, edge_index)
                        
                        # Calculate metrics
                        mae = F.l1_loss(predictions, y).item()
                        r2 = r2_score(y.cpu().numpy(), predictions.detach().cpu().numpy())
                        
                        # Get current learning rate
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        
                        # Log to CSV
                        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                            fieldnames = ['epoch', 'loss', 'mae', 'learning_rate',]
                            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                            writer.writerow({
                                'epoch': epoch + 1,
                                'loss': loss.item(),
                                'mae': mae,
                                'learning_rate': current_lr,

                            })
                        
                        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, LR: {current_lr:.6f}")
                        
                        self.model.train()
            
            # Final evaluation
            self.model.eval()
            with torch.no_grad():
                final_predictions = self.model(x, edge_index)
                final_loss = F.smooth_l1_loss(final_predictions, y)

                mae = F.l1_loss(final_predictions, y)
                mse = F.mse_loss(final_predictions, y)
            
                
                # Calculate R² score
                r2 = r2_score(y.cpu().numpy(), final_predictions.detach().cpu().numpy())
                # Calculate absolute differences for detailed analysis
                abs_diffs = torch.abs(final_predictions - y).squeeze()
                abs_diff_max = abs_diffs.max().item()
                
            
            # Calculate total training time
            training_end_time = time.time()
            total_training_time = training_end_time - training_start_time
            epoch_time = total_training_time / num_epochs
            
            print("\n" + "="*60)
            print("MODEL TRAINING COMPLETED!")
            print("="*60)
            print(f"FINAL TRAINING METRICS:")
            print(f"   Final Loss: {final_loss.item():.4f}")
            print(f"   MAE: {mae.item():.4f}")
            print(f"   MSE: {mse.item():.4f}")
            print(f"   R2 Score: {r2:.4f}")
            print(f"   Worst acc: {(1-abs_diff_max) * 100}%")
            print(f"   Total Epochs: {num_epochs}")
            print(f"   Training Samples: {len(y)}")
            
            print("\n" + "="*60)
            print("TRAINING TIMING")
            print("="*60)
            print(f"Total Training Time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
            print(f"Average Time per Epoch: {epoch_time:.4f} seconds")
            print(f"Training Speed: {len(y)} samples × {num_epochs} epochs / {total_training_time:.2f}s = {(len(y) * num_epochs) / total_training_time:.2f} samples/sec")
            if total_training_time > 60:
                hours = int(total_training_time // 3600)
                minutes = int((total_training_time % 3600) // 60)
                seconds = int(total_training_time % 60)
                if hours > 0:
                    print(f"Training Duration: {hours}h {minutes}m {seconds}s")
                else:
                    print(f"Training Duration: {minutes}m {seconds}s")
            
            print("Model training completed")
            
            # Save the trained model  
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
                'feature_mean': getattr(self, 'feature_mean', None),
                'feature_std': getattr(self, 'feature_std', None),
                'model_config': {
                    'input_dim': 5,
                    'hidden_dim': 128,
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
            
            # Load normalization parameters if available
            if 'feature_mean' in checkpoint and checkpoint['feature_mean'] is not None:
                self.feature_mean = checkpoint['feature_mean']
                self.feature_std = checkpoint['feature_std']
                print(f"Loaded normalization parameters")
            else:
                print(f"No normalization parameters in saved model")
                self.feature_mean = None
                self.feature_std = None
            
            # Initialize optimizer
            self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
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
        
        # Record start time for prediction
        prediction_start_time = time.time()
        
        try:
            # Check if model exists
            if self.model is None:
                raise Exception("Model not initialized. Please train the model first.")

            if hasattr(self, 'cities') and self.cities:
                test_data = {'cities': self.cities, 'edges': self.edges}
                print(f"Using full dataset ({len(self.cities)} cities) for prediction")
            else:
                processed_path = 'data/processed/processed.json'
                if os.path.exists(processed_path):
                    with open(processed_path, 'r') as f:
                        test_data = json.load(f)
                    print(f"Loaded full dataset from processed.json ({len(test_data.get('cities', []))} cities)")
                else:
                    raise Exception("No city data available for prediction")
            
            cities_data = test_data['cities']
            edges_data = test_data['edges']
            
            # Prepare features for prediction
            feature_keys = ['population', 'gdp_per_capita', 'education_score', 'infrastructure_score', 'location_score']
            
            node_features = []
            actual_labels = []
            
            for city in cities_data:
                features = [city.get(key, 0) for key in feature_keys]
                node_features.append(features)
                actual_labels.append(city.get('agglomeration', 0))
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            y_actual = torch.tensor(actual_labels, dtype=torch.float).view(-1, 1)
            
            # Apply same normalization as training (CRITICAL for consistent predictions)
            if hasattr(self, 'feature_mean') and self.feature_mean is not None:
                x = (x - self.feature_mean) / self.feature_std
                print(f"Applied training normalization to features")
            else:
                # Fallback: calculate new normalization (warning - may cause inconsistent predictions)
                print(f"No training normalization found - calculating new normalization")
                x_mean = x.mean(dim=0, keepdim=True)
                x_std = x.std(dim=0, keepdim=True) + 1e-8
                x = (x - x_mean) / x_std
            
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
            
            # Record time for model inference only
            inference_start_time = time.time()
            
            with torch.no_grad():
                predictions = self.model(x, edge_index)
                
                # Record inference end time
                inference_end_time = time.time()
                inference_time = inference_end_time - inference_start_time
                
                # Validate prediction range
                pred_min, pred_max = predictions.min().item(), predictions.max().item()
                if pred_min < 0 or pred_max > 1:
                    print(f"Warning: Predictions outside [0,1] range - Min: {pred_min:.3f}, Max: {pred_max:.3f}")
                    predictions = torch.clamp(predictions, 0.0, 1.0)
                
                # Calculate prediction metrics with appropriate loss functions
                mae_loss = F.l1_loss(predictions, y_actual)


                # Calculate R² score
                r2 = r2_score(y_actual.cpu().numpy(), predictions.detach().cpu().numpy())
                # Calculate absolute differences
                abs_diffs = torch.abs(predictions - y_actual).squeeze().tolist()


                # Store predictions and metrics
                self.predictions = {
                    'predicted_values': predictions.squeeze().tolist(),
                    'actual_values': actual_labels,
                    'mae_loss': mae_loss.item(),
                    'r2_score': r2,
                    'num_predictions': len(actual_labels),
                    'absolute_differences':abs_diffs,
                    'prediction_stats': {
                        'min': pred_min,
                        'max': pred_max,
                        'mean': predictions.mean().item(),
                        'std': predictions.std().item()
                    }
                }
                
                # Update cities with predicted agglomeration values for visualization
                predicted_values = predictions.squeeze().tolist()
                
                # Clear all existing predictions first
                for city in self.cities:
                    city.pop('predicted_agglomeration', None)
                
                # Apply predictions based on which data was used for prediction
                if test_data['cities'] == self.cities:
                    # If using full dataset, apply predictions directly
                    for i, city in enumerate(self.cities):
                        if i < len(predicted_values):
                            city['predicted_agglomeration'] = predicted_values[i]
                else:
                    # If using split data, need to map predictions back to correct cities
                    # Match cities by name to handle split datasets correctly
                    prediction_mapping = {}
                    for i, test_city in enumerate(cities_data):
                        if i < len(predicted_values):
                            city_name = test_city.get('city_name')
                            if city_name:
                                prediction_mapping[city_name] = predicted_values[i]
                    
                    # Apply predictions to matching cities in self.cities
                    predicted_count = 0
                    for city in self.cities:
                        city_name = city.get('city_name')
                        if city_name in prediction_mapping:
                            city['predicted_agglomeration'] = prediction_mapping[city_name]
                            predicted_count += 1
                    
                    print(f"Applied predictions to {predicted_count} out of {len(self.cities)} total cities")
                
                print(f"Predictions completed successfully!")
                print(f"Number of predictions: {self.predictions['num_predictions']}")
                print(f"MAE Loss: {self.predictions['mae_loss']:.4f}")
                print(f"R² Score: {self.predictions['r2_score']:.4f} ({self.predictions['r2_score']*100:.1f}%)")
                print(f"Prediction range: [{pred_min:.3f}, {pred_max:.3f}]")
                
                # Record total prediction time
                prediction_end_time = time.time()
                total_prediction_time = prediction_end_time - prediction_start_time
                
                # Print timing information
                print("\n" + "="*50)
                print("PREDICTION TIMING")
                print("="*50)
                print(f"Model Inference Time: {inference_time:.4f} seconds")
                print(f"Total Prediction Time: {total_prediction_time:.4f} seconds")
                print(f"Predictions per second: {len(predictions):.0f} / {inference_time:.4f} = {len(predictions)/inference_time:.2f} predictions/sec")
                
                # Print detailed evaluation summary
                print("\n" + "="*50)
                print("FINAL EVALUATION METRICS")
                print("="*50)
                print(f"R2Init Score: {self.predictions['r2_score']:.6f} ({self.predictions['r2_score']*100:.2f}%)")
                print(f"Individual Absolute Differences:")
                abs_diffs = self.predictions['absolute_differences']
                print(f"   Min Difference: {min(abs_diffs):.6f}")
                print(f"   Max Difference: {max(abs_diffs):.6f}")
                print(f"   Std Deviation: {np.std(abs_diffs):.6f}")
                
                # Print some sample predictions
                print("\nSample predictions (first 5):")
                for i in range(min(5, len(self.predictions['predicted_values']))):
                    predicted = self.predictions['predicted_values'][i]
                    actual = self.predictions['actual_values'][i]
                    abs_diff = abs(predicted - actual)
                    print(f"  City {i}: Predicted={predicted:.3f}, Actual={actual:.3f}, |Diff|={abs_diff:.3f}")
            
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
            'split_data_enabled': self.split_data,
            'nr_of_cities': self.nr_of_cities,
            'use_lr_scheduler': self.use_lr_scheduler,
            'predictions_summary': {
                'num_predictions': len(self.predictions['predicted_values']) if self.predictions else 0,
                'mae_loss': self.predictions['mae_loss'] if self.predictions else None,
                'r2_score': self.predictions['r2_score'] if self.predictions else None
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



# state_machine = DataProcessingStateMachine(nr_of_cities=1, split_data=False, use_lr_scheduler=False)
# state_machine = DataProcessingStateMachine(nr_of_cities=75, split_data=True, use_lr_scheduler=dynamic_learing_rate)
state_machine = DataProcessingStateMachine(nr_of_cities=302, split_data=True, use_lr_scheduler=True)



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
        elif action == "predict_all":
            if state_machine.current_state == State.IDLE:
                original_split = state_machine.split_data
                state_machine.split_data = False
                state_machine.transition_to(State.PREDICT)
                state_machine.execute_current_state()
                state_machine.split_data = original_split  # Restore original setting
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

@app.route("/api/config", methods=["POST"])
def update_config():
    """Update configuration parameters"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No configuration data provided"}), 400
        
        # Update split_data parameter
        if 'split_data' in data:
            state_machine.split_data = bool(data['split_data'])
            print(f"Data splitting {'enabled' if state_machine.split_data else 'disabled'}")
        
        # Update number of cities
        if 'nr_of_cities' in data:
            nr_cities = int(data['nr_of_cities'])
            if 5 <= nr_cities <= 100:  # Reasonable bounds
                state_machine.nr_of_cities = nr_cities
                print(f"Number of cities set to: {nr_cities}")
            else:
                return jsonify({"error": "Number of cities must be between 5 and 100"}), 400
        
        # Update learning rate scheduler setting
        if 'use_lr_scheduler' in data:
            state_machine.use_lr_scheduler = bool(data['use_lr_scheduler'])
            print(f"Learning rate scheduler {'enabled' if state_machine.use_lr_scheduler else 'disabled'}")
        
        return jsonify({
            "message": "Configuration updated successfully",
            "config": {
                "split_data": state_machine.split_data,
                "nr_of_cities": state_machine.nr_of_cities,
                "use_lr_scheduler": state_machine.use_lr_scheduler
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current configuration"""
    return jsonify({
        "split_data": state_machine.split_data,
        "nr_of_cities": state_machine.nr_of_cities,
        "use_lr_scheduler": state_machine.use_lr_scheduler,
        "current_state": state_machine.current_state.value
    })

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
            "predictions": state_machine.predictions,  # Add predictions data
            "metadata": {
                "num_nodes": len(state_machine.cities),
                "num_edges": len(state_machine.edges),
                "state": state_machine.current_state.value,
                "model_trained": state_machine.model is not None,
                "has_predictions": state_machine.predictions is not None
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
    PATCH /api/model endpoint that changes city parameters and triggers prediction
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
        
        # Check if state machine is in IDLE state
        if state_machine.current_state != State.IDLE:
            return jsonify({
                "error": f"Cannot modify parameters while in {state_machine.current_state.value} state. Please wait for operation to complete.",
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
            
            # After reset, transition to PREDICT state to update predictions
            state_machine.transition_to(State.PREDICT)
            state_machine.execute_current_state()
            
            # Return current data
            graph_data = {
                "nodes": state_machine.cities,
                "edges": state_machine.edges,
                "predictions": state_machine.predictions,
                "metadata": {
                    "num_nodes": len(state_machine.cities),
                    "num_edges": len(state_machine.edges),
                    "state": state_machine.current_state.value,
                    "model_trained": state_machine.model is not None,
                    "has_predictions": state_machine.predictions is not None,
                    "reset_applied": True,
                    "affected_cities": affected_cities
                }
            }
            return jsonify(graph_data)
        
        # Apply parameter changes to cities
        affected_cities = data.get('affected_cities', list(range(len(state_machine.cities))))
        parameter_changes = {k: v for k, v in data.items() if k not in ['affected_cities', 'individual_values']}
        individual_values = data.get('individual_values', False)
        
        # Modify the actual cities in state machine
        for city_idx in affected_cities:
            if city_idx < len(state_machine.cities):
                city = state_machine.cities[city_idx]
                
                for param, new_value in parameter_changes.items():
                    if param in city:
                        if individual_values:
                            # Use absolute values directly
                            city[param] = new_value
                        else:
                            # Use multipliers (old behavior)
                            if param in ['population', 'gdp_per_capita']:
                                city[param] *= new_value
                            else:
                                city[param] = new_value
                        
                        # Ensure values stay within reasonable bounds
                        if param in ['education_score', 'infrastructure_score', 'location_score']:
                            city[param] = max(0.1, min(1.0, float(city[param])))
                        elif param == 'population':
                            city[param] = max(1000, int(city[param]))
                        elif param == 'gdp_per_capita':
                            city[param] = max(10000, float(city[param]))
        
        # Transition to PREDICT state to update predictions with new parameters
        print(f"Parameters updated for {len(affected_cities)} cities. Generating predictions...")
        state_machine.transition_to(State.PREDICT)
        state_machine.execute_current_state()
        
        # Prepare response with updated graph data
        graph_data = {
            "nodes": state_machine.cities,
            "edges": state_machine.edges,
            "predictions": state_machine.predictions,
            "metadata": {
                "num_nodes": len(state_machine.cities),
                "num_edges": len(state_machine.edges),
                "state": state_machine.current_state.value,
                "model_trained": state_machine.model is not None,
                "has_predictions": state_machine.predictions is not None,
                "parameters_updated": True,
                "parameter_changes": parameter_changes,
                "affected_cities": affected_cities,
                "total_affected": len(affected_cities)
            }
        }
        
        return jsonify(graph_data)
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to update parameters and predict: {str(e)}"
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
    

