"""
Multi-Objective Optimization Algorithms for Circular Economy Seismic Retrofit
Kevin Karanja Kuria - Supporting Code for Publication

This module contains the optimization algorithms and mathematical formulations
used in the circular economy seismic retrofit framework.
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

class CircularRetrofitOptimizer:
    """
    Multi-objective optimization for circular economy seismic retrofits
    """
    
    def __init__(self, building_params: Dict[str, Any]):
        """
        Initialize optimizer with building parameters
        
        Args:
            building_params: Dictionary containing building specifications
        """
        self.building_params = building_params
        self.materials_db = self._load_materials_database()
        self.scaler = MinMaxScaler()
        
    def _load_materials_database(self) -> pd.DataFrame:
        """Load materials database with embodied carbon and cost data"""
        materials_data = {
            'material': ['concrete_jacketing', 'steel_plates', 'frp_wraps', 
                        'precast_overlay', 'damping_device'],
            'embodied_carbon': [0.15, 2.60, 9.00, 0.12, 3.00],  # kg CO2 eq/kg
            'cost_per_kg': [0.10, 1.80, 25.0, 0.11, 4.00],     # €/kg
            'recovery_rate': [0.10, 0.85, 0.70, 0.90, 0.90],   # fraction
            'density': [2400, 7850, 1600, 2300, 1200]           # kg/m³
        }
        return pd.DataFrame(materials_data)
    
    def seismic_performance_index(self, design_vars: np.ndarray) -> float:
        """
        Calculate seismic performance index (f1)
        
        Args:
            design_vars: Design variables [thickness, reinforcement, etc.]
        
        Returns:
            Seismic performance index (0-1, higher is better)
        """
        # Simplified seismic capacity calculation
        thickness = design_vars[0]  # mm
        reinforcement_ratio = design_vars[1]  # %
        material_type = int(design_vars[2])  # material index
        
        # Base capacity calculation
        base_capacity = 0.6  # baseline building capacity
        
        # Improvement from thickness
        thickness_factor = min(1.5, 1.0 + thickness / 200)
        
        # Improvement from reinforcement
        reinf_factor = min(1.4, 1.0 + reinforcement_ratio / 2.0)
        
        # Material efficiency factor
        material_factors = [1.0, 1.3, 1.2, 1.1, 1.4]  # based on material type
        material_factor = material_factors[material_type]
        
        # Calculate performance index
        performance = base_capacity * thickness_factor * reinf_factor * material_factor
        return min(1.0, performance)
    
    def embodied_carbon(self, design_vars: np.ndarray) -> float:
        """
        Calculate total embodied carbon (f2) - to be minimized
        
        Args:
            design_vars: Design variables
        
        Returns:
            Total embodied carbon in kg CO2 eq/m²
        """
        thickness = design_vars[0] / 1000  # convert mm to m
        reinforcement_ratio = design_vars[1] / 100  # convert % to fraction
        material_type = int(design_vars[2])
        area = self.building_params.get('wall_area', 100)  # m²
        
        # Get material properties
        material_row = self.materials_db.iloc[material_type]
        carbon_factor = material_row['embodied_carbon']
        density = material_row['density']
        
        # Calculate material quantity
        volume = thickness * area  # m³
        mass = volume * density    # kg
        
        # Add reinforcement if applicable
        if reinforcement_ratio > 0:
            steel_mass = mass * reinforcement_ratio
            steel_carbon = steel_mass * 2.6  # kg CO2 eq/kg for steel
        else:
            steel_carbon = 0
        
        # Total embodied carbon
        total_carbon = mass * carbon_factor + steel_carbon
        return total_carbon / area  # kg CO2 eq/m²
    
    def lifecycle_cost(self, design_vars: np.ndarray) -> float:
        """
        Calculate lifecycle cost (f3) - to be minimized
        
        Args:
            design_vars: Design variables
        
        Returns:
            Total lifecycle cost in €/m²
        """
        thickness = design_vars[0] / 1000  # convert mm to m
        reinforcement_ratio = design_vars[1] / 100
        material_type = int(design_vars[2])
        area = self.building_params.get('wall_area', 100)
        
        # Get material properties
        material_row = self.materials_db.iloc[material_type]
        cost_per_kg = material_row['cost_per_kg']
        density = material_row['density']
        
        # Calculate material costs
        volume = thickness * area
        mass = volume * density
        material_cost = mass * cost_per_kg
        
        # Installation costs (30% of material cost)
        installation_cost = material_cost * 0.3
        
        # Maintenance costs (present value, 5% of initial cost)
        maintenance_cost = material_cost * 0.05
        
        # End-of-life costs/credits
        recovery_rate = material_row['recovery_rate']
        eol_credit = material_cost * 0.2 * recovery_rate  # 20% value recovery
        
        total_cost = material_cost + installation_cost + maintenance_cost - eol_credit
        return total_cost / area  # €/m²
    
    def material_circularity_index(self, design_vars: np.ndarray) -> float:
        """
        Calculate Material Circularity Index (f4) - to be maximized
        
        Args:
            design_vars: Design variables
        
        Returns:
            MCI value (0-1, higher is better for circularity)
        """
        material_type = int(design_vars[2])
        
        # Get material properties
        material_row = self.materials_db.iloc[material_type]
        recovery_rate = material_row['recovery_rate']
        
        # Simplified MCI calculation based on:
        # - Material recovery potential
        # - Design for disassembly score
        # - Reuse potential
        
        # Base circularity from material properties
        base_mci = recovery_rate
        
        # Design for disassembly bonus
        if material_type in [1, 3, 4]:  # Steel, precast, damping - easier to disassemble
            dfd_bonus = 0.2
        else:
            dfd_bonus = 0.0
        
        # Modularity bonus
        thickness = design_vars[0]
        if thickness > 100:  # Thicker elements are more modular
            modularity_bonus = 0.1
        else:
            modularity_bonus = 0.0
        
        mci = min(1.0, base_mci + dfd_bonus + modularity_bonus)
        return mci
    
    def objective_function(self, design_vars: np.ndarray) -> np.ndarray:
        """
        Multi-objective function combining all objectives
        
        Args:
            design_vars: Design variables
        
        Returns:
            Array of objective function values
        """
        # Calculate individual objectives
        f1 = self.seismic_performance_index(design_vars)
        f2 = self.embodied_carbon(design_vars)
        f3 = self.lifecycle_cost(design_vars)
        f4 = self.material_circularity_index(design_vars)
        
        # Convert to minimization problem
        # f1 and f4 are maximized, so negate them
        objectives = np.array([-f1, f2, f3, -f4])
        
        return objectives
    
    def optimize(self, n_generations: int = 100) -> Dict[str, Any]:
        """
        Perform multi-objective optimization using differential evolution
        
        Args:
            n_generations: Number of optimization generations
        
        Returns:
            Optimization results
        """
        # Define bounds for design variables
        # [thickness (mm), reinforcement_ratio (%), material_type]
        bounds = [(50, 300), (0, 4), (0, 4)]
        
        # Store Pareto front solutions
        pareto_solutions = []
        pareto_objectives = []
        
        # Run multiple optimizations with different weights
        weights_combinations = [
            [0.4, 0.2, 0.2, 0.2],  # Performance emphasis
            [0.2, 0.4, 0.2, 0.2],  # Carbon emphasis
            [0.2, 0.2, 0.4, 0.2],  # Cost emphasis
            [0.2, 0.2, 0.2, 0.4],  # Circularity emphasis
            [0.25, 0.25, 0.25, 0.25]  # Balanced
        ]
        
        for weights in weights_combinations:
            def weighted_objective(x):
                objectives = self.objective_function(x)
                return np.sum(np.array(weights) * objectives)
            
            result = differential_evolution(
                weighted_objective,
                bounds,
                maxiter=n_generations,
                seed=42
            )
            
            if result.success:
                pareto_solutions.append(result.x)
                objectives = self.objective_function(result.x)
                pareto_objectives.append(objectives)
        
        # Convert to arrays
        pareto_solutions = np.array(pareto_solutions)
        pareto_objectives = np.array(pareto_objectives)
        
        # Find the best compromise solution
        # Normalize objectives and find minimum distance to ideal point
        normalized_obj = self.scaler.fit_transform(pareto_objectives)
        ideal_point = np.min(normalized_obj, axis=0)
        distances = np.sqrt(np.sum((normalized_obj - ideal_point)**2, axis=1))
        best_idx = np.argmin(distances)
        
        best_solution = pareto_solutions[best_idx]
        best_objectives = pareto_objectives[best_idx]
        
        # Calculate performance metrics for best solution
        metrics = {
            'seismic_index': -best_objectives[0],
            'embodied_carbon': best_objectives[1],
            'lifecycle_cost': best_objectives[2],
            'circularity_index': -best_objectives[3],
            'design_variables': {
                'thickness_mm': best_solution[0],
                'reinforcement_ratio': best_solution[1],
                'material_type': int(best_solution[2])
            }
        }
        
        return {
            'best_solution': best_solution,
            'best_objectives': best_objectives,
            'metrics': metrics,
            'pareto_front': {
                'solutions': pareto_solutions,
                'objectives': pareto_objectives
            }
        }

def run_case_study():
    """Run optimization for sample building"""
    building_params = {
        'wall_area': 150,  # m²
        'height': 12,      # m
        'occupancy': 'office'
    }
    
    optimizer = CircularRetrofitOptimizer(building_params)
    results = optimizer.optimize(n_generations=50)
    
    print("🏗️  Optimization Results:")
    print(f"   Seismic Index: {results['metrics']['seismic_index']:.3f}")
    print(f"   Embodied Carbon: {results['metrics']['embodied_carbon']:.1f} kg CO₂ eq/m²")
    print(f"   Lifecycle Cost: {results['metrics']['lifecycle_cost']:.1f} €/m²")
    print(f"   Circularity Index: {results['metrics']['circularity_index']:.3f}")
    
    return results

if __name__ == "__main__":
    results = run_case_study()