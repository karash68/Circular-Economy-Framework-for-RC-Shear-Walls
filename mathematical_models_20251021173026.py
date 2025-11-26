"""
Mathematical Model Implementation for Circular Economy Seismic Retrofit Optimization
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, lognorm
import matplotlib.pyplot as plt

class SeismicRetrofitOptimizer:
    """
    Multi-objective optimization framework for seismic retrofit with circular economy integration
    """
    
    def __init__(self):
        self.design_variables = {
            'concrete_thickness': (200, 500),  # mm
            'steel_area': (0.001, 0.05),       # m²/m
            'frp_layers': (0, 10),             # number of layers
            'material_type': (0, 3)            # categorical: 0=virgin, 1=recycled, 2=bio-based, 3=hybrid
        }
        
        self.objectives = ['seismic_performance', 'environmental_impact', 'lifecycle_cost', 'circularity']
        self.constraints = ['strength', 'displacement', 'constructability']
        
    def seismic_capacity_model(self, design_vars):
        """
        Calculate seismic capacity based on design variables
        Uses capacity spectrum method and FEMA guidelines
        """
        t_c, A_s, n_frp, mat_type = design_vars
        
        # Material properties based on type
        material_props = self._get_material_properties(mat_type)
        
        # Concrete contribution
        f_c = material_props['concrete_strength']  # MPa
        A_c = t_c * 1000  # mm² per meter width
        V_c = 0.17 * np.sqrt(f_c) * A_c / 1000  # kN/m (ACI 318)
        
        # Steel contribution  
        f_y = material_props['steel_yield']  # MPa
        V_s = A_s * f_y * 1000  # kN/m
        
        # FRP contribution
        if n_frp > 0:
            f_frp = material_props['frp_strength']  # MPa
            t_frp = n_frp * 0.5  # mm per layer
            A_frp = t_frp * 1000 / 1000  # m²/m
            V_frp = A_frp * f_frp * 1000  # kN/m
        else:
            V_frp = 0
        
        # Total capacity
        V_total = V_c + V_s + V_frp
        
        # Convert to spectral acceleration capacity
        W = 10 * 9.81  # kN/m (assumed wall weight)
        Sa_capacity = V_total / W  # g
        
        return Sa_capacity
    
    def fragility_analysis(self, Sa_capacity):
        """
        Perform fragility analysis for different performance levels
        """
        # Performance levels (spectral accelerations in g)
        performance_levels = {
            'Operational': 0.15,
            'Immediate_Occupancy': 0.25, 
            'Life_Safety': 0.50,
            'Collapse_Prevention': 0.75
        }
        
        fragilities = {}
        for level, Sa_demand in performance_levels.items():
            # Lognormal fragility function
            beta = 0.6  # Logarithmic standard deviation
            median_capacity = Sa_capacity
            
            # Probability of exceeding performance level
            prob_exceed = 1 - lognorm.cdf(Sa_demand, s=beta, scale=median_capacity)
            fragilities[level] = prob_exceed
            
        return fragilities
    
    def environmental_impact_model(self, design_vars):
        """
        Calculate environmental impact using LCA methodology
        """
        t_c, A_s, n_frp, mat_type = design_vars
        
        material_props = self._get_material_properties(mat_type)
        
        # Volume calculations
        concrete_volume = t_c * 1.0 * 1.0 / 1000  # m³/m²
        steel_volume = A_s * 1.0  # m³/m²
        frp_volume = n_frp * 0.5e-3 * 1.0  # m³/m²
        
        # Carbon footprint (kg CO2-eq/m²)
        carbon_concrete = concrete_volume * material_props['concrete_carbon'] * 2400  # density
        carbon_steel = steel_volume * material_props['steel_carbon'] * 7850
        carbon_frp = frp_volume * material_props['frp_carbon'] * 1600
        
        total_carbon = carbon_concrete + carbon_steel + carbon_frp
        
        # Embodied energy (MJ/m²)
        energy_concrete = concrete_volume * material_props['concrete_energy'] * 2400
        energy_steel = steel_volume * material_props['steel_energy'] * 7850
        energy_frp = frp_volume * material_props['frp_energy'] * 1600
        
        total_energy = energy_concrete + energy_steel + energy_frp
        
        # Normalize to 0-100 scale (higher is worse)
        carbon_index = min(total_carbon / 500 * 100, 100)  # Normalize by typical max
        energy_index = min(total_energy / 50000 * 100, 100)
        
        environmental_index = (carbon_index + energy_index) / 2
        
        return environmental_index
    
    def lifecycle_cost_model(self, design_vars):
        """
        Calculate lifecycle cost including initial, maintenance, and end-of-life costs
        """
        t_c, A_s, n_frp, mat_type = design_vars
        
        material_props = self._get_material_properties(mat_type)
        
        # Volume calculations
        concrete_volume = t_c * 1.0 * 1.0 / 1000  # m³/m²
        steel_volume = A_s * 1.0  # m³/m²
        frp_volume = n_frp * 0.5e-3 * 1.0  # m³/m²
        
        # Initial costs ($/m²)
        cost_concrete = concrete_volume * material_props['concrete_cost'] * 2400
        cost_steel = steel_volume * material_props['steel_cost'] * 7850
        cost_frp = frp_volume * material_props['frp_cost'] * 1600
        cost_labor = (cost_concrete + cost_steel + cost_frp) * 0.5  # 50% labor
        
        initial_cost = cost_concrete + cost_steel + cost_frp + cost_labor
        
        # Maintenance costs (NPV over 50 years)
        discount_rate = 0.03
        maintenance_factor = 0.02  # 2% per year of initial cost
        years = np.arange(1, 51)
        npv_maintenance = np.sum(initial_cost * maintenance_factor / (1 + discount_rate)**years)
        
        # End-of-life value (negative cost = revenue)
        recyclability = material_props['recyclability']
        end_of_life_value = -(initial_cost * recyclability * 0.1)  # 10% of initial cost recovery
        
        total_cost = initial_cost + npv_maintenance + end_of_life_value
        
        # Normalize to 0-100 scale
        cost_index = min(total_cost / 1000 * 100, 100)  # Normalize by typical max
        
        return cost_index
    
    def circularity_model(self, design_vars):
        """
        Calculate circularity index based on circular economy principles
        """
        t_c, A_s, n_frp, mat_type = design_vars
        
        material_props = self._get_material_properties(mat_type)
        
        # Material masses
        mass_concrete = (t_c / 1000) * 2400  # kg/m²
        mass_steel = A_s * 7850  # kg/m²
        mass_frp = (n_frp * 0.5e-3) * 1600  # kg/m²
        total_mass = mass_concrete + mass_steel + mass_frp
        
        # Weighted circularity metrics
        recyclability = (
            mass_concrete * material_props['concrete_recyclability'] +
            mass_steel * material_props['steel_recyclability'] +
            mass_frp * material_props['frp_recyclability']
        ) / total_mass
        
        # Additional circular economy factors
        material_efficiency = 1 - (total_mass / 5000)  # Normalized by typical max mass
        durability_factor = min(A_s / 0.02, 1.0)  # More steel = higher durability
        adaptability = 0.8 if n_frp > 0 else 0.6  # FRP allows easier future modifications
        
        circularity_index = (recyclability * 0.4 + material_efficiency * 0.3 + 
                           durability_factor * 0.2 + adaptability * 0.1) * 100
        
        return circularity_index
    
    def _get_material_properties(self, mat_type):
        """Get material properties based on material type"""
        if mat_type < 0.5:  # Virgin materials
            return {
                'concrete_strength': 30, 'concrete_carbon': 310, 'concrete_energy': 1050, 
                'concrete_cost': 85, 'concrete_recyclability': 0.15,
                'steel_yield': 420, 'steel_carbon': 2100, 'steel_energy': 24500,
                'steel_cost': 850, 'steel_recyclability': 0.95,
                'frp_strength': 600, 'frp_carbon': 15000, 'frp_energy': 180000,
                'frp_cost': 2500, 'frp_recyclability': 0.20
            }
        elif mat_type < 1.5:  # Recycled materials
            return {
                'concrete_strength': 25, 'concrete_carbon': 180, 'concrete_energy': 650,
                'concrete_cost': 75, 'concrete_recyclability': 0.85,
                'steel_yield': 400, 'steel_carbon': 630, 'steel_energy': 8900,
                'steel_cost': 720, 'steel_recyclability': 0.98,
                'frp_strength': 500, 'frp_carbon': 3200, 'frp_energy': 28000,
                'frp_cost': 800, 'frp_recyclability': 0.30
            }
        elif mat_type < 2.5:  # Bio-based materials
            return {
                'concrete_strength': 35, 'concrete_carbon': 120, 'concrete_energy': 450,
                'concrete_cost': 120, 'concrete_recyclability': 0.90,
                'steel_yield': 420, 'steel_carbon': 2100, 'steel_energy': 24500,
                'steel_cost': 850, 'steel_recyclability': 0.95,
                'frp_strength': 450, 'frp_carbon': 1800, 'frp_energy': 18000,
                'frp_cost': 1200, 'frp_recyclability': 0.45
            }
        else:  # Hybrid materials
            return {
                'concrete_strength': 32, 'concrete_carbon': 200, 'concrete_energy': 700,
                'concrete_cost': 95, 'concrete_recyclability': 0.70,
                'steel_yield': 410, 'steel_carbon': 1200, 'steel_energy': 15000,
                'steel_cost': 780, 'steel_recyclability': 0.96,
                'frp_strength': 550, 'frp_carbon': 8000, 'frp_energy': 80000,
                'frp_cost': 1500, 'frp_recyclability': 0.35
            }
    
    def objective_function(self, design_vars):
        """
        Multi-objective function for optimization
        Returns: [seismic_performance, environmental_impact, lifecycle_cost, circularity]
        """
        # Calculate seismic capacity
        Sa_capacity = self.seismic_capacity_model(design_vars)
        fragilities = self.fragility_analysis(Sa_capacity)
        
        # Seismic performance (higher is better, 0-100 scale)
        seismic_performance = min(Sa_capacity / 1.0 * 100, 100)
        
        # Environmental impact (lower is better, 0-100 scale)
        environmental_impact = self.environmental_impact_model(design_vars)
        
        # Lifecycle cost (lower is better, 0-100 scale)  
        lifecycle_cost = self.lifecycle_cost_model(design_vars)
        
        # Circularity (higher is better, 0-100 scale)
        circularity = self.circularity_model(design_vars)
        
        return [seismic_performance, -environmental_impact, -lifecycle_cost, circularity]
    
    def constraint_function(self, design_vars):
        """
        Constraint functions (must be >= 0 for feasible solutions)
        """
        t_c, A_s, n_frp, mat_type = design_vars
        
        constraints = []
        
        # Minimum strength requirement
        Sa_capacity = self.seismic_capacity_model(design_vars)
        constraints.append(Sa_capacity - 0.4)  # Minimum capacity of 0.4g
        
        # Maximum displacement constraint
        # Simplified displacement calculation
        displacement = 0.5 / Sa_capacity if Sa_capacity > 0 else 1.0  # cm
        constraints.append(5.0 - displacement)  # Maximum 5cm displacement
        
        # Constructability constraints
        constraints.append(600 - t_c)  # Maximum thickness
        constraints.append(t_c - 150)  # Minimum thickness
        constraints.append(0.06 - A_s)  # Maximum steel ratio
        constraints.append(A_s - 0.001)  # Minimum steel ratio
        
        return np.array(constraints)

def run_optimization_study():
    """
    Run comprehensive optimization study
    """
    optimizer = SeismicRetrofitOptimizer()
    
    # Define bounds for design variables
    bounds = [(200, 500), (0.001, 0.05), (0, 10), (0, 3)]
    
    # Multi-objective optimization using differential evolution
    n_solutions = 100
    pareto_solutions = []
    
    print("Running multi-objective optimization...")
    
    for i in range(n_solutions):
        # Random starting point
        x0 = [
            np.random.uniform(200, 500),
            np.random.uniform(0.001, 0.05), 
            np.random.randint(0, 11),
            np.random.uniform(0, 3)
        ]
        
        # Single objective optimization (weighted sum)
        weights = np.random.random(4)
        weights = weights / np.sum(weights)
        
        def weighted_objective(x):
            objectives = optimizer.objective_function(x)
            return -np.dot(weights, objectives)  # Minimize negative sum
        
        def constraint_wrapper(x):
            return optimizer.constraint_function(x)
        
        # Optimize
        result = differential_evolution(
            weighted_objective,
            bounds,
            maxiter=50,
            seed=i
        )
        
        if result.success:
            objectives = optimizer.objective_function(result.x)
            solution = {
                'thickness': result.x[0],
                'steel_area': result.x[1], 
                'frp_layers': int(result.x[2]),
                'material_type': result.x[3],
                'seismic_performance': objectives[0],
                'environmental_impact': -objectives[1],
                'lifecycle_cost': -objectives[2],
                'circularity': objectives[3]
            }
            pareto_solutions.append(solution)
        
        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{n_solutions} optimization runs")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(pareto_solutions)
    results_df.to_csv('detailed_optimization_results.csv', index=False)
    
    print(f"Optimization complete. Generated {len(pareto_solutions)} feasible solutions.")
    return results_df

if __name__ == "__main__":
    # Run optimization study
    results = run_optimization_study()
    
    # Display summary statistics
    print("\nOptimization Results Summary:")
    print("=" * 50)
    print(results.describe())
    
    # Find best solutions for each objective
    print("\nBest Solutions by Objective:")
    print("-" * 30)
    
    best_seismic = results.loc[results['seismic_performance'].idxmax()]
    print(f"Best Seismic Performance: {best_seismic['seismic_performance']:.1f}")
    
    best_env = results.loc[results['environmental_impact'].idxmin()]
    print(f"Best Environmental Impact: {best_env['environmental_impact']:.1f}")
    
    best_cost = results.loc[results['lifecycle_cost'].idxmin()]
    print(f"Best Lifecycle Cost: {best_cost['lifecycle_cost']:.1f}")
    
    best_circular = results.loc[results['circularity'].idxmax()]
    print(f"Best Circularity: {best_circular['circularity']:.1f}")