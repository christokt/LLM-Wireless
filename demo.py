import json
from llm_interface import ScenarioValidator, SimulatorRunner, ResultsInterpreter

def create_demo_scenarios():
    """Create example scenarios for demonstration"""
    
    # Scenario 1: Simple line topology
    scenario_line = {
        "name": "Simple Line Network",
        "description": "4 nodes in a line, 10m apart",
        "duration": 5.0,
        "num_nodes": 4,
        "node_configs": [
            {"x": 0.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 10.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 20.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 30.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0}
        ],
        "traffic_flows": [
            {"source_id": 0, "dest_id": 3, "inter_arrival_time": 0.01, "packet_size": 1500},
            {"source_id": 1, "dest_id": 2, "inter_arrival_time": 0.01, "packet_size": 1500}
        ]
    }
    
    # Scenario 2: Dense cluster (high interference)
    scenario_dense = {
        "name": "Dense Cluster Network",
        "description": "8 nodes in close proximity",
        "duration": 5.0,
        "num_nodes": 8,
        "node_configs": [
            {"x": 0.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 5.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 0.0, "y": 5.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 5.0, "y": 5.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 2.5, "y": 2.5, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 7.5, "y": 2.5, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 2.5, "y": 7.5, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 7.5, "y": 7.5, "tx_power": 20.0, "carrier_sense_threshold": -82.0}
        ],
        "traffic_flows": [
            {"source_id": 0, "dest_id": 4, "inter_arrival_time": 0.01, "packet_size": 1500},
            {"source_id": 1, "dest_id": 5, "inter_arrival_time": 0.01, "packet_size": 1500},
            {"source_id": 2, "dest_id": 6, "inter_arrival_time": 0.01, "packet_size": 1500},
            {"source_id": 3, "dest_id": 7, "inter_arrival_time": 0.01, "packet_size": 1500}
        ]
    }
    
    # Scenario 3: Sparse network (low interference)
    scenario_sparse = {
        "name": "Sparse Network",
        "description": "6 nodes spread far apart",
        "duration": 5.0,
        "num_nodes": 6,
        "node_configs": [
            {"x": 0.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 50.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 100.0, "y": 0.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 25.0, "y": 50.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 75.0, "y": 50.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0},
            {"x": 50.0, "y": 100.0, "tx_power": 20.0, "carrier_sense_threshold": -82.0}
        ],
        "traffic_flows": [
            {"source_id": 0, "dest_id": 1, "inter_arrival_time": 0.01, "packet_size": 1500},
            {"source_id": 1, "dest_id": 2, "inter_arrival_time": 0.01, "packet_size": 1500},
            {"source_id": 3, "dest_id": 5, "inter_arrival_time": 0.01, "packet_size": 1500}
        ]
    }
    
    return [scenario_line, scenario_dense, scenario_sparse]

def run_demo():
    """Run demonstration of the LLM interface pipeline"""
    
    print("=" * 80)
    print("LLM-Simulator Interface Demonstration")
    print("=" * 80)
    
    validator = ScenarioValidator()
    runner = SimulatorRunner()
    interpreter = ResultsInterpreter()
    
    scenarios = create_demo_scenarios()
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario['name']}")
        print(f"{'='*80}")
        
        # Step 1: Validate
        print(f"\n[Step 1] Validating scenario...")
        is_valid, error_msg = validator.validate(scenario)
        
        if not is_valid:
            print(f"❌ Validation failed: {error_msg}")
            continue
        
        print(f"✓ Scenario validated successfully")
        
        # Step 2: Run simulation
        print(f"\n[Step 2] Running simulation...")
        result = runner.run_scenario(scenario)
        print(f"✓ Simulation completed")
        
        # Step 3: Interpret results
        print(f"\n[Step 3] Interpreting results...")
        interpretation = interpreter.interpret(result)
        print(interpretation)
    
    print("\n" + "=" * 80)
    print("Demonstration Complete")
    print("=" * 80)

def demonstrate_llm_interface():
    """Show what the LLM interface does"""
    
    print("\n" + "=" * 80)
    print("LLM Interface Capabilities")
    print("=" * 80)
    
    print("""
The LLM interface layer provides the following capabilities:

1. NATURAL LANGUAGE INPUT
   - Users describe wireless scenarios in plain English
   - Example: "Create a WiFi network with 5 nodes arranged in a circle"
   
2. LLM-BASED CODE GENERATION
   - Claude generates valid JSON scenario specifications
   - Specifications include node positions, traffic flows, parameters
   
3. VALIDATION
   - Input validation ensures specifications are valid
   - Prevents invalid configurations from reaching the simulator
   - Provides clear error messages for invalid specs
   
4. SIMULATION
   - Validated scenarios are passed to the simulator
   - Simulator runs discrete-event wireless network simulation
   - Collects performance metrics
   
5. RESULTS INTERPRETATION
   - Simulator output is analyzed and summarized
   - Key insights are extracted (collision rates, delays, PDR)
   - Human-readable report is generated

RESEARCH CONTRIBUTIONS:
- Novel interface design for LLM-simulator integration
- Schema validation approach for reliable LLM outputs
- Methodology for structuring simulator results for LLM analysis
- Empirical study on LLM reliability for network scenario generation
""")

if __name__ == "__main__":
    # Show capabilities
    demonstrate_llm_interface()
    
    # Run demonstration
    run_demo()
