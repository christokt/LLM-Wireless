import json
import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import anthropic
from wireless_simulator import WirelessSimulator, NodeConfig, TrafficConfig

@dataclass
class ScenarioSpec:
    """Structured scenario specification"""
    name: str
    description: str
    duration: float
    num_nodes: int
    node_configs: List[Dict[str, float]]
    traffic_flows: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        return asdict(self)

class ScenarioValidator:
    """Validates scenario specifications"""
    
    @staticmethod
    def validate(spec: Dict) -> Tuple[bool, str]:
        """
        Validate a scenario specification
        
        Returns:
            (is_valid, error_message)
        """
        required_fields = ['name', 'duration', 'num_nodes', 'node_configs', 'traffic_flows']
        for field in required_fields:
            if field not in spec:
                return False, f"Missing required field: {field}"
        
        # Validate duration
        if not isinstance(spec['duration'], (int, float)) or spec['duration'] <= 0:
            return False, "Duration must be positive number"
        
        # Validate num_nodes
        if not isinstance(spec['num_nodes'], int) or spec['num_nodes'] < 2:
            return False, "num_nodes must be integer >= 2"
        
        # Validate node_configs
        if len(spec['node_configs']) != spec['num_nodes']:
            return False, f"Number of node configs ({len(spec['node_configs'])}) must match num_nodes ({spec['num_nodes']})"
        
        for i, node_config in enumerate(spec['node_configs']):
            if 'x' not in node_config or 'y' not in node_config:
                return False, f"Node {i} missing x,y coordinates"
            if not isinstance(node_config['x'], (int, float)) or not isinstance(node_config['y'], (int, float)):
                return False, f"Node {i} coordinates must be numbers"
        
        # Validate traffic flows
        if not isinstance(spec['traffic_flows'], list):
            return False, "traffic_flows must be a list"
        
        for i, flow in enumerate(spec['traffic_flows']):
            if 'source_id' not in flow or 'dest_id' not in flow:
                return False, f"Traffic flow {i} missing source_id or dest_id"
            source_id = flow['source_id']
            dest_id = flow['dest_id']
            
            if not isinstance(source_id, int) or not isinstance(dest_id, int):
                return False, f"Traffic flow {i} IDs must be integers"
            
            if source_id == dest_id:
                return False, f"Traffic flow {i}: source and dest cannot be same"
            
            if source_id < 0 or source_id >= spec['num_nodes']:
                return False, f"Traffic flow {i}: source_id {source_id} out of range"
            
            if dest_id < 0 or dest_id >= spec['num_nodes']:
                return False, f"Traffic flow {i}: dest_id {dest_id} out of range"
            
            if 'inter_arrival_time' in flow:
                if not isinstance(flow['inter_arrival_time'], (int, float)) or flow['inter_arrival_time'] <= 0:
                    return False, f"Traffic flow {i}: inter_arrival_time must be positive"
        
        return True, ""

class SimulatorRunner:
    """Runs simulator with validated specifications"""
    
    @staticmethod
    def run_scenario(spec: Dict) -> Dict:
        """
        Run simulator with given scenario specification
        
        Returns:
            Dictionary with results and metrics
        """
        # Create simulator
        sim = WirelessSimulator(duration=spec['duration'], time_step=0.001)
        
        # Add nodes
        for i, node_config in enumerate(spec['node_configs']):
            config = NodeConfig(
                node_id=i,
                x=node_config.get('x', 0.0),
                y=node_config.get('y', 0.0),
                tx_power=node_config.get('tx_power', 20.0),
                carrier_sense_threshold=node_config.get('carrier_sense_threshold', -82.0)
            )
            sim.add_node(config)
        
        # Add traffic flows
        for flow in spec['traffic_flows']:
            config = TrafficConfig(
                source_id=flow['source_id'],
                dest_id=flow['dest_id'],
                packet_size=flow.get('packet_size', 1500),
                inter_arrival_time=flow.get('inter_arrival_time', 0.01)
            )
            sim.add_traffic_flow(config)
        
        # Run simulation
        metrics = sim.run()
        
        return {
            'scenario_name': spec.get('name', 'Unnamed'),
            'metrics': metrics,
            'spec': spec
        }

class ResultsInterpreter:
    """Interprets simulator results into human-readable insights"""
    
    @staticmethod
    def interpret(result: Dict) -> str:
        """
        Generate human-readable interpretation of simulation results
        
        Returns:
            String with formatted results and insights
        """
        metrics = result['metrics']
        spec = result['spec']
        
        # Calculate statistics
        pdr = metrics['pdr']
        collision_rate = metrics['collision_rate']
        avg_delay = metrics['avg_delay']
        
        # Determine quality
        if pdr > 0.95:
            pdr_quality = "excellent"
        elif pdr > 0.80:
            pdr_quality = "good"
        elif pdr > 0.50:
            pdr_quality = "moderate"
        else:
            pdr_quality = "poor"
        
        # Build interpretation
        interpretation = f"""
=== Simulation Results: {result['scenario_name']} ===

Scenario Summary:
  - Number of nodes: {spec['num_nodes']}
  - Simulation duration: {spec['duration']} seconds
  - Number of traffic flows: {len(spec['traffic_flows'])}

Performance Metrics:
  - Packet Delivery Ratio: {pdr:.2%} ({pdr_quality})
  - Collision Rate: {collision_rate:.2%}
  - Average Delay: {avg_delay*1000:.3f} ms
  - Total packets sent: {metrics['packets_sent']}
  - Total packets received: {metrics['packets_received']}
  - Total packets collided: {metrics['packets_collided']}

Per-Node Breakdown:
"""
        for node_id, node_metrics in metrics['per_node'].items():
            sent = node_metrics['sent']
            received = node_metrics['received']
            node_pdr = received / sent if sent > 0 else 0
            interpretation += f"  - Node {node_id}: sent={sent}, received={received}, PDR={node_pdr:.2%}\n"
        
        # Add insights
        interpretation += "\nKey Insights:\n"
        if collision_rate > 0.3:
            interpretation += f"  - High collision rate ({collision_rate:.2%}) detected. Consider reducing traffic intensity or increasing node distances.\n"
        if pdr < 0.5:
            interpretation += f"  - Low packet delivery ratio ({pdr:.2%}). Channel conditions may be poor or nodes too far apart.\n"
        if avg_delay > 0.1:
            interpretation += f"  - Significant average delay ({avg_delay*1000:.1f} ms). Network congestion or high collision rates may be present.\n"
        
        return interpretation

class LLMScenarioGenerator:
    """Uses Claude to generate scenario specifications from natural language"""
    
    def __init__(self):
        self.client = anthropic.Anthropic()
    
    def generate_scenario(self, description: str) -> Tuple[Dict, str]:
        """
        Generate scenario specification from natural language description
        
        Args:
            description: Natural language description of desired scenario
        
        Returns:
            (scenario_dict, reasoning_text)
        """
        prompt = f"""You are a wireless network scenario designer. Based on the following natural language description, 
generate a JSON specification for a wireless network simulation scenario.

The JSON must follow this exact structure:
{{
    "name": "scenario name",
    "description": "scenario description",
    "duration": <float: simulation duration in seconds>,
    "num_nodes": <int: number of nodes>,
    "node_configs": [
        {{"x": <float>, "y": <float>, "tx_power": <float (dBm)>, "carrier_sense_threshold": <float (dBm)>}},
        ...
    ],
    "traffic_flows": [
        {{"source_id": <int>, "dest_id": <int>, "inter_arrival_time": <float>, "packet_size": <int>}},
        ...
    ]
}}

Important guidelines:
- Duration should typically be 5-60 seconds
- Position nodes on a 2D plane (x,y coordinates in meters)
- tx_power is typically 20 dBm for WiFi
- carrier_sense_threshold is typically -82 dBm
- inter_arrival_time should be small (0.001-0.1 seconds)
- packet_size is typically 1500 bytes
- Create realistic scenarios with 2-50 nodes
- Ensure traffic flows connect different nodes

User's description: {description}

Generate only the JSON object, no other text before or after. Make sure the JSON is valid and complete."""

        message = self.client.messages.create(
            model="claude-opus-4-1-20250805",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text
        
        # Try to extract JSON from response
        try:
            # Look for JSON in code blocks first
            json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response_text
            
            scenario_dict = json.loads(json_str)
            return scenario_dict, response_text
        except json.JSONDecodeError as e:
            return None, f"Failed to parse JSON: {e}\nResponse: {response_text}"

class SimulationPipeline:
    """End-to-end pipeline: natural language -> simulation -> results"""
    
    def __init__(self):
        self.llm_generator = LLMScenarioGenerator()
        self.validator = ScenarioValidator()
        self.runner = SimulatorRunner()
        self.interpreter = ResultsInterpreter()
    
    def run_from_description(self, description: str) -> Dict:
        """
        Run complete pipeline from natural language description
        
        Returns:
            Dictionary with scenario, validation result, simulation result, and interpretation
        """
        result = {
            'description': description,
            'scenario': None,
            'llm_reasoning': None,
            'validation': None,
            'simulation_result': None,
            'interpretation': None,
            'error': None
        }
        
        # Step 1: Generate scenario from LLM
        print(f"\n[Step 1] Generating scenario from description...")
        scenario, reasoning = self.llm_generator.generate_scenario(description)
        result['llm_reasoning'] = reasoning
        
        if scenario is None:
            result['error'] = reasoning
            return result
        
        result['scenario'] = scenario
        print(f"✓ Scenario generated: {scenario.get('name', 'Unnamed')}")
        
        # Step 2: Validate scenario
        print(f"\n[Step 2] Validating scenario...")
        is_valid, error_msg = self.validator.validate(scenario)
        result['validation'] = {'is_valid': is_valid, 'error': error_msg}
        
        if not is_valid:
            result['error'] = error_msg
            return result
        
        print(f"✓ Scenario validated successfully")
        
        # Step 3: Run simulation
        print(f"\n[Step 3] Running simulation...")
        sim_result = self.runner.run_scenario(scenario)
        result['simulation_result'] = sim_result
        print(f"✓ Simulation completed")
        
        # Step 4: Interpret results
        print(f"\n[Step 4] Interpreting results...")
        interpretation = self.interpreter.interpret(sim_result)
        result['interpretation'] = interpretation
        print(f"✓ Results interpreted")
        
        return result


if __name__ == "__main__":
    # Example usage
    pipeline = SimulationPipeline()
    
    # Test scenario description
    description = """
    Simulate a small WiFi office network with 4 nodes arranged in a line 10 meters apart.
    Node 0 should send traffic to Node 3, and Node 1 should send traffic to Node 2.
    Run for 5 seconds with moderate traffic intensity.
    """
    
    result = pipeline.run_from_description(description)
    
    if result['error']:
        print(f"\n❌ Error: {result['error']}")
    else:
        print(result['interpretation'])
