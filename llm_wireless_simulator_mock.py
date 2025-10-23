"""
Mock LLM-Driven Wireless Network Simulation
Demonstrates LLM integration without requiring API key
Uses realistic example responses from Claude
"""

import json
from typing import Dict, List, Tuple
from wireless_simulator_extended import (
    ExtendedWirelessSimulator, MobilityConfig, MobilityModel,
    FadingConfig, FadingModel
)
from wireless_simulator import NodeConfig, TrafficConfig

# Pre-generated Claude responses (as examples of what LLM would generate)
MOCK_SCENARIOS = {
    "simple_line": {
        "description": "Simple line topology with Rayleigh fading",
        "num_nodes": 5,
        "area_width": 100,
        "area_height": 100,
        "duration": 10.0,
        "mobility": {"enabled": False, "model": "static", "speed": 0.0},
        "fading": {"enabled": True, "model": "rayleigh", "severity": "moderate"},
        "mac": {"enabled": False, "type": "none"},
        "traffic": [{"source": 0, "dest": 4, "rate": 20}],
        "topology": "line"
    },
    
    "dense_urban": {
        "description": "Dense urban network with 10 mobile nodes, Rayleigh fading, and backoff MAC",
        "num_nodes": 10,
        "area_width": 50,
        "area_height": 50,
        "duration": 15.0,
        "mobility": {"enabled": True, "model": "random_walk", "speed": 2.0},
        "fading": {"enabled": True, "model": "rayleigh", "severity": "severe"},
        "mac": {"enabled": True, "type": "exponential_backoff"},
        "traffic": [
            {"source": 0, "dest": 5, "rate": 30},
            {"source": 2, "dest": 7, "rate": 25},
            {"source": 4, "dest": 9, "rate": 20}
        ],
        "topology": "random"
    },
    
    "star_network": {
        "description": "Star topology with central hub and 8 clients",
        "num_nodes": 9,
        "area_width": 100,
        "area_height": 100,
        "duration": 10.0,
        "mobility": {"enabled": False, "model": "static", "speed": 0.0},
        "fading": {"enabled": True, "model": "shadowing", "severity": "moderate"},
        "mac": {"enabled": False, "type": "none"},
        "traffic": [
            {"source": i, "dest": 0, "rate": 15} for i in range(1, 9)
        ],
        "topology": "star"
    }
}

class MockLLMWirelessSimulator:
    """Mock LLM-driven simulator (demonstrates concept)"""
    
    def __init__(self):
        self.conversation_history = []
        self.last_results = None
        self.current_scenario = None
    
    def _generate_mock_response(self, user_message: str) -> str:
        """Generate realistic mock responses"""
        if "simple" in user_message.lower() and "line" in user_message.lower():
            scenario_json = json.dumps(MOCK_SCENARIOS["simple_line"], indent=2)
            return f"""Based on your description, here's an ideal scenario:

This will create a simple wireless network with 5 nodes arranged in a line.
We'll use Rayleigh fading to simulate realistic indoor propagation (NLOS).

{scenario_json}

This scenario tests basic path loss and fading effects in a simple topology."""
        
        elif "dense" in user_message.lower() and ("urban" in user_message.lower() or "mobile" in user_message.lower()):
            scenario_json = json.dumps(MOCK_SCENARIOS["dense_urban"], indent=2)
            return f"""Great! A dense urban network is challenging. Here's what I recommend:

This creates a 50x50m area with 10 mobile nodes using random walk mobility.
Rayleigh fading simulates NLOS propagation (typical indoor/urban).
Exponential backoff MAC is crucial for managing collisions in this dense scenario.

{scenario_json}

This stress-tests the network with multiple simultaneous traffic flows."""
        
        elif "star" in user_message.lower():
            scenario_json = json.dumps(MOCK_SCENARIOS["star_network"], indent=2)
            return f"""Perfect! A star topology is typical for WiFi networks. Here's my recommendation:

Central hub at (50, 50) with 8 client nodes arranged in a circle.
Log-normal shadowing better represents large-scale fading effects.
All clients transmit to the central hub.

{scenario_json}

This tests hub congestion and client coordination."""
        
        else:
            return """I'd be happy to help design a wireless network scenario. 

Could you describe what you'd like to test? For example:
- Number of nodes and their arrangement (line, star, grid, random)
- Movement patterns (static, random walk, waypoint-based)
- Channel effects (fading model, severity)
- Traffic patterns (light, moderate, heavy)
- Protocol features (backoff MAC, rate adaptation)

Once you provide details, I'll generate a structured simulation scenario."""
    
    def generate_scenario(self, description: str) -> Dict:
        """Generate scenario from natural language description"""
        print(f"\n📝 Generating scenario...")
        print(f"   User request: {description}")
        
        # Get mock response
        response = self._generate_mock_response(description)
        print(f"\n🤖 Claude's Response:")
        print(response)
        
        # Parse JSON from response
        scenario = self.parse_scenario_from_response(response)
        self.current_scenario = scenario
        return scenario
    
    def parse_scenario_from_response(self, response: str) -> Dict:
        """Extract JSON scenario from response"""
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response[start:end]
        return json.loads(json_str)
    
    def run_scenario(self, scenario: Dict) -> Dict:
        """Run simulation with scenario"""
        print(f"\n▶️  Running simulation: {scenario.get('description', 'Custom scenario')}")
        
        duration = scenario.get('duration', 10.0)
        
        # Configure fading
        fading_config = None
        if scenario.get('fading', {}).get('enabled'):
            fading_model = {
                'rayleigh': FadingModel.RAYLEIGH,
                'rician': FadingModel.RICIAN,
                'shadowing': FadingModel.SHADOWING,
            }.get(scenario['fading']['model'], FadingModel.RAYLEIGH)
            
            fading_config = FadingConfig(model=fading_model)
        
        # Create simulator
        sim = ExtendedWirelessSimulator(
            duration=duration,
            fading_config=fading_config,
            use_mobility=scenario.get('mobility', {}).get('enabled', False),
            use_backoff_mac=scenario.get('mac', {}).get('enabled', False)
        )
        
        # Add nodes
        topology = scenario.get('topology', 'random')
        num_nodes = scenario.get('num_nodes', 5)
        
        positions = self._generate_positions(
            topology, num_nodes,
            scenario.get('area_width', 100),
            scenario.get('area_height', 100)
        )
        
        for i, (x, y) in enumerate(positions):
            config = NodeConfig(node_id=i, x=float(x), y=float(y))
            
            if scenario.get('mobility', {}).get('enabled'):
                mobility_model = {
                    'random_walk': MobilityModel.RANDOM_WALK,
                    'random_waypoint': MobilityModel.RANDOM_WAYPOINT,
                    'circular': MobilityModel.CIRCULAR,
                    'manhattan': MobilityModel.MANHATTAN,
                }.get(scenario['mobility']['model'], MobilityModel.RANDOM_WALK)
                
                mobility_config = MobilityConfig(
                    model=mobility_model,
                    speed=scenario['mobility'].get('speed', 2.0),
                    area_width=scenario.get('area_width', 100),
                    area_height=scenario.get('area_height', 100)
                )
                sim.add_mobile_node(config, mobility_config)
            elif scenario.get('mac', {}).get('enabled'):
                sim.add_backoff_node(config)
            else:
                sim.add_node(config)
        
        # Add traffic
        for traffic in scenario.get('traffic', []):
            sim.add_traffic_flow(TrafficConfig(
                source_id=traffic['source'],
                dest_id=traffic['dest'],
                inter_arrival_time=1.0 / traffic.get('rate', 10.0)
            ))
        
        # Run simulation
        metrics = sim.run()
        
        print(f"\n📊 Results:")
        print(f"   PDR: {metrics['pdr']:.2%}")
        print(f"   Collision Rate: {metrics['collision_rate']:.2%}")
        print(f"   Packets Sent: {metrics['packets_sent']}")
        print(f"   Packets Received: {metrics['packets_received']}")
        print(f"   Avg Delay: {metrics.get('avg_delay', 0)*1000:.2f} ms")
        
        self.last_results = metrics
        return metrics
    
    def _generate_positions(self, topology: str, num_nodes: int, 
                           width: float, height: float) -> List[Tuple[float, float]]:
        """Generate positions based on topology"""
        import math
        
        if topology == 'line':
            spacing = width / (num_nodes + 1)
            return [(spacing * (i + 1), height / 2) for i in range(num_nodes)]
        
        elif topology == 'star':
            center = (width / 2, height / 2)
            radius = min(width, height) / 3
            positions = [center]
            for i in range(1, num_nodes):
                angle = 2 * math.pi * i / (num_nodes - 1)
                x = center[0] + radius * math.cos(angle)
                y = center[1] + radius * math.sin(angle)
                positions.append((x, y))
            return positions
        
        elif topology == 'grid':
            cols = int(math.sqrt(num_nodes))
            rows = (num_nodes + cols - 1) // cols
            spacing_x = width / (cols + 1)
            spacing_y = height / (rows + 1)
            positions = []
            for i in range(num_nodes):
                row = i // cols
                col = i % cols
                x = spacing_x * (col + 1)
                y = spacing_y * (row + 1)
                positions.append((x, y))
            return positions
        
        else:  # random
            import random
            return [(random.uniform(0, width), random.uniform(0, height)) 
                   for _ in range(num_nodes)]
    
    def analyze_results(self) -> str:
        """Analyze last simulation results"""
        if self.last_results is None:
            return "No results to analyze."
        
        pdr = self.last_results['pdr']
        collision_rate = self.last_results['collision_rate']
        
        print(f"\n🔍 Analyzing results...")
        
        # Generate mock analysis
        if pdr > 0.9:
            analysis = f"""Excellent network performance! PDR of {pdr:.1%} indicates:

1. **Strong Connectivity**: Nodes can reliably reach each other
2. **Low Collision Rate**: {collision_rate:.1%} collision rate is acceptable
3. **Implications**: 
   - Topology and spacing are well-designed
   - Channel conditions (fading) are manageable
   - MAC protocol (if used) is effective

Recommendations:
- This configuration is suitable for real-world deployment
- Consider testing with increased traffic load
- Evaluate mobility impact on sustained performance"""
        
        elif pdr > 0.5:
            analysis = f"""Moderate network performance. PDR of {pdr:.1%} suggests:

1. **Manageable Conditions**: Network functions but with limitations
2. **Collision Issues**: {collision_rate:.1%} collision rate indicates congestion
3. **Bottlenecks**:
   - Some nodes struggle to communicate
   - MAC protocol needs tuning
   - Consider reducing node density or traffic

Recommendations:
- Test with exponential backoff MAC to reduce collisions
- Consider topology changes (spread nodes further apart)
- Implement rate adaptation for unreliable links"""
        
        else:
            analysis = f"""Poor network performance. PDR of {pdr:.1%} indicates:

1. **Severe Issues**: Network is severely congested
2. **Collision Cascade**: {collision_rate:.1%} collision rate shows protocol breakdown
3. **Root Causes**:
   - Likely: Too many nodes in small area
   - Possibly: Severe fading without backoff MAC
   - Maybe: Traffic load exceeds network capacity

Recommendations:
- Implement exponential backoff MAC immediately
- Reduce node density or increase transmission power
- Consider mesh routing instead of single-hop
- Test individual features in isolation"""
        
        print(f"\n🤖 Analysis:\n{analysis}")
        return analysis
    
    def suggest_experiments(self) -> str:
        """Suggest follow-up experiments"""
        if self.last_results is None:
            return "Run a simulation first."
        
        print(f"\n💡 Suggesting follow-up experiments...")
        
        pdr = self.last_results['pdr']
        
        suggestions = """Based on the current results, here are 3 interesting follow-up experiments:

**Experiment 1: Vary Node Density**
- Currently testing: Current number of nodes
- Vary: 3, 5, 7, 10, 15 nodes in same area
- Why: Understand how density affects performance
- Expected: PDR degrades with more nodes due to collisions
- Publication: "Density Limits in Wireless Networks"

**Experiment 2: Compare MAC Protocols**
- Currently testing: Current MAC (if any)
- Test: Without backoff vs with exponential backoff
- Why: Quantify MAC protocol effectiveness
- Expected: Backoff reduces collision rate by 30-50%
- Publication: "MAC Protocol Comparison in Fading Channels"

**Experiment 3: Test Mobility Models**
- Currently testing: Static or current model
- Test: Random Walk, Waypoint, Circular, Manhattan
- Why: Understand how movement patterns affect connectivity
- Expected: Different mobility models have different collision dynamics
- Publication: "Mobility Impact on Wireless Network Performance"
"""
        
        print(suggestions)
        return suggestions


def run_mock_demonstrations():
    """Run mock LLM scenarios"""
    sim = MockLLMWirelessSimulator()
    
    # Scenario 1: Simple line topology
    print("\n" + "="*80)
    print("SCENARIO 1: Simple Line Topology with Fading")
    print("="*80)
    
    scenario1 = sim.generate_scenario(
        "Create a simple wireless network with 5 nodes in a line topology. "
        "Use Rayleigh fading to simulate realistic indoor propagation. "
        "Run for 10 seconds with moderate traffic."
    )
    
    results1 = sim.run_scenario(scenario1)
    analysis1 = sim.analyze_results()
    
    # Scenario 2: Dense urban
    print("\n" + "="*80)
    print("SCENARIO 2: Dense Urban Network with Mobility & Backoff")
    print("="*80)
    
    scenario2 = sim.generate_scenario(
        "Design a dense urban wireless network with 10 mobile nodes. "
        "Nodes move randomly. Include Rayleigh fading and exponential backoff MAC. "
        "Use heavy traffic to stress-test."
    )
    
    results2 = sim.run_scenario(scenario2)
    analysis2 = sim.analyze_results()
    
    # Scenario 3: Star topology
    print("\n" + "="*80)
    print("SCENARIO 3: Star Topology Network")
    print("="*80)
    
    scenario3 = sim.generate_scenario(
        "Create a star topology network with central hub and 8 clients. "
        "Include fading. Focus on medium traffic load."
    )
    
    results3 = sim.run_scenario(scenario3)
    analysis3 = sim.analyze_results()
    
    # Get suggestions
    print("\n" + "="*80)
    print("EXPERIMENT SUGGESTIONS FROM CLAUDE")
    print("="*80)
    
    suggestions = sim.suggest_experiments()
    
    print("\n" + "="*80)
    print("LLM-DRIVEN SCENARIOS COMPLETE")
    print("="*80)
    print("\n✅ Demonstrated:")
    print("  • Claude generates realistic network scenarios")
    print("  • Scenarios run in simulator")
    print("  • Results are analyzed by Claude")
    print("  • Suggestions for follow-up research")
    print("\n💡 This workflow enables AI-assisted network research!")


if __name__ == "__main__":
    run_mock_demonstrations()
