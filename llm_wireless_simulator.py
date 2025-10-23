"""
LLM-Driven Wireless Network Simulation
Uses Claude to generate scenarios, run simulations, and interpret results
"""

import json
import os
from typing import Dict, List, Tuple
from anthropic import Anthropic
from wireless_simulator_extended import (
    ExtendedWirelessSimulator, MobilityConfig, MobilityModel,
    FadingConfig, FadingModel, RateAdaptationAlgorithm
)
from wireless_simulator import NodeConfig, TrafficConfig

# Initialize Anthropic client
client = Anthropic()

# System prompt for Claude
SYSTEM_PROMPT = """You are an expert wireless network researcher helping design network simulation scenarios.

Your role is to:
1. Understand natural language descriptions of wireless networks
2. Convert them into structured simulation scenarios
3. Analyze simulation results and provide insights

SCENARIO FORMAT (return as JSON):
{
    "description": "What this scenario tests",
    "num_nodes": number,
    "area_width": meters,
    "area_height": meters,
    "duration": seconds,
    "mobility": {
        "enabled": true/false,
        "model": "random_walk|random_waypoint|circular|manhattan",
        "speed": m/s
    },
    "fading": {
        "enabled": true/false,
        "model": "none|rayleigh|rician|shadowing",
        "severity": "light|moderate|severe"
    },
    "mac": {
        "enabled": true/false,
        "type": "exponential_backoff"
    },
    "traffic": [
        {
            "source": node_id,
            "dest": node_id,
            "rate": packets_per_second
        }
    ],
    "topology": "line|star|grid|random"
}

Be creative with scenarios but ensure they're realistic and testable.
For topology, suggest positions that make sense for the topology type."""

class LLMWirelessSimulator:
    """LLM-driven wireless network simulator"""
    
    def __init__(self):
        self.conversation_history = []
        self.last_results = None
    
    def _chat(self, user_message: str) -> str:
        """Send message to Claude and get response"""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=self.conversation_history
        )
        
        assistant_message = response.content[0].text
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message
    
    def parse_scenario_from_response(self, response: str) -> Dict:
        """Extract JSON scenario from Claude's response"""
        # Find JSON in response
        start = response.find('{')
        end = response.rfind('}') + 1
        
        if start == -1 or end == 0:
            raise ValueError("No JSON found in response")
        
        json_str = response[start:end]
        return json.loads(json_str)
    
    def generate_scenario(self, description: str) -> Dict:
        """Ask Claude to generate a scenario"""
        print(f"\n📝 Generating scenario for: {description}")
        
        prompt = f"""Generate a wireless network simulation scenario for the following:

{description}

Respond with:
1. A brief explanation of the scenario
2. A JSON scenario object

Make sure the JSON is valid and complete."""
        
        response = self._chat(prompt)
        print(f"\n🤖 Claude's Response:\n{response}")
        
        scenario = self.parse_scenario_from_response(response)
        return scenario
    
    def run_scenario(self, scenario: Dict) -> Dict:
        """Run simulation with LLM-generated scenario"""
        print(f"\n▶️  Running simulation: {scenario.get('description', 'Custom scenario')}")
        
        # Extract configuration
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
        
        # Add nodes based on topology
        topology = scenario.get('topology', 'random')
        num_nodes = scenario.get('num_nodes', 5)
        
        positions = self._generate_positions(topology, num_nodes, 
                                            scenario.get('area_width', 100),
                                            scenario.get('area_height', 100))
        
        for i, (x, y) in enumerate(positions):
            config = NodeConfig(node_id=i, x=float(x), y=float(y))
            
            # Add mobility if enabled
            if scenario.get('mobility', {}).get('enabled'):
                mobility_model = {
                    'random_walk': MobilityModel.RANDOM_WALK,
                    'random_waypoint': MobilityModel.RANDOM_WAYPOINT,
                    'circular': MobilityModel.CIRCULAR,
                    'manhattan': MobilityModel.MANHATTAN,
                }.get(scenario['mobility']['model'], MobilityModel.RANDOM_WAYPOINT)
                
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
        
        # Add traffic flows
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
        
        self.last_results = metrics
        return metrics
    
    def _generate_positions(self, topology: str, num_nodes: int, 
                           width: float, height: float) -> List[Tuple[float, float]]:
        """Generate node positions based on topology"""
        if topology == 'line':
            spacing = width / (num_nodes + 1)
            return [(spacing * (i + 1), height / 2) for i in range(num_nodes)]
        
        elif topology == 'star':
            center = (width / 2, height / 2)
            radius = min(width, height) / 3
            positions = [center]
            for i in range(1, num_nodes):
                angle = 2 * 3.14159 * i / (num_nodes - 1)
                x = center[0] + radius * __import__('math').cos(angle)
                y = center[1] + radius * __import__('math').sin(angle)
                positions.append((x, y))
            return positions
        
        elif topology == 'grid':
            cols = int(__import__('math').sqrt(num_nodes))
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
    
    def analyze_results(self, query: str = "") -> str:
        """Ask Claude to analyze the simulation results"""
        if self.last_results is None:
            return "No results to analyze. Run a simulation first."
        
        print(f"\n🔍 Analyzing results: {query}")
        
        prompt = f"""Based on the simulation results below, provide insights:

Results:
- PDR: {self.last_results['pdr']:.2%}
- Collision Rate: {self.last_results['collision_rate']:.2%}
- Packets Sent: {self.last_results['packets_sent']}
- Packets Received: {self.last_results['packets_received']}
- Average Delay: {self.last_results.get('avg_delay', 0):.3f}s

{f"Additional context: {query}" if query else ""}

Provide:
1. What these results mean
2. Why they make sense
3. Potential improvements
4. Next experiments to try"""
        
        response = self._chat(prompt)
        return response
    
    def suggest_experiments(self) -> str:
        """Ask Claude to suggest follow-up experiments"""
        print("\n💡 Suggesting follow-up experiments...")
        
        if self.last_results is None:
            return "No results. Run a simulation first."
        
        prompt = f"""Based on the last simulation results:
- PDR: {self.last_results['pdr']:.2%}
- Collision Rate: {self.last_results['collision_rate']:.2%}

Suggest 3 follow-up experiments to understand this network better.
For each experiment:
1. What to change (describe in natural language)
2. Why it's interesting
3. What you expect to find"""
        
        response = self._chat(prompt)
        return response
    
    def conversation_mode(self):
        """Interactive conversation mode for exploring scenarios"""
        print("\n" + "="*80)
        print("LLM-DRIVEN WIRELESS SIMULATOR - INTERACTIVE MODE")
        print("="*80)
        print("\nCommands:")
        print("  'generate <description>' - Generate scenario from description")
        print("  'run' - Run the last generated scenario")
        print("  'analyze' - Analyze last results")
        print("  'suggest' - Get experiment suggestions")
        print("  'chat <message>' - Chat with Claude about wireless networks")
        print("  'quit' - Exit")
        print()
        
        current_scenario = None
        
        while True:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            elif user_input.lower().startswith('generate '):
                desc = user_input[9:]
                try:
                    current_scenario = self.generate_scenario(desc)
                    print(f"✅ Scenario ready. Type 'run' to execute.")
                except Exception as e:
                    print(f"❌ Error: {e}")
            
            elif user_input.lower() == 'run':
                if current_scenario is None:
                    print("❌ No scenario. Use 'generate <description>' first.")
                else:
                    try:
                        self.run_scenario(current_scenario)
                        print("✅ Simulation complete.")
                    except Exception as e:
                        print(f"❌ Error: {e}")
            
            elif user_input.lower() == 'analyze':
                analysis = self.analyze_results()
                print(f"\n🤖 Analysis:\n{analysis}")
            
            elif user_input.lower() == 'suggest':
                suggestions = self.suggest_experiments()
                print(f"\n🤖 Suggestions:\n{suggestions}")
            
            elif user_input.lower().startswith('chat '):
                msg = user_input[5:]
                response = self._chat(msg)
                print(f"\n🤖 Claude:\n{response}")
            
            else:
                print("Unknown command. Type 'generate', 'run', 'analyze', 'suggest', 'chat', or 'quit'.")


def run_llm_scenarios():
    """Run a series of LLM-generated scenarios"""
    sim = LLMWirelessSimulator()
    
    # Scenario 1: Simple wireless network
    print("\n" + "="*80)
    print("SCENARIO 1: Simple Wireless Network")
    print("="*80)
    
    scenario1 = sim.generate_scenario(
        "Create a simple wireless network with 5 nodes in a line topology. "
        "Use Rayleigh fading to simulate realistic indoor propagation. "
        "Run for 10 seconds with moderate traffic."
    )
    
    results1 = sim.run_scenario(scenario1)
    analysis1 = sim.analyze_results()
    print(f"\n🤖 Analysis:\n{analysis1}")
    
    # Scenario 2: Dense urban network
    print("\n" + "="*80)
    print("SCENARIO 2: Dense Urban Network with Mobility")
    print("="*80)
    
    scenario2 = sim.generate_scenario(
        "Design a dense urban wireless network with 10 mobile nodes. "
        "Nodes move randomly in a 50x50m area. "
        "Include Rayleigh fading and exponential backoff MAC. "
        "Use heavy traffic to stress-test the network. "
        "Run for 15 seconds."
    )
    
    results2 = sim.run_scenario(scenario2)
    analysis2 = sim.analyze_results()
    print(f"\n🤖 Analysis:\n{analysis2}")
    
    # Scenario 3: Star topology with rate adaptation
    print("\n" + "="*80)
    print("SCENARIO 3: Star Topology with Rate Adaptation")
    print("="*80)
    
    scenario3 = sim.generate_scenario(
        "Create a star topology network with central hub and 8 client nodes. "
        "Test both with and without fading to understand channel impact. "
        "Focus on medium traffic load. "
        "Run for 10 seconds."
    )
    
    results3 = sim.run_scenario(scenario3)
    analysis3 = sim.analyze_results()
    print(f"\n🤖 Analysis:\n{analysis3}")
    
    # Get experiment suggestions
    print("\n" + "="*80)
    print("EXPERIMENT SUGGESTIONS")
    print("="*80)
    
    suggestions = sim.suggest_experiments()
    print(f"\n🤖 Claude's Suggestions:\n{suggestions}")
    
    print("\n" + "="*80)
    print("AUTOMATED SCENARIOS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # Interactive mode
        sim = LLMWirelessSimulator()
        sim.conversation_mode()
    else:
        # Run preset scenarios
        run_llm_scenarios()
