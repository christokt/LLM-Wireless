import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json

@dataclass
class NodeConfig:
    """Configuration for a network node"""
    node_id: int
    x: float
    y: float
    tx_power: float = 20.0  # dBm
    carrier_sense_threshold: float = -82.0  # dBm
    
@dataclass
class TrafficConfig:
    """Configuration for traffic generation"""
    source_id: int
    dest_id: int
    packet_size: int = 1500  # bytes
    inter_arrival_time: float = 0.01  # seconds (100 packets/sec)

class WirelessSimulator:
    """Simple discrete-event wireless network simulator"""
    
    def __init__(self, duration: float = 10.0, time_step: float = 0.001):
        """
        Initialize simulator
        
        Args:
            duration: Simulation duration in seconds
            time_step: Discrete time step in seconds
        """
        self.duration = duration
        self.time_step = time_step
        self.current_time = 0.0
        self.nodes: Dict[int, 'Node'] = {}
        self.traffic_flows: List[TrafficConfig] = []
        
        # Metrics
        self.packets_sent = 0
        self.packets_received = 0
        self.packets_collided = 0
        self.total_delay = 0.0
        self.per_node_metrics = {}
        
    def add_node(self, config: NodeConfig) -> None:
        """Add a node to the network"""
        self.nodes[config.node_id] = Node(config, self)
        self.per_node_metrics[config.node_id] = {
            'sent': 0,
            'received': 0,
            'collided': 0,
            'total_delay': 0.0
        }
    
    def add_traffic_flow(self, config: TrafficConfig) -> None:
        """Add a traffic flow between two nodes"""
        self.traffic_flows.append(config)
        self.nodes[config.source_id].add_traffic_flow(config)
    
    def get_distance(self, node_id1: int, node_id2: int) -> float:
        """Calculate Euclidean distance between two nodes"""
        n1 = self.nodes[node_id1]
        n2 = self.nodes[node_id2]
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    
    def path_loss(self, distance: float, frequency_ghz: float = 2.4) -> float:
        """
        Calculate path loss using log-distance model
        
        Args:
            distance: Distance in meters
            frequency_ghz: Frequency in GHz (default 2.4 GHz for WiFi)
        
        Returns:
            Path loss in dB
        """
        if distance < 1.0:
            distance = 1.0
        
        # Free space path loss at 1m
        pl0 = 20 * np.log10(frequency_ghz * 1e9 / (4 * np.pi * 1e8))
        # Path loss exponent (2.0 for free space)
        n = 2.0
        
        return pl0 + 10 * n * np.log10(distance)
    
    def get_received_power(self, tx_node_id: int, rx_node_id: int) -> float:
        """Calculate received power at rx_node from tx_node (in dBm)"""
        tx_node = self.nodes[tx_node_id]
        distance = self.get_distance(tx_node_id, rx_node_id)
        pl = self.path_loss(distance)
        return tx_node.tx_power - pl
    
    def step(self) -> None:
        """Execute one simulation step"""
        # Generate traffic
        for node_id, node in self.nodes.items():
            node.generate_traffic(self.current_time)
        
        # Simulate transmission attempts and collisions
        transmitting_nodes = []
        for node_id, node in self.nodes.items():
            if node.has_packet_to_send():
                transmitting_nodes.append(node_id)
        
        # Check for collisions and deliver packets
        # Track which receivers have collisions
        receiver_collision_count = {}
        
        for tx_id in transmitting_nodes:
            for rx_id, rx_node in self.nodes.items():
                if tx_id == rx_id:
                    continue
                
                # Calculate received power
                rx_power = self.get_received_power(tx_id, rx_id)
                
                # Check if signal is strong enough (above carrier sense threshold)
                if rx_power > self.nodes[rx_id].carrier_sense_threshold:
                    if rx_id not in receiver_collision_count:
                        receiver_collision_count[rx_id] = 0
                    receiver_collision_count[rx_id] += 1
        
        # Process receptions and collisions
        for tx_id in transmitting_nodes:
            for rx_id, rx_node in self.nodes.items():
                if tx_id == rx_id:
                    continue
                
                # Calculate received power
                rx_power = self.get_received_power(tx_id, rx_id)
                
                # Check if signal is strong enough (above carrier sense threshold)
                if rx_power > self.nodes[rx_id].carrier_sense_threshold:
                    # Collision if multiple transmitters sending to this receiver
                    if receiver_collision_count[rx_id] > 1:
                        self.packets_collided += 1
                        self.per_node_metrics[tx_id]['collided'] += 1
                    else:
                        # Successful reception
                        delay = self.current_time - self.nodes[tx_id].packet_send_time
                        self.packets_received += 1
                        self.total_delay += delay
                        self.per_node_metrics[rx_id]['received'] += 1
                        self.per_node_metrics[tx_id]['total_delay'] += delay
        
        # Update sent count
        for tx_id in transmitting_nodes:
            self.packets_sent += 1
            self.per_node_metrics[tx_id]['sent'] += 1
        
        # Clear transmitted packets
        for node_id in transmitting_nodes:
            self.nodes[node_id].clear_current_packet()
        
        self.current_time += self.time_step
    
    def run(self) -> Dict:
        """Run the simulation and return metrics"""
        while self.current_time < self.duration:
            self.step()
        
        return self.get_metrics()
    
    def get_metrics(self) -> Dict:
        """Get simulation metrics"""
        pdr = self.packets_received / self.packets_sent if self.packets_sent > 0 else 0
        avg_delay = self.total_delay / self.packets_received if self.packets_received > 0 else 0
        collision_rate = self.packets_collided / self.packets_sent if self.packets_sent > 0 else 0
        
        return {
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'packets_collided': self.packets_collided,
            'pdr': pdr,  # Packet Delivery Ratio
            'avg_delay': avg_delay,  # Average end-to-end delay
            'collision_rate': collision_rate,
            'per_node': self.per_node_metrics,
            'simulation_time': self.duration
        }

class Node:
    """Represents a wireless network node"""
    
    def __init__(self, config: NodeConfig, simulator: WirelessSimulator):
        self.node_id = config.node_id
        self.x = config.x
        self.y = config.y
        self.tx_power = config.tx_power
        self.carrier_sense_threshold = config.carrier_sense_threshold
        self.simulator = simulator
        
        self.traffic_flows: List[TrafficConfig] = []
        self.current_packet = None
        self.packet_send_time = None
        self.next_packet_time: Dict[int, float] = {}  # Next packet for each flow
    
    def add_traffic_flow(self, config: TrafficConfig) -> None:
        """Add a traffic flow originating from this node"""
        self.traffic_flows.append(config)
        self.next_packet_time[config.dest_id] = config.inter_arrival_time
    
    def generate_traffic(self, current_time: float) -> None:
        """Generate packets according to traffic flows"""
        for flow in self.traffic_flows:
            if current_time >= self.next_packet_time[flow.dest_id]:
                self.current_packet = {
                    'source': self.node_id,
                    'dest': flow.dest_id,
                    'size': flow.packet_size,
                    'created_at': current_time
                }
                self.packet_send_time = current_time
                self.next_packet_time[flow.dest_id] = current_time + flow.inter_arrival_time
    
    def has_packet_to_send(self) -> bool:
        """Check if node has a packet ready to send"""
        return self.current_packet is not None
    
    def clear_current_packet(self) -> None:
        """Clear the current packet after transmission"""
        self.current_packet = None


if __name__ == "__main__":
    # Example: Simple WiFi scenario
    sim = WirelessSimulator(duration=10.0, time_step=0.001)
    
    # Add nodes
    sim.add_node(NodeConfig(node_id=0, x=0.0, y=0.0))
    sim.add_node(NodeConfig(node_id=1, x=10.0, y=0.0))
    sim.add_node(NodeConfig(node_id=2, x=20.0, y=0.0))
    
    # Add traffic flows
    sim.add_traffic_flow(TrafficConfig(source_id=0, dest_id=1, inter_arrival_time=0.01))
    sim.add_traffic_flow(TrafficConfig(source_id=1, dest_id=2, inter_arrival_time=0.01))
    
    # Run simulation
    metrics = sim.run()
    
    print("=== Simulation Results ===")
    print(f"Packets Sent: {metrics['packets_sent']}")
    print(f"Packets Received: {metrics['packets_received']}")
    print(f"Packets Collided: {metrics['packets_collided']}")
    print(f"Packet Delivery Ratio: {metrics['pdr']:.4f}")
    print(f"Average Delay: {metrics['avg_delay']:.6f} seconds")
    print(f"Collision Rate: {metrics['collision_rate']:.4f}")
    print(f"\nPer-Node Metrics:")
    for node_id, node_metrics in metrics['per_node'].items():
        print(f"  Node {node_id}: sent={node_metrics['sent']}, received={node_metrics['received']}")
