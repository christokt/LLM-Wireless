"""
Extended Wireless Simulator Demonstrations
Shows all new features: mobility, MAC, fading, rate adaptation
"""

import numpy as np
from wireless_simulator_extended import (
    ExtendedWirelessSimulator, MobilityConfig, MobilityModel, 
    FadingConfig, FadingModel, RateAdaptationAlgorithm
)
from wireless_simulator import NodeConfig, TrafficConfig

def demo_mobility():
    """Demo 1: Node Mobility"""
    print("\n" + "="*80)
    print("DEMO 1: NODE MOBILITY")
    print("="*80)
    
    print("\nScenario: 4 nodes with random waypoint mobility in 100x100m area")
    
    sim = ExtendedWirelessSimulator(duration=10.0, time_step=0.01, use_mobility=True)
    
    # Add nodes with mobility
    for i in range(4):
        config = NodeConfig(node_id=i, x=float(i * 20), y=0.0)
        mobility_config = MobilityConfig(
            model=MobilityModel.RANDOM_WAYPOINT,
            speed=5.0,  # 5 m/s
            pause_time=1.0,
            area_width=100.0,
            area_height=100.0
        )
        sim.add_mobile_node(config, mobility_config)
    
    # Add traffic
    sim.add_traffic_flow(TrafficConfig(source_id=0, dest_id=2, inter_arrival_time=0.05))
    sim.add_traffic_flow(TrafficConfig(source_id=1, dest_id=3, inter_arrival_time=0.05))
    
    metrics = sim.run()
    
    print(f"\nResults:")
    print(f"  PDR: {metrics['pdr']:.2%}")
    print(f"  Total packets: {metrics['packets_sent']}")
    print(f"  Total distance traveled: {metrics.get('total_distance_traveled', 0):.1f}m")
    print(f"  Average distance per node: {metrics.get('avg_distance_per_node', 0):.1f}m")

def demo_backoff_mac():
    """Demo 2: Exponential Backoff MAC Protocol"""
    print("\n" + "="*80)
    print("DEMO 2: EXPONENTIAL BACKOFF MAC")
    print("="*80)
    
    print("\nScenario: 3 nodes in close proximity with backoff MAC")
    print("Expected: Collisions reduce over time due to backoff")
    
    sim = ExtendedWirelessSimulator(duration=10.0, use_backoff_mac=True)
    
    # Add nodes with backoff MAC
    positions = [(0.0, 0.0), (5.0, 0.0), (2.5, 4.0)]
    for i, (x, y) in enumerate(positions):
        config = NodeConfig(node_id=i, x=x, y=y)
        sim.add_backoff_node(config)
    
    # Heavy traffic to trigger collisions
    sim.add_traffic_flow(TrafficConfig(source_id=0, dest_id=1, inter_arrival_time=0.005))
    sim.add_traffic_flow(TrafficConfig(source_id=1, dest_id=0, inter_arrival_time=0.005))
    sim.add_traffic_flow(TrafficConfig(source_id=2, dest_id=1, inter_arrival_time=0.005))
    
    metrics = sim.run()
    
    print(f"\nResults:")
    print(f"  PDR: {metrics['pdr']:.2%}")
    print(f"  Collision Rate: {metrics['collision_rate']:.2%}")
    print(f"  Average collisions per node: {metrics.get('avg_collisions_per_node', 0):.2f}")

def demo_fading_rayleigh():
    """Demo 3: Rayleigh Fading Channel"""
    print("\n" + "="*80)
    print("DEMO 3: RAYLEIGH FADING CHANNEL")
    print("="*80)
    
    print("\nScenario: 2 nodes with Rayleigh fading (non-line-of-sight)")
    print("Expected: Lower PDR than free space due to fading")
    
    # Compare with and without fading
    for use_fading in [False, True]:
        label = "Rayleigh Fading" if use_fading else "No Fading (Baseline)"
        
        fading_config = FadingConfig(model=FadingModel.RAYLEIGH) if use_fading else None
        sim = ExtendedWirelessSimulator(duration=5.0, fading_config=fading_config)
        
        # Add nodes far apart (at edge of range)
        sim.add_node(NodeConfig(node_id=0, x=0.0, y=0.0))
        sim.add_node(NodeConfig(node_id=1, x=80.0, y=0.0))
        
        sim.add_traffic_flow(TrafficConfig(source_id=0, dest_id=1, inter_arrival_time=0.01))
        
        metrics = sim.run()
        
        print(f"\n  {label}:")
        print(f"    PDR: {metrics['pdr']:.2%}")
        print(f"    Packets sent: {metrics['packets_sent']}")
        print(f"    Packets received: {metrics['packets_received']}")

def demo_fading_comparison():
    """Demo 4: Compare Different Fading Models"""
    print("\n" + "="*80)
    print("DEMO 4: FADING MODEL COMPARISON")
    print("="*80)
    
    print("\nScenario: Compare Rayleigh, Rician, and Shadowing")
    
    fading_models = [
        (FadingModel.NONE, "No Fading"),
        (FadingModel.RAYLEIGH, "Rayleigh"),
        (FadingModel.RICIAN, "Rician (K=5)"),
        (FadingModel.SHADOWING, "Log-Normal Shadowing")
    ]
    
    results = {}
    
    for model, label in fading_models:
        fading_config = FadingConfig(model=model, rician_k=5.0, shadowing_std=8.0)
        sim = ExtendedWirelessSimulator(duration=5.0, fading_config=fading_config)
        
        # Two nodes at moderate distance
        sim.add_node(NodeConfig(node_id=0, x=0.0, y=0.0))
        sim.add_node(NodeConfig(node_id=1, x=30.0, y=0.0))
        
        sim.add_traffic_flow(TrafficConfig(source_id=0, dest_id=1, inter_arrival_time=0.01))
        
        metrics = sim.run()
        results[label] = metrics['pdr']
        
        print(f"  {label:20s}: PDR = {metrics['pdr']:.2%}")

def demo_rate_adaptation():
    """Demo 5: Rate Adaptation"""
    print("\n" + "="*80)
    print("DEMO 5: RATE ADAPTATION")
    print("="*80)
    
    print("\nScenario: Node adapts transmission rate based on SNR")
    print("Expected: Higher PDR with rate adaptation vs static rate")
    
    for use_rate_adapt in [False, True]:
        label = "With Rate Adaptation" if use_rate_adapt else "Static Rate (54 Mbps)"
        
        sim = ExtendedWirelessSimulator(
            duration=10.0, 
            use_rate_adaptation=use_rate_adapt
        )
        
        # Two nodes with varying distance to simulate changing SNR
        sim.add_node(NodeConfig(node_id=0, x=0.0, y=0.0))
        sim.add_node(NodeConfig(node_id=1, x=50.0, y=0.0))
        
        sim.add_traffic_flow(TrafficConfig(source_id=0, dest_id=1, inter_arrival_time=0.01))
        
        metrics = sim.run()
        
        print(f"\n  {label}:")
        print(f"    PDR: {metrics['pdr']:.2%}")

def demo_combined_features():
    """Demo 6: All Features Combined"""
    print("\n" + "="*80)
    print("DEMO 6: COMBINED FEATURES (Mobility + Backoff + Fading)")
    print("="*80)
    
    print("\nScenario: Realistic WiFi network with all advanced features")
    print("  - 5 mobile nodes moving randomly")
    print("  - Exponential backoff MAC protocol")
    print("  - Rayleigh fading channels")
    print("  - Run for 20 seconds")
    
    fading_config = FadingConfig(model=FadingModel.RAYLEIGH)
    sim = ExtendedWirelessSimulator(
        duration=20.0,
        time_step=0.01,
        fading_config=fading_config,
        use_mobility=True,
        use_backoff_mac=True
    )
    
    # Add 5 mobile nodes
    for i in range(5):
        config = NodeConfig(node_id=i, x=float(i * 15), y=0.0)
        mobility_config = MobilityConfig(
            model=MobilityModel.RANDOM_WALK,
            speed=2.0,
            area_width=100.0,
            area_height=100.0
        )
        sim.add_mobile_node(config, mobility_config)
    
    # Create some traffic flows
    sim.add_traffic_flow(TrafficConfig(source_id=0, dest_id=2, inter_arrival_time=0.02))
    sim.add_traffic_flow(TrafficConfig(source_id=1, dest_id=3, inter_arrival_time=0.02))
    sim.add_traffic_flow(TrafficConfig(source_id=3, dest_id=4, inter_arrival_time=0.02))
    
    metrics = sim.run()
    
    print(f"\nResults:")
    print(f"  Packets sent: {metrics['packets_sent']}")
    print(f"  Packets received: {metrics['packets_received']}")
    print(f"  PDR: {metrics['pdr']:.2%}")
    print(f"  Collision Rate: {metrics['collision_rate']:.2%}")
    print(f"  Avg Delay: {metrics['avg_delay']*1000:.2f} ms")
    print(f"  Total distance: {metrics.get('total_distance_traveled', 0):.0f}m")

def demo_topology_comparison():
    """Demo 7: Compare Topologies with Advanced Features"""
    print("\n" + "="*80)
    print("DEMO 7: TOPOLOGY COMPARISON (with Fading + Backoff)")
    print("="*80)
    
    topologies = {
        "Line (4 nodes)": [(0, 0), (20, 0), (40, 0), (60, 0)],
        "Star (5 nodes)": [(50, 50), (30, 50), (70, 50), (50, 30), (50, 70)],
        "Grid (9 nodes)": [
            (0, 0), (30, 0), (60, 0),
            (0, 30), (30, 30), (60, 30),
            (0, 60), (30, 60), (60, 60)
        ]
    }
    
    fading_config = FadingConfig(model=FadingModel.SHADOWING)
    
    for topology_name, positions in topologies.items():
        sim = ExtendedWirelessSimulator(
            duration=10.0,
            fading_config=fading_config,
            use_backoff_mac=True
        )
        
        # Add nodes
        for i, (x, y) in enumerate(positions):
            config = NodeConfig(node_id=i, x=float(x), y=float(y))
            sim.add_backoff_node(config)
        
        # Add some realistic traffic flows
        for i in range(len(positions) - 1):
            sim.add_traffic_flow(TrafficConfig(
                source_id=i, 
                dest_id=i+1,
                inter_arrival_time=0.02
            ))
        
        metrics = sim.run()
        
        print(f"\n  {topology_name}:")
        print(f"    Nodes: {len(positions)}")
        print(f"    PDR: {metrics['pdr']:.2%}")
        print(f"    Collision Rate: {metrics['collision_rate']:.2%}")

def run_all_demos():
    """Run all demonstrations"""
    print("\n" + "="*80)
    print("EXTENDED WIRELESS SIMULATOR - COMPREHENSIVE DEMONSTRATIONS")
    print("="*80)
    
    demos = [
        demo_mobility,
        demo_backoff_mac,
        demo_fading_rayleigh,
        demo_fading_comparison,
        demo_rate_adaptation,
        demo_combined_features,
        demo_topology_comparison
    ]
    
    for i, demo in enumerate(demos, 1):
        try:
            demo()
        except Exception as e:
            print(f"\nError in demo {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    run_all_demos()
