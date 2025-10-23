import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
from enum import Enum
from wireless_simulator import WirelessSimulator, NodeConfig, TrafficConfig, Node

# ============================================================================
# EXTENSION 1: NODE MOBILITY MODELS
# ============================================================================

class MobilityModel(Enum):
    """Supported mobility models"""
    STATIC = "static"
    RANDOM_WALK = "random_walk"
    RANDOM_WAYPOINT = "random_waypoint"
    CIRCULAR = "circular"
    MANHATTAN = "manhattan"

@dataclass
class MobilityConfig:
    """Configuration for node mobility"""
    model: MobilityModel
    speed: float  # meters per second
    pause_time: float = 0.0  # seconds (for random waypoint)
    area_width: float = 100.0
    area_height: float = 100.0

class MobileNode(Node):
    """Node with mobility capabilities"""
    
    def __init__(self, config: NodeConfig, simulator, mobility_config: MobilityConfig = None):
        super().__init__(config, simulator)
        self.mobility_config = mobility_config
        self.vx = 0.0  # velocity x
        self.vy = 0.0  # velocity y
        self.waypoint_x = self.x
        self.waypoint_y = self.y
        self.pause_time_remaining = 0.0
        self.total_distance = 0.0
        
        if mobility_config:
            self._initialize_mobility()
    
    def _initialize_mobility(self):
        """Initialize velocity based on mobility model"""
        if self.mobility_config.model == MobilityModel.RANDOM_WALK:
            self._set_random_velocity()
        elif self.mobility_config.model == MobilityModel.RANDOM_WAYPOINT:
            self._pick_random_waypoint()
    
    def _set_random_velocity(self):
        """Set random velocity (random walk)"""
        angle = np.random.uniform(0, 2 * np.pi)
        self.vx = self.mobility_config.speed * np.cos(angle)
        self.vy = self.mobility_config.speed * np.sin(angle)
    
    def _pick_random_waypoint(self):
        """Pick random waypoint (random waypoint)"""
        self.waypoint_x = np.random.uniform(0, self.mobility_config.area_width)
        self.waypoint_y = np.random.uniform(0, self.mobility_config.area_height)
        
        # Calculate velocity toward waypoint
        dx = self.waypoint_x - self.x
        dy = self.waypoint_y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0:
            self.vx = (dx / dist) * self.mobility_config.speed
            self.vy = (dy / dist) * self.mobility_config.speed
    
    def update_position(self, time_step: float):
        """Update node position based on mobility model"""
        if not self.mobility_config:
            return
        
        if self.mobility_config.model == MobilityModel.STATIC:
            return
        
        elif self.mobility_config.model == MobilityModel.RANDOM_WALK:
            self._update_random_walk(time_step)
        
        elif self.mobility_config.model == MobilityModel.RANDOM_WAYPOINT:
            self._update_random_waypoint(time_step)
        
        elif self.mobility_config.model == MobilityModel.CIRCULAR:
            self._update_circular(time_step)
        
        elif self.mobility_config.model == MobilityModel.MANHATTAN:
            self._update_manhattan(time_step)
    
    def _update_random_walk(self, time_step: float):
        """Random walk: move in current direction, occasionally change direction"""
        # Update position
        new_x = self.x + self.vx * time_step
        new_y = self.y + self.vy * time_step
        
        # Bounce off boundaries
        if new_x < 0 or new_x > self.mobility_config.area_width:
            self.vx *= -1
            new_x = np.clip(new_x, 0, self.mobility_config.area_width)
        
        if new_y < 0 or new_y > self.mobility_config.area_height:
            self.vy *= -1
            new_y = np.clip(new_y, 0, self.mobility_config.area_height)
        
        # Track distance
        dx = new_x - self.x
        dy = new_y - self.y
        self.total_distance += np.sqrt(dx**2 + dy**2)
        
        self.x = new_x
        self.y = new_y
        
        # Occasionally change direction (5% per step)
        if np.random.random() < 0.05:
            self._set_random_velocity()
    
    def _update_random_waypoint(self, time_step: float):
        """Random waypoint: move toward target waypoint"""
        dx = self.waypoint_x - self.x
        dy = self.waypoint_y - self.y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Move toward waypoint
        self.x += self.vx * time_step
        self.y += self.vy * time_step
        
        # Check if reached waypoint
        if dist < self.mobility_config.speed * time_step:
            self.x = self.waypoint_x
            self.y = self.waypoint_y
            
            # Pause at waypoint
            self.pause_time_remaining = self.mobility_config.pause_time
            self.vx = 0
            self.vy = 0
        
        # If paused, decrement pause time
        if self.pause_time_remaining > 0:
            self.pause_time_remaining -= time_step
            if self.pause_time_remaining <= 0:
                self._pick_random_waypoint()
    
    def _update_circular(self, time_step: float):
        """Circular motion around center"""
        center_x = self.mobility_config.area_width / 2
        center_y = self.mobility_config.area_height / 2
        radius = min(center_x, center_y) * 0.8
        
        # Update angle
        angular_velocity = self.mobility_config.speed / radius
        angle = np.arctan2(self.y - center_y, self.x - center_x)
        angle += angular_velocity * time_step
        
        self.x = center_x + radius * np.cos(angle)
        self.y = center_y + radius * np.sin(angle)
    
    def _update_manhattan(self, time_step: float):
        """Manhattan movement: move only horizontally or vertically"""
        # Move in current direction
        self.x += self.vx * time_step
        self.y += self.vy * time_step
        
        # Bounce off boundaries
        if self.x < 0 or self.x > self.mobility_config.area_width:
            self.vx = -self.vx
            self.x = np.clip(self.x, 0, self.mobility_config.area_width)
        
        if self.y < 0 or self.y > self.mobility_config.area_height:
            self.vy = -self.vy
            self.y = np.clip(self.y, 0, self.mobility_config.area_height)
        
        # Change direction with some probability
        if np.random.random() < 0.02:
            # Pick random cardinal direction
            direction = np.random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
            self.vx = direction[0] * self.mobility_config.speed
            self.vy = direction[1] * self.mobility_config.speed

# ============================================================================
# EXTENSION 2: ADVANCED MAC LAYER WITH BACKOFF
# ============================================================================

class BackoffMAC(Node):
    """Node with exponential backoff MAC protocol"""
    
    def __init__(self, config: NodeConfig, simulator, max_backoff_exp: int = 6):
        super().__init__(config, simulator)
        self.backoff_exponent = 0
        self.max_backoff_exp = max_backoff_exp
        self.backoff_counter = 0
        self.collision_count = 0
        self.successful_transmissions = 0
    
    def has_packet_to_send(self) -> bool:
        """Check if ready to send (accounting for backoff)"""
        if self.backoff_counter > 0:
            self.backoff_counter -= 1
            return False
        
        return super().has_packet_to_send()
    
    def on_collision(self):
        """Handle collision by increasing backoff"""
        self.collision_count += 1
        self.backoff_exponent = min(self.backoff_exponent + 1, self.max_backoff_exp)
        
        # Set random backoff: 0 to 2^backoff_exponent - 1 time slots
        backoff_slots = np.random.randint(0, 2**self.backoff_exponent)
        self.backoff_counter = backoff_slots
    
    def on_success(self):
        """Reset backoff after successful transmission"""
        self.successful_transmissions += 1
        self.backoff_exponent = 0

# ============================================================================
# EXTENSION 3: FADING MODELS
# ============================================================================

class FadingModel(Enum):
    """Supported fading models"""
    NONE = "none"
    RAYLEIGH = "rayleigh"
    RICIAN = "rician"
    SHADOWING = "shadowing"

@dataclass
class FadingConfig:
    """Configuration for channel fading"""
    model: FadingModel
    rician_k: float = 5.0  # Rician K factor
    shadowing_std: float = 5.0  # Standard deviation in dB

class FadingSimulator(WirelessSimulator):
    """Simulator with fading channel support"""
    
    def __init__(self, duration: float = 10.0, time_step: float = 0.001, 
                 fading_config: FadingConfig = None):
        super().__init__(duration, time_step)
        self.fading_config = fading_config or FadingConfig(FadingModel.NONE)
        self.shadowing_map = {}  # Store shadowing values for path pairs
    
    def get_fading_attenuation(self, tx_id: int, rx_id: int) -> float:
        """Calculate fading attenuation in dB"""
        if self.fading_config.model == FadingModel.NONE:
            return 0.0
        
        elif self.fading_config.model == FadingModel.RAYLEIGH:
            # Rayleigh fading: exponentially distributed envelope
            return -10 * np.log10(np.random.exponential(1.0))
        
        elif self.fading_config.model == FadingModel.RICIAN:
            # Rician fading: sum of LOS component and scattered components
            k = self.fading_config.rician_k
            los_power = k / (k + 1)
            scattered_power = 1 / (k + 1)
            
            # LOS component
            los = np.sqrt(los_power)
            # Scattered components
            scattered_i = np.sqrt(scattered_power / 2) * np.random.normal()
            scattered_q = np.sqrt(scattered_power / 2) * np.random.normal()
            
            envelope = np.sqrt((los + scattered_i)**2 + scattered_q**2)
            return -10 * np.log10(envelope**2)
        
        elif self.fading_config.model == FadingModel.SHADOWING:
            # Log-normal shadowing: slow fading due to obstacles
            path_key = (min(tx_id, rx_id), max(tx_id, rx_id))
            
            if path_key not in self.shadowing_map:
                # Generate new shadowing value for this path
                self.shadowing_map[path_key] = np.random.normal(
                    0, self.fading_config.shadowing_std
                )
            
            return self.shadowing_map[path_key]
        
        return 0.0
    
    def get_received_power(self, tx_node_id: int, rx_node_id: int) -> float:
        """Calculate received power including fading"""
        base_power = super().get_received_power(tx_node_id, rx_node_id)
        fading_attenuation = self.get_fading_attenuation(tx_node_id, rx_node_id)
        return base_power + fading_attenuation

# ============================================================================
# EXTENSION 4: RATE ADAPTATION
# ============================================================================

class RateAdaptationAlgorithm(Enum):
    """Supported rate adaptation algorithms"""
    STATIC = "static"
    SNR_BASED = "snr_based"
    FRAME_SUCCESS_RATE = "fsr"

class RateAdaptiveNode(Node):
    """Node with dynamic rate adaptation"""
    
    def __init__(self, config: NodeConfig, simulator, algorithm: RateAdaptationAlgorithm = RateAdaptationAlgorithm.SNR_BASED):
        super().__init__(config, simulator)
        self.algorithm = algorithm
        self.current_rate = 54.0  # Mbps (802.11g)
        self.available_rates = [54.0, 48.0, 36.0, 24.0, 18.0, 12.0, 9.0, 6.0]  # Mbps
        self.success_count = 0
        self.total_transmissions = 0
        self.last_adaptation_time = 0
        self.adaptation_interval = 1.0  # seconds
    
    def adapt_rate(self, snr_db: float, current_time: float):
        """Adapt transmission rate based on channel conditions"""
        if current_time - self.last_adaptation_time < self.adaptation_interval:
            return
        
        if self.algorithm == RateAdaptationAlgorithm.STATIC:
            pass
        
        elif self.algorithm == RateAdaptationAlgorithm.SNR_BASED:
            # SNR-to-rate mapping (simplified 802.11g)
            if snr_db > 30:
                self.current_rate = 54.0
            elif snr_db > 25:
                self.current_rate = 48.0
            elif snr_db > 20:
                self.current_rate = 36.0
            elif snr_db > 15:
                self.current_rate = 24.0
            elif snr_db > 10:
                self.current_rate = 12.0
            else:
                self.current_rate = 6.0
        
        elif self.algorithm == RateAdaptationAlgorithm.FRAME_SUCCESS_RATE:
            # Adjust based on success rate
            if self.total_transmissions > 10:
                success_rate = self.success_count / self.total_transmissions
                
                if success_rate > 0.9:
                    # Try higher rate
                    idx = self.available_rates.index(self.current_rate)
                    if idx > 0:
                        self.current_rate = self.available_rates[idx - 1]
                
                elif success_rate < 0.5:
                    # Use lower rate
                    idx = self.available_rates.index(self.current_rate)
                    if idx < len(self.available_rates) - 1:
                        self.current_rate = self.available_rates[idx + 1]
                
                # Reset counters
                self.success_count = 0
                self.total_transmissions = 0
        
        self.last_adaptation_time = current_time

# ============================================================================
# EXTENSION 5: EXTENDED SIMULATOR WITH ALL FEATURES
# ============================================================================

class ExtendedWirelessSimulator(FadingSimulator):
    """Full-featured wireless simulator with mobility, MAC, fading, rate adaptation"""
    
    def __init__(self, duration: float = 10.0, time_step: float = 0.001,
                 fading_config: FadingConfig = None, use_mobility: bool = False,
                 use_backoff_mac: bool = False, use_rate_adaptation: bool = False):
        super().__init__(duration, time_step, fading_config)
        self.use_mobility = use_mobility
        self.use_backoff_mac = use_backoff_mac
        self.use_rate_adaptation = use_rate_adaptation
        self.mobility_nodes: Dict[int, MobileNode] = {}
    
    def add_mobile_node(self, config: NodeConfig, mobility_config: MobilityConfig):
        """Add a node with mobility"""
        node = MobileNode(config, self, mobility_config)
        self.nodes[config.node_id] = node
        self.mobility_nodes[config.node_id] = node
        self.per_node_metrics[config.node_id] = {
            'sent': 0, 'received': 0, 'collided': 0, 'total_delay': 0.0
        }
    
    def add_backoff_node(self, config: NodeConfig):
        """Add a node with exponential backoff MAC"""
        node = BackoffMAC(config, self)
        self.nodes[config.node_id] = node
        self.per_node_metrics[config.node_id] = {
            'sent': 0, 'received': 0, 'collided': 0, 'total_delay': 0.0
        }
    
    def add_rate_adaptive_node(self, config: NodeConfig, 
                              algorithm: RateAdaptationAlgorithm = RateAdaptationAlgorithm.SNR_BASED):
        """Add a node with rate adaptation"""
        node = RateAdaptiveNode(config, self, algorithm)
        self.nodes[config.node_id] = node
        self.per_node_metrics[config.node_id] = {
            'sent': 0, 'received': 0, 'collided': 0, 'total_delay': 0.0
        }
    
    def step(self) -> None:
        """Extended step with mobility and MAC enhancements"""
        # Update node positions if using mobility
        if self.use_mobility:
            for node in self.mobility_nodes.values():
                node.update_position(self.time_step)
        
        # Generate traffic
        for node_id, node in self.nodes.items():
            node.generate_traffic(self.current_time)
        
        # Simulate transmission attempts
        transmitting_nodes = []
        for node_id, node in self.nodes.items():
            if node.has_packet_to_send():
                transmitting_nodes.append(node_id)
        
        # Detect collisions per receiver
        receiver_collision_count = {}
        for tx_id in transmitting_nodes:
            for rx_id, rx_node in self.nodes.items():
                if tx_id == rx_id:
                    continue
                
                rx_power = self.get_received_power(tx_id, rx_id)
                
                if rx_power > self.nodes[rx_id].carrier_sense_threshold:
                    if rx_id not in receiver_collision_count:
                        receiver_collision_count[rx_id] = 0
                    receiver_collision_count[rx_id] += 1
        
        # Process receptions and collisions
        for tx_id in transmitting_nodes:
            for rx_id, rx_node in self.nodes.items():
                if tx_id == rx_id:
                    continue
                
                rx_power = self.get_received_power(tx_id, rx_id)
                
                if rx_power > self.nodes[rx_id].carrier_sense_threshold:
                    if receiver_collision_count[rx_id] > 1:
                        # Collision
                        self.packets_collided += 1
                        self.per_node_metrics[tx_id]['collided'] += 1
                        
                        # Notify node if using backoff MAC
                        if isinstance(self.nodes[tx_id], BackoffMAC):
                            self.nodes[tx_id].on_collision()
                    else:
                        # Successful reception
                        delay = self.current_time - self.nodes[tx_id].packet_send_time
                        self.packets_received += 1
                        self.total_delay += delay
                        self.per_node_metrics[rx_id]['received'] += 1
                        self.per_node_metrics[tx_id]['total_delay'] += delay
                        
                        # Notify node if using backoff MAC
                        if isinstance(self.nodes[tx_id], BackoffMAC):
                            self.nodes[tx_id].on_success()
        
        # Update sent count
        for tx_id in transmitting_nodes:
            self.packets_sent += 1
            self.per_node_metrics[tx_id]['sent'] += 1
        
        # Clear transmitted packets
        for node_id in transmitting_nodes:
            self.nodes[node_id].clear_current_packet()
        
        self.current_time += self.time_step
    
    def get_metrics(self) -> Dict:
        """Get extended metrics including mobility information"""
        metrics = super().get_metrics()
        
        # Add mobility stats
        if self.use_mobility:
            total_distance = sum(node.total_distance for node in self.mobility_nodes.values())
            metrics['total_distance_traveled'] = total_distance
            metrics['avg_distance_per_node'] = total_distance / len(self.mobility_nodes) if self.mobility_nodes else 0
        
        # Add MAC stats
        if self.use_backoff_mac:
            backoff_nodes = {node_id: node for node_id, node in self.nodes.items() 
                           if isinstance(node, BackoffMAC)}
            if backoff_nodes:
                avg_collisions = np.mean([node.collision_count for node in backoff_nodes.values()])
                metrics['avg_collisions_per_node'] = avg_collisions
        
        return metrics


if __name__ == "__main__":
    print("Extended Wireless Simulator Loaded")
    print("\nAvailable Extensions:")
    print("1. Node Mobility (RANDOM_WALK, RANDOM_WAYPOINT, CIRCULAR, MANHATTAN)")
    print("2. Exponential Backoff MAC Protocol")
    print("3. Fading Channels (RAYLEIGH, RICIAN, SHADOWING)")
    print("4. Rate Adaptation (SNR-based, Frame Success Rate)")
    print("5. Combined Features in ExtendedWirelessSimulator")
