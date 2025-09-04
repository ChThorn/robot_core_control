import numpy as np

class Tool:
    """
    Represents a tool attached to the robot's TCP (Tool Center Point).
    
    The tool's position and orientation are specified relative to the robot's TCP frame,
    which is already 100mm offset from the flange according to the URDF.
    
    TCP Frame Convention:
    - Z-axis: Points away from the flange (tool direction)
    - X-axis: Forward direction in flange plane  
    - Y-axis: Side direction in flange plane (right-hand rule)
    """
    
    def __init__(self, name, position_offset_mm, orientation_deg):
        """
        Initializes the tool with its specific geometry relative to the TCP.

        Args:
            name (str): The name of the tool (e.g., "gripper", "welder", "camera").
            position_offset_mm (list): The [X, Y, Z] offset from TCP in millimeters.
                                     Positive Z extends further from the flange.
            orientation_deg (list): The [Roll, Pitch, Yaw] orientation relative to TCP in degrees.
                                  Uses ZYX extrinsic Euler convention (same as robot targets).
        
        Example:
            # A gripper that extends 50mm further from TCP with no rotation
            gripper = Tool("gripper", [0, 0, 50], [0, 0, 0])
            
            # A camera offset 20mm to the side and angled 45° down
            camera = Tool("camera", [20, 0, 30], [0, 45, 0])
        """
        self.name = name
        self.position_offset_mm = np.array(position_offset_mm)
        self.orientation_deg = np.array(orientation_deg)
        
        # Create the transformation matrix from TCP to tool tip
        self.transform = self._create_transform_matrix(position_offset_mm, orientation_deg)
        self.transform_inv = np.linalg.inv(self.transform)
        
        # Store for debugging/inspection
        self.position_offset_m = self.position_offset_mm / 1000.0
        
        print(f"Tool '{self.name}' created:")
        print(f"  Position offset: {self.position_offset_mm} mm")
        print(f"  Orientation: {self.orientation_deg}° (Roll, Pitch, Yaw)")

    def _create_transform_matrix(self, position_mm, orientation_deg):
        """
        Creates the 4x4 homogeneous transformation matrix from TCP to tool tip.
        
        Uses ZYX extrinsic Euler convention to match robot_kinematics.py
        """
        # Convert position to meters
        position_m = np.array(position_mm) / 1000.0
        
        # Convert angles to radians
        roll_rad, pitch_rad, yaw_rad = np.deg2rad(orientation_deg)

        # Create individual rotation matrices
        c_r, s_r = np.cos(roll_rad), np.sin(roll_rad)
        c_p, s_p = np.cos(pitch_rad), np.sin(pitch_rad)
        c_y, s_y = np.cos(yaw_rad), np.sin(yaw_rad)

        # Individual rotation matrices
        Rx = np.array([[1, 0, 0], 
                       [0, c_r, -s_r], 
                       [0, s_r, c_r]])
        
        Ry = np.array([[c_p, 0, s_p], 
                       [0, 1, 0], 
                       [-s_p, 0, c_p]])
        
        Rz = np.array([[c_y, -s_y, 0], 
                       [s_y, c_y, 0], 
                       [0, 0, 1]])
        
        # ZYX extrinsic rotation (same convention as robot targets)
        # This means: first rotate about Z, then Y, then X (all in fixed world frame)
        rotation_matrix = Rz @ Ry @ Rx

        # Create homogeneous transformation matrix
        T = np.identity(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = position_m
        return T

    def get_transform(self):
        """Returns the tool's 4x4 transformation matrix (TCP to tool tip)."""
        return self.transform

    def get_inverse_transform(self):
        """Returns the inverse transformation matrix (tool tip to TCP)."""
        return self.transform_inv
    
    def get_tool_info(self):
        """Returns tool information for debugging/verification."""
        return {
            'name': self.name,
            'position_offset_mm': self.position_offset_mm,
            'orientation_deg': self.orientation_deg,
            'transform_matrix': self.transform
        }
    
    def visualize_tool_frame(self):
        """Print the tool's coordinate frame for verification."""
        print(f"\n--- Tool '{self.name}' Frame Information ---")
        print("Transformation Matrix (TCP -> Tool Tip):")
        print(self.transform)
        print("\nTool Tip Axes (relative to TCP):")
        print(f"  X-axis: {self.transform[:3, 0]}")
        print(f"  Y-axis: {self.transform[:3, 1]}")  
        print(f"  Z-axis: {self.transform[:3, 2]}")
        print(f"  Origin: {self.transform[:3, 3] * 1000} mm")


# Example tool definitions for common applications
class ToolLibrary:
    """Pre-defined tools for common applications."""
    
    @staticmethod
    def create_gripper(finger_length_mm=50):
        """Create a simple gripper tool extending from TCP."""
        return Tool("gripper", [0, 0, finger_length_mm], [0, 0, 0])
    
    @staticmethod  
    def create_welding_torch(torch_length_mm=80):
        """Create a welding torch extending from TCP."""
        return Tool("welding_torch", [0, 0, torch_length_mm], [0, 0, 0])
    
    @staticmethod
    def create_camera(side_offset_mm=30, down_angle_deg=30, z_offset_mm=20):
        """Create a camera tool offset to the side and angled down."""
        # Note: Added z_offset_mm to the method signature
        return Tool("camera", [side_offset_mm, 0, z_offset_mm], [0, down_angle_deg, 0])
    
    @staticmethod
    def create_drill(drill_length_mm=60):
        """Create a drill tool extending from TCP."""
        return Tool("drill", [0, 0, drill_length_mm], [0, 0, 0])

    # --- NEW METHOD ---
    @staticmethod
    def create_from_config(tool_config: dict):
        """
        Factory method to create a Tool object from a configuration dictionary.
        
        Args:
            tool_config (dict): A dictionary from config.yaml, e.g., {'type': 'gripper', 'finger_length_mm': 60}
        
        Returns:
            Tool: An initialized Tool object.
        """
        tool_type = tool_config.get("type")
        if not tool_type:
            raise ValueError("Tool configuration must have a 'type' key.")

        if tool_type == "gripper":
            length = tool_config.get("finger_length_mm", 50) # Use 50 as default
            return ToolLibrary.create_gripper(length)
        
        elif tool_type == "camera":
            offset = tool_config.get("side_offset_mm", 30)
            angle = tool_config.get("down_angle_deg", 30)
            z_offset = tool_config.get("z_offset_mm", 20)
            return ToolLibrary.create_camera(offset, angle, z_offset)

        elif tool_type == "welding_torch":
            length = tool_config.get("torch_length_mm", 80)
            return ToolLibrary.create_welding_torch(length)
            
        else:
            raise ValueError(f"Unknown tool type in config: '{tool_type}'")


if __name__ == '__main__':
    """Test the tool creation and transformations."""
    
    print("=== Tool Flange Testing ===\n")
    
    # Test 1: Simple gripper
    print("1. Testing Simple Gripper:")
    gripper = ToolLibrary.create_gripper(60)  # 60mm fingers
    gripper.visualize_tool_frame()
    
    # Test 2: Angled camera
    print("\n" + "="*50)
    print("2. Testing Angled Camera:")
    camera = Tool("inspection_camera", [25, 0, 40], [0, 45, 0])
    camera.visualize_tool_frame()
    
    # Test 3: Custom tool with complex orientation
    print("\n" + "="*50)
    print("3. Testing Complex Custom Tool:")
    custom_tool = Tool("custom_tool", [10, 5, 30], [15, -30, 45])
    custom_tool.visualize_tool_frame()
    
    # Test 4: Verify transformation properties
    print("\n" + "="*50)
    print("4. Verifying Inverse Transform:")
    T = gripper.get_transform()
    T_inv = gripper.get_inverse_transform()
    identity_check = T @ T_inv
    print("T @ T_inv should be identity:")
    print(identity_check)
    print(f"Is identity? {np.allclose(identity_check, np.eye(4))}")
    
    # Test 5: Show how to use with robot
    print("\n" + "="*50)
    print("5. Integration Example:")
    print("""
# Example usage with robot:
from robot_kinematics import RobotKinematics
from tool_flange import Tool, ToolLibrary

# Create robot and tool
robot = RobotKinematics()
gripper = ToolLibrary.create_gripper(50)

# Attach tool
robot.set_tool(gripper)

# Now when you do FK, you get both TCP and tool tip poses
joint_angles = [0, 0, 0, 0, 0, 0]
poses = robot.forward_kinematics(joint_angles)
tcp_pose = poses[6]      # TCP (flange + 100mm Z)
tool_tip_pose = poses[7] # Tool tip (TCP + tool offset)

# For IK, specify the desired tool tip pose
target_tool_tip_pose = robot.create_target_matrix([400, 0, 600], [0, 0, 0])
joint_solution = robot.inverse_kinematics(target_tool_tip_pose)
    """)