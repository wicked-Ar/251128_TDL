"""
PyBullet Multi-Robot Simulation Environment
A robust simulation environment for multi-robot collaborative task planning.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import math


class MultiRobotEnv:
    """
    A multi-robot collaborative environment using PyBullet.
    Features two robots (UR5 and Panda) working around a central table with multiple objects.
    """

    def __init__(self, gui=True):
        """
        Initialize the simulation environment.

        Args:
            gui (bool): Whether to use GUI mode (True) or DIRECT mode (False)
        """
        # Connect to PyBullet
        if gui:
            # Use GUI_SERVER with custom resolution to ensure H.264-compatible dimensions
            # Must use even height (e.g., 1024x768, not 1024x757) for H.264 encoding
            options = "--width=1024 --height=768"
            self.client = p.connect(p.GUI, options=options)
        else:
            self.client = p.connect(p.DIRECT)

        # Set additional search path for URDF files
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity
        p.setGravity(0, 0, -9.81)

        # Object registry to track all objects with metadata
        self.objects = {}
        self.robot_ids = {}

        # Load environment components
        self._load_environment()

        print("MultiRobotEnv initialized successfully!")

    def _load_environment(self):
        """Load all environment components: floor, table, robots, and objects."""

        # Load plane
        self.plane_id = p.loadURDF("plane.urdf")

        # Load table at center
        # Note: table.urdf might need height adjustment
        self.table_id = p.loadURDF("table/table.urdf",
                                    basePosition=[0, 0, 0],
                                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))

        # Get table dimensions for proper object placement
        # Table height is approximately 0.625 based on standard table.urdf
        self.table_height = 0.625
        self.table_surface_height = self.table_height + 0.05  # Slight offset above table

        # Load Robot A (KUKA IIWA - UR5 equivalent)
        kuka_start_pos = [-0.6, 0, self.table_height]
        kuka_start_orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robot_ids['kuka'] = p.loadURDF("kuka_iiwa/model.urdf",
                                            basePosition=kuka_start_pos,
                                            baseOrientation=kuka_start_orientation,
                                            useFixedBase=True)

        # Load Robot B (Panda)
        panda_start_pos = [0.6, 0, self.table_height]
        panda_start_orientation = p.getQuaternionFromEuler([0, 0, math.pi])  # Face the table

        self.robot_ids['panda'] = p.loadURDF("franka_panda/panda.urdf",
                                             basePosition=panda_start_pos,
                                             baseOrientation=panda_start_orientation,
                                             useFixedBase=True)

        # Initialize robot joint positions to prevent collapse
        self._init_robot_poses()

        # Create objects on the table
        self._create_objects()

    def _init_robot_poses(self):
        """Initialize robots to home poses."""

        # KUKA IIWA home pose (neutral position)
        kuka_id = self.robot_ids['kuka']
        kuka_num_joints = p.getNumJoints(kuka_id)
        kuka_home_pose = [0, 0, 0, -1.57, 0, 1.57, 0]  # Safe home position

        for i in range(min(len(kuka_home_pose), kuka_num_joints)):
            p.resetJointState(kuka_id, i, kuka_home_pose[i])

        # Panda home pose
        panda_id = self.robot_ids['panda']
        panda_num_joints = p.getNumJoints(panda_id)
        panda_home_pose = [0, 0, 0, -1.5, 0, 1.8, 0.785, 0.04, 0.04]  # Safe home position

        for i in range(min(len(panda_home_pose), panda_num_joints)):
            p.resetJointState(panda_id, i, panda_home_pose[i])

    def _create_objects(self):
        """Create diverse objects on the table using procedural shapes."""

        # Object 1: Apple (red sphere)
        apple_pos = [0.1, 0.15, self.table_surface_height]
        apple_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[1, 0, 0, 1])
        apple_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        apple_id = p.createMultiBody(baseMass=0.1,
                                     baseCollisionShapeIndex=apple_collision,
                                     baseVisualShapeIndex=apple_visual,
                                     basePosition=apple_pos)

        self.objects[apple_id] = {
            "obj_id": apple_id,
            "name": "apple",
            "category": "fruit",
            "shape": "sphere",
            "color": "red",
            "radius": 0.04
        }

        # Object 2: Banana (yellow capsule/cylinder)
        banana_pos = [-0.1, 0.2, self.table_surface_height]
        banana_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.12, rgbaColor=[1, 1, 0, 1])
        banana_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=0.12)
        banana_id = p.createMultiBody(baseMass=0.08,
                                      baseCollisionShapeIndex=banana_collision,
                                      baseVisualShapeIndex=banana_visual,
                                      basePosition=banana_pos,
                                      baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, 0.3]))

        self.objects[banana_id] = {
            "obj_id": banana_id,
            "name": "banana",
            "category": "fruit",
            "shape": "cylinder",
            "color": "yellow",
            "dimensions": [0.02, 0.02, 0.12]
        }

        # Object 3: Tuna Can (gray cylinder)
        tuna_pos = [0.15, -0.1, self.table_surface_height]
        tuna_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.035, length=0.05, rgbaColor=[0.5, 0.5, 0.5, 1])
        tuna_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.035, height=0.05)
        tuna_id = p.createMultiBody(baseMass=0.15,
                                    baseCollisionShapeIndex=tuna_collision,
                                    baseVisualShapeIndex=tuna_visual,
                                    basePosition=tuna_pos)

        self.objects[tuna_id] = {
            "obj_id": tuna_id,
            "name": "tuna_can",
            "category": "food",
            "shape": "cylinder",
            "color": "gray",
            "dimensions": [0.035, 0.035, 0.05]
        }

        # Object 4: Hinge (small metallic box)
        hinge_pos = [-0.15, -0.15, self.table_surface_height]
        hinge_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.02, 0.01], rgbaColor=[0.7, 0.7, 0.75, 1])
        hinge_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.02, 0.01])
        hinge_id = p.createMultiBody(baseMass=0.05,
                                     baseCollisionShapeIndex=hinge_collision,
                                     baseVisualShapeIndex=hinge_visual,
                                     basePosition=hinge_pos)

        self.objects[hinge_id] = {
            "obj_id": hinge_id,
            "name": "hinge",
            "category": "hardware",
            "shape": "box",
            "color": "metallic",
            "dimensions": [0.06, 0.04, 0.02]  # Full extents
        }

        # Object 5: Orange (orange sphere)
        orange_pos = [0.0, -0.2, self.table_surface_height]
        orange_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.035, rgbaColor=[1, 0.5, 0, 1])
        orange_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.035)
        orange_id = p.createMultiBody(baseMass=0.09,
                                      baseCollisionShapeIndex=orange_collision,
                                      baseVisualShapeIndex=orange_visual,
                                      basePosition=orange_pos)

        self.objects[orange_id] = {
            "obj_id": orange_id,
            "name": "orange",
            "category": "fruit",
            "shape": "sphere",
            "color": "orange",
            "radius": 0.035
        }

        # Let objects settle
        for _ in range(100):
            p.stepSimulation()

    def reset(self):
        """
        Reset the environment.
        Resets robots to home poses and randomizes object positions slightly on the table.
        """
        # Reset robot poses
        self._init_robot_poses()

        # Randomize object positions slightly
        for obj_id in self.objects.keys():
            # Get current position
            pos, orn = p.getBasePositionAndOrientation(obj_id)

            # Add small random offset while keeping objects on table
            new_x = np.random.uniform(-0.2, 0.2)
            new_y = np.random.uniform(-0.2, 0.2)
            new_z = self.table_surface_height + 0.05

            # Random orientation
            new_orn = p.getQuaternionFromEuler([
                np.random.uniform(0, 2*math.pi),
                np.random.uniform(0, 2*math.pi),
                np.random.uniform(0, 2*math.pi)
            ])

            p.resetBasePositionAndOrientation(obj_id, [new_x, new_y, new_z], new_orn)

        # Let objects settle after reset
        for _ in range(100):
            p.stepSimulation()

        print("Environment reset complete!")

    def step(self):
        """Step the simulation forward by one timestep."""
        p.stepSimulation()

    def get_object_info(self, obj_id):
        """
        Get detailed information about a specific object.

        Args:
            obj_id (int): The PyBullet object ID

        Returns:
            dict: Object metadata including position, orientation, and properties
        """
        if obj_id not in self.objects:
            return None

        # Get current position and orientation
        pos, orn = p.getBasePositionAndOrientation(obj_id)

        # Get base metadata
        obj_info = self.objects[obj_id].copy()

        # Update with current state
        obj_info["position"] = list(pos)
        obj_info["orientation"] = list(orn)

        # Calculate bounding box based on shape
        if obj_info["shape"] == "sphere":
            radius = obj_info.get("radius", 0.04)
            obj_info["bounding_box"] = [radius*2, radius*2, radius*2]
        elif obj_info["shape"] in ["cylinder", "box"]:
            obj_info["bounding_box"] = obj_info.get("dimensions", [0.1, 0.1, 0.1])

        return obj_info

    def get_all_objects_info(self):
        """
        Get information about all objects in the environment.

        Returns:
            list: List of dictionaries containing metadata for each object
        """
        all_info = []

        for obj_id in self.objects.keys():
            info = self.get_object_info(obj_id)
            if info:
                all_info.append(info)

        return all_info

    def get_env_state(self):
        """
        Get complete environment state including robots and objects.

        Returns:
            dict: Complete state information
        """
        state = {
            "objects": self.get_all_objects_info(),
            "robots": {}
        }

        # Add robot information
        for robot_name, robot_id in self.robot_ids.items():
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            state["robots"][robot_name] = {
                "id": robot_id,
                "position": list(pos),
                "orientation": list(orn),
                "num_joints": p.getNumJoints(robot_id)
            }

        return state

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect()
        print("Simulation closed!")


if __name__ == "__main__":
    """
    Test execution:
    - Initialize environment
    - Run simulation for 1000 steps
    - Print object metadata
    """

    print("=" * 60)
    print("PyBullet Multi-Robot Simulation Environment Test")
    print("=" * 60)

    # Initialize environment
    env = MultiRobotEnv(gui=True)

    print("\nRunning simulation for 1000 steps...")
    print("(The GUI window should show two robots and objects on a table)")

    # Run simulation
    for i in range(1000):
        env.step()
        time.sleep(1./240.)  # Real-time visualization at 240Hz

        # Print object info every 200 steps
        if (i + 1) % 200 == 0:
            print(f"\nStep {i + 1}/1000")

    print("\n" + "=" * 60)
    print("Simulation Complete - Object Metadata:")
    print("=" * 60)

    # Get and print all object information
    objects_info = env.get_all_objects_info()

    for idx, obj in enumerate(objects_info, 1):
        print(f"\nObject {idx}:")
        print(f"  ID: {obj['obj_id']}")
        print(f"  Name: {obj['name']}")
        print(f"  Category: {obj['category']}")
        print(f"  Position: [{obj['position'][0]:.3f}, {obj['position'][1]:.3f}, {obj['position'][2]:.3f}]")
        print(f"  Orientation: [{obj['orientation'][0]:.3f}, {obj['orientation'][1]:.3f}, {obj['orientation'][2]:.3f}, {obj['orientation'][3]:.3f}]")
        print(f"  Bounding Box: {obj['bounding_box']}")
        print(f"  Shape: {obj['shape']}")
        print(f"  Color: {obj['color']}")

    print("\n" + "=" * 60)
    print("Complete Environment State:")
    print("=" * 60)

    env_state = env.get_env_state()
    print(f"\nTotal Objects: {len(env_state['objects'])}")
    print(f"Robots: {list(env_state['robots'].keys())}")

    for robot_name, robot_info in env_state['robots'].items():
        print(f"\n{robot_name.upper()}:")
        print(f"  Position: [{robot_info['position'][0]:.3f}, {robot_info['position'][1]:.3f}, {robot_info['position'][2]:.3f}]")
        print(f"  Joints: {robot_info['num_joints']}")

    print("\n" + "=" * 60)
    print("Test Complete! Press Ctrl+C or close GUI to exit.")
    print("=" * 60)

    # Keep simulation running until user closes
    try:
        while True:
            env.step()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nShutting down...")
        env.close()
