import rclpy
from rclpy.node import Node
import torch
from model.model import Model
from rclpy.action import ActionClient
from snaak_manipulation.action import ExecutePolicy, PlaceInBin, ReturnHome, ExecuteTrajectory
from sensor_msgs.msg import Image
from snaak_weight_read.srv import ReadWeight
from dynamixel_sdk_custom_interfaces.msg import SetPosition
from rclpy.task import Future
from rclpy.qos import qos_profile_sensor_data
import time
from rclpy.task import Future
from cv_bridge import CvBridge
from action_msgs.msg import GoalStatus
import numpy as np

MODEL_PATH = "../model/model.pth"

class ActionPlanner(Node):
    def __init__(self):
        super().__init__('action_planner')

        self.get_logger().info("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(MODEL_FILEPATH).to(self.device)
        self.model.eval()

        self.action_dim = 6
        self.bridge = CvBridge()

        # Clients
        self.left_weight_bins_client = self.create_client(
            ReadWeight, "/snaak_weight_read/snaak_scale_bins_left/read_weight"
        )
        self.right_weight_bins_client = self.create_client(
            ReadWeight, "/snaak_weight_read/snaak_scale_bins_right/read_weight"
        )

        self.reset_arm_client = ActionClient(self, ReturnHome, "/snaak_manipulation/return_home")
        self.execute_trajectory_client = ActionClient(self, ExecuteTrajectory, "/snaak_manipulation/execute_trajectory")
        self.execute_policy_client = ActionClient(self, ExecutePolicy, '/snaak_manipulation/execute_policy')
        self.place_in_bin_client = ActionClient(self, PlaceInBin, '/snaak_manipulation/place_in_bin')

        self.set_position_publisher = self.create_publisher(SetPosition, '/dynamixel/set_position', 10)
        msg = SetPosition()
        msg.id = 1
        msg.position = 1030
        self.set_position_publisher.publish(msg)
        self.prev_action = np.zeros(self.action_dim)
        time.sleep(0.5)
        self.reset_arm()

    def reset_arm(self):
        self.reset_arm_client.wait_for_server()
        goal_msg = ReturnHome.Goal()
        send_goal_future = self.reset_arm_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Return Home Goal Rejected.")
            raise RuntimeError("Return Home Goal Rejected.")

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        result = get_result_future.result()
        if not result.status == 3:  # GoalStatus.STATUS_SUCCEEDED
            self.get_logger().error("Return Home Goal Failed.")
            raise RuntimeError("Return Home Goal Failed.")

    def load_model(self, checkpoint_path):
        model = Model()
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    def get_action(self, rgb, depth, weight_curr, weight_goal, method="cem"):
        if method == "gd":
            action = torch.tensor(self.prev_action, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
            opt = torch.optim.Adam([action], lr=0.05)
            for _ in range(50):
                opt.zero_grad()
                weight_pred = self.model(rgb, depth, weight_curr, action)
                loss = ((weight_pred - weight_goal) ** 2).mean()
                loss.backward()
                opt.step()
            return action.detach().cpu().numpy()

        elif method == "cem":
            N, K, iters = 256, 32, 5
            mu = torch.from_numpy(self.prev_action).to(self.device)
            sigma = torch.ones_like(mu) * 0.5
            for _ in range(iters):
                samples = mu + sigma * torch.randn(N, self.action_dim, device=self.device)
                costs = []
                for s in samples:
                    pred = self.model(rgb, depth, weight_curr, s.unsqueeze(0))
                    cost = ((pred - weight_goal) ** 2).item()
                    costs.append(cost)
                costs = torch.tensor(costs, device=self.device)
                elite_idx = costs.topk(K, largest=False).indices
                elites = samples[elite_idx]
                mu, sigma = elites.mean(dim=0), elites.std(dim=0)
            return mu.detach().cpu().numpy()

        else:
            raise ValueError(f"Unknown method: {method}")

    def get_latest_image(self, topic_name, timeout_sec=2.0):
        """Grab the most recent message from a topic."""
        future = Future()
        subscription = self.create_subscription(
            Image, topic_name, lambda msg: future.set_result(msg), qos_profile_sensor_data
        )
        try:
            rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        except Exception as e:
            self.get_logger().error(f"Failed to get message from {topic_name}: {e}")
            return None
        finally:
            self.destroy_subscription(subscription)
        return future.result()

    def perform_test(self, pick_bin, place_bin, desired_pick_amount):

        rgb_msg = self.get_latest_image("/camera/camera/color/image_rect_raw")
        depth_msg = self.get_latest_image("/camera/camera/depth/image_rect_raw")

        if rgb_msg is None or depth_msg is None:
            self.get_logger().error("Could not get images for perform_test")
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().to(self.device)

        read_weight = ReadWeight.Request()

        if (pick_bin == 4) or (pick_bin == 5) or (pick_bin == 6):
            future = self.left_weight_bins_client.call_async(read_weight)
        else:
            future = self.right_weight_bins_client.call_async(read_weight)        
        rclpy.spin_until_future_complete(self, future)
        weight_start = future.result().weight
        weight_curr = torch.tensor([weight_start], device=self.device)
        weight_goal = torch.tensor([weight_curr - desired_pick_amount], device=self.device)

        action = self.get_action(rgb_tensor, depth_tensor, weight_curr, weight_goal, method="cem")
        self.prev_action = action

        # Execute Policy
        goal_msg = ExecutePolicy.Goal()
        goal_msg.actions = action
        goal_msg.bin_id = pick_bin

        send_goal_future = self.execute_policy_client.send_goal_async(
            goal_msg,
        )
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("ExecutePolicy goal rejected.")
            raise RuntimeError("Execute Policy Goal Rejected")


        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result()
        if not result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"Execute Policy Failed")
            raise RuntimeError("Execute Policy Failed")
        
        # Place ingredients back in bin
        goal_msg = PlaceInBin.Goal()
        goal_msg.bin_id = place_bin

        send_goal_future = self.place_in_bin_client.send_goal_async(
            goal_msg,
        )
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Place Ingredients Goal Rejected.")
            raise RuntimeError("Place Ingredients Goal Rejected.")

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result = get_result_future.result()
        if not result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error("Place Ingredients Failed.")
            raise RuntimeError("Place Ingredients Failed.")
        

        read_weight = ReadWeight.Request()

        if (pick_bin == 4) or (pick_bin == 5) or (pick_bin == 6):
            future = self.left_weight_bins_client.call_async(read_weight)
        else:
            future = self.right_weight_bins_client.call_async(read_weight)        
        rclpy.spin_until_future_complete(self, future)
        
        weight_end  = future.result().weight

        print("---------------------------------------")
        print(f"Desired Pickup: {desired_pick_amount}")
        print(f"Actual Pickup: {weight_start - weight_end}")
