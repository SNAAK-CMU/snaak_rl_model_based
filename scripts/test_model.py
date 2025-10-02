#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import torch
from model import Model
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
import torchvision.transforms.functional as TF
import cv2

WEIGHT_MEAN = 104.62
WEIGHT_STD = 61.09
ACTION_MEAN = np.array([-0.0002, 0.0003, 0.0021, 0.0007, 0.0019, -0.0328]) # uniform sampling
ACTION_STD = np.array([0.0193, 0.0540, 0.0317, 0.0193, 0.0539, 0.0147])
IMAGENET_MEAN = [0.485, 0.456, 0.406] # keep these for now, in future can recompute with own dataset
IMAGENET_STD  = [0.229, 0.224, 0.225]
DEPTH_MEAN = [337.65]
DEPTH_STD = [66.147]
MODEL_PATH = "/home/snaak/Documents/manipulation_ws/src/snaak_rl_model_based/model/model.pth"

BIN_WIDTH = 0.140
BIN_LENGTH = 0.240
BIN_DEPTH = 0.046
A1_MAX_HEIGHT = 0.050
ACTION_MEAN = np.array([-0.0002, 0.0003, 0.0021, 0.0007, 0.0019, -0.0328]) # uniform sampling
ACTION_STD = np.array([0.0193, 0.0540, 0.0317, 0.0193, 0.0539, 0.0147])

BIN2_XMIN = 250
BIN2_YMIN = 0
BIN2_XMAX = 470
BIN2_YMAX = 340

# Bin1 coords
BIN1_XMIN = 240
BIN1_YMIN = 0
BIN1_XMAX = 450
BIN1_YMAX = 330

# Bin3 coords 
BIN3_XMIN = 355 
BIN3_YMIN = 0 
BIN3_XMAX = 565  
BIN3_YMAX = 330 

# Bin 4 coords
BIN4_XMIN = 81
BIN4_YMIN = 66
BIN4_XMAX = 440
BIN4_YMAX = 297

# Bin 5 coords
BIN5_XMIN = 67
BIN5_YMIN = 72
BIN5_XMAX = 434
BIN5_YMAX = 305

# Bin 6 coords
BIN6_XMIN = 69
BIN6_YMIN = 94
BIN6_XMAX = 433
BIN6_YMAX = 316

# BIN_COORDS
BIN_COORDS = [
    [BIN1_XMIN, BIN1_YMIN, BIN1_XMAX, BIN1_YMAX],
    [BIN2_XMIN, BIN2_YMIN, BIN2_XMAX, BIN2_YMAX],
    [BIN3_XMIN, BIN3_YMIN, BIN3_XMAX, BIN3_YMAX],
    [BIN4_XMIN, BIN4_YMIN, BIN4_XMAX, BIN4_YMAX],
    [BIN5_XMIN, BIN5_YMIN, BIN5_XMAX, BIN5_YMAX],
    [BIN6_XMIN, BIN6_YMIN, BIN6_XMAX, BIN6_YMAX],
]
class ActionPlanner(Node):
    def __init__(self):
        super().__init__('action_planner')

        self.get_logger().info("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = self.load_model(MODEL_PATH).to(self.device)
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
        self.start = True

    def crop(self, image, bin_number, rotate=False):
        xmin, ymin, xmax, ymax = BIN_COORDS[bin_number-1]
        cropped = image[ymin:ymax, xmin:xmax]
        # print(f"Cropping to bin {bin_number} with coords: ({xmin}, {ymin}), ({xmax}, {ymax}), rotation: {rotate}, shape : {cropped.shape}")
        if rotate:
            #print("Rotating image for bin", bin_number)
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)
        cropped = cv2.resize(cropped, (224, 340))
        #cv2.imshow("Image", cropped)
        #print(cropped.shape)

        #cv2.waitKey(0)
        return cropped

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
        if not GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error("Return Home Goal Failed.")
            raise RuntimeError("Return Home Goal Failed.")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location="cpu")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint  # assume it's a raw state_dict

        model = Model()  # your actual model architecture
        model.load_state_dict(state_dict)
        return model
    
    def get_action(self, rgb, depth, weight_curr, weight_goal, method="cem"):
        if method == "gd":
            action = np.random.randn(self.action_dim) * ACTION_STD + ACTION_MEAN
            action = torch.tensor(self.prev_action, dtype=torch.float32, device=self.device).unsqueeze(0)
            action = action.detach().requires_grad_() # need to make sure tensor is leaf (created by user)
            opt = torch.optim.Adam([action], lr=0.05)
            for _ in range(50):
                opt.zero_grad()
                weight_pred = self.model(rgb, depth, weight_curr, action)
                loss = ((weight_pred - weight_goal) ** 2).mean()
                loss.backward()
                opt.step()
            return action.detach().cpu().numpy()

        elif method == "cem":
            N, K, iters = 256, 32, 15
            #mu = torch.from_numpy(self.prev_action).to(self.device).float()
            mu = torch.zeros(self.action_dim, device=self.device)
            sigma = torch.ones_like(mu) * 0.5
            samples = mu + sigma * torch.randn(N, self.action_dim, device=self.device).float()
            for _ in range(iters):
                print(f"Step: {_}")
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
        if self.start:
            self.execute_trajectory_client.wait_for_server()
            goal_msg = ExecuteTrajectory.Goal()
            goal_msg.desired_location = f"bin{pick_bin}"
            send_goal_future = self.execute_trajectory_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, send_goal_future)

            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Execute Trajectory Goal Rejected.")
                raise RuntimeError("Execute Trajectory Goal Rejected.")
            get_result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, get_result_future)
            result = get_result_future.result()
            if not result.status == GoalStatus.STATUS_SUCCEEDED:
                self.get_logger().error("Execute Trajectory Goal Failed.")
                raise RuntimeError("Execute Trajectory Goal Failed.")
            self.start = False
        rgb_msg = self.get_latest_image("/camera/camera/color/image_rect_raw")
        depth_msg = self.get_latest_image("/camera/camera/depth/image_rect_raw")

        if rgb_msg is None or depth_msg is None:
            self.get_logger().error("Could not get images for perform_test")
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        rotate = ((pick_bin - 1) // 3 > 0)
        rgb = self.crop(rgb, pick_bin, rotate=rotate)
        depth = self.crop(depth, pick_bin, rotate=rotate)

        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        rgb_tensor = rgb_tensor / 255.0  # scale to [0,1]
        rgb_tensor = TF.normalize(rgb_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)

         # For depth (likely 2D): add channel dim -> (1, 1, H, W)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float().to(self.device)
        depth_tensor = TF.normalize(depth_tensor, DEPTH_MEAN, DEPTH_STD)
        read_weight = ReadWeight.Request()

        if (pick_bin == 4) or (pick_bin == 5) or (pick_bin == 6):
            future = self.left_weight_bins_client.call_async(read_weight)
        else:
            future = self.right_weight_bins_client.call_async(read_weight)        
        rclpy.spin_until_future_complete(self, future)
        weight_start = future.result().weight.data
        weight_curr = torch.tensor([float(weight_start)], dtype=torch.float32, device=self.device).unsqueeze(0)
        weight_curr = (weight_curr - WEIGHT_MEAN) / WEIGHT_STD
        weight_goal = torch.tensor([weight_curr.item() - desired_pick_amount], dtype=torch.float32, device=self.device).unsqueeze(0)
        weight_goal = (weight_goal - WEIGHT_MEAN) / WEIGHT_STD

        action = self.get_action(rgb_tensor, depth_tensor, weight_curr, weight_goal, method="gd")
        action = action * ACTION_STD + ACTION_MEAN  # unnormalize
        action = action.flatten()
        action[:3] = np.clip(action[:3], -np.array([BIN_LENGTH/2, BIN_WIDTH/2, BIN_DEPTH]), np.array([BIN_LENGTH/2, BIN_WIDTH/2, A1_MAX_HEIGHT]))
        action[3:] = np.clip(action[3:], -np.array([BIN_LENGTH/2, BIN_WIDTH/2, BIN_DEPTH]), np.array([BIN_LENGTH/2, BIN_WIDTH/2, 0]))
        self.prev_action = action

        # Execute Policy
        goal_msg = ExecutePolicy.Goal()
        goal_msg.actions = action.tolist()
        self.get_logger().info(f"Executing action: {goal_msg.actions}")
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
        
        weight_end  = future.result().weight.data

        print("---------------------------------------")
        print(f"Desired Pickup: {desired_pick_amount}")
        print(f"Actual Pickup: {weight_start - weight_end}")

def main(args=None):
    rclpy.init(args=args)

    try:
        planner = ActionPlanner()

        while True:
            #try:
            pick_bin = int(input("Enter pick bin ID (int): "))
            place_bin = int(input("Enter place bin ID (int): "))
            desired_amount = float(input("Enter desired pickup amount (float): "))
        #except ValueError:
            #print("Invalid input. Please enter numeric values.")
            #continue

            #try:
            planner.perform_test(pick_bin, place_bin, desired_amount)
            # except Exception as e:
            #     print(f"Error during perform_test: {e}")

            # cont = input("Do you want to run another test? (y/n): ").lower()
            # if cont != 'y':
            #     break

    finally:
        planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()