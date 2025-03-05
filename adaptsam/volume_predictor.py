"""
Initialize a SAM video predictor based on the sam2 model and provide functions to help process a 3D volume.    
"""

import os
import tifffile as tif

import cv2
import numpy as np
import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor


class AdaptSAMPredictor(object):
    def __init__(self, model_cfg, sam2_checkpoint, tmp_dir="cache"):
        # Find the proper device for inference
        self.device = self.setup()
        self.predictor = self.load_predictor(self.device, model_cfg, sam2_checkpoint)

        # Create the tmp dir
        self.tmp_dir = tmp_dir
        os.makedirs(self.tmp_dir, exist_ok=True)

    def clear_tmp(self):
        pass

    def load_predictor(self, device, model_cfg, sam2_checkpoint):
        return build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    def setup(self):
        # select the device for computation
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        return device

    def _infer_half_plane(self, video_dir, X, Y):
        # scan all the JPEG frame names in this directory
        frame_names = [
            p
            for p in os.listdir(video_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split("_")[-1]))

        inference_state = self.predictor.init_state(video_path=video_dir)
        self.predictor.reset_state(inference_state)

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        # Add the initial prompt
        points = np.array([[X, Y]], dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1], np.int32)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            inference_state
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i]).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            if out_mask_logits.max() <= 0:
                break

        out_mat = np.zeros(((len(frame_names),) + out_mask_logits[0].shape[1:]))
        for out_frame_idx in range(len(frame_names)):
            if out_frame_idx in video_segments:
                for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                    mask = self.post_process(out_mask[0])
                    out_mat[out_frame_idx] = np.maximum(out_mat[out_frame_idx], mask)

        return out_mat

    def _assemble_plane(self, left, right):
        # Left is max to 0
        # Right is 0 to max

        plane = np.zeros(((left.shape[0] + right.shape[0],) + left.shape[1:]))
        left = np.flip(left, axis=0)
        plane[: left.shape[0]] = left
        plane[plane.shape[0] - right.shape[0] :] = right
        # T, X, Y -> X, Y, T
        return np.moveaxis(plane, 0, -1)

    def _process_plane(self, plane_dir, X, Y):
        y_left = self._infer_half_plane(os.path.join(plane_dir, "left"), X, Y)
        y_right = self._infer_half_plane(os.path.join(plane_dir, "right"), X, Y)

        y = self._assemble_plane(y_left, y_right)

        return y

    def _generate_plane_dir(self, mat, dim, center):
        # create left dir
        plane_dir = os.path.join(self.tmp_dir, f"plane_{dim}")
        left_dir = os.path.join(plane_dir, "left")
        right_dir = os.path.join(plane_dir, "right")
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)

        mat_cpy = mat.copy()
        mat_cpy = np.moveaxis(mat, dim, 0)
        for i in range(center, 0, -1):
            cv2.imwrite(os.path.join(left_dir, f"{center-i}.jpg"), mat_cpy[i])

        for j in range(center, mat.shape[dim]):
            cv2.imwrite(os.path.join(right_dir, f"{j}.jpg"), mat_cpy[j])

        return plane_dir

    def value_in_range(self, start, end, value):
        return value >= start and value < end

    def point_in_range(self, mat, point):
        assert len(mat.shape) == len(point)
        for i in range(len(mat.shape)):
            if not self.value_in_range(0, mat.shape[i], point[i]):
                return False
        return True

    def predict(self, mat, point_prompt):
        # Clear any existing file in the tmp folder
        self.clear_tmp()

        # Plane 1
        # Generate a video dir for the given plane
        plane_dir = self._generate_plane_dir(mat, dim=2, center=point_prompt[2])
        # Process plane left to right and right to left
        y_plane1 = self._process_plane(plane_dir, X=point_prompt[0], Y=point_prompt[1])

        # Plane 2
        plane_dir = self._generate_plane_dir(mat, dim=0, center=point_prompt[0])
        y_plane2 = self._process_plane(plane_dir, X=point_prompt[1], Y=point_prompt[2])

        # Plane 3
        plane_dir = self._generate_plane_dir(mat, dim=1, center=point_prompt[1])
        y_plane3 = self._process_plane(plane_dir, X=point_prompt[0], Y=point_prompt[2])

        # Rotate plane 2
        # Y,Z,X to X,Y,Z
        y_plane2 = np.moveaxis(y_plane2, (0, 1, 2), (2, 0, 1))

        # Rotate plane 3
        # X,Z,Y to X,Y,Z
        y_plane3 = np.moveaxis(y_plane3, (0, 1, 2), (0, 2, 1))

        planes = np.stack([y_plane1, y_plane2, y_plane3], axis=0)
        predicted = np.mean(planes, axis=0)

        return predicted

    def post_process(self, masks):
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Refine contours
        approx_contours = []
        for contour in contours:
            # Approximate contour
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approx_contours.append(approx)

        # Remove too big contours ( >90% of image size)
        if len(approx_contours) > 1:
            image_size = masks.shape[0] * masks.shape[1]
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            filtered_approx_contours = [
                contour for contour, area in zip(approx_contours, areas) if area < image_size * 0.9
            ]

        # Remove small contours (area < 20% of average area)
        if len(approx_contours) > 1:
            areas = [cv2.contourArea(contour) for contour in approx_contours]
            avg_area = np.mean(areas)

            filtered_approx_contours = [
                contour for contour, area in zip(approx_contours, areas) if area > avg_area * 0.2
            ]
            approx_contours = filtered_approx_contours
        mask_cpy = np.zeros_like(masks)
        for cont in approx_contours:
            cv2.fillPoly(mask_cpy, pts=[cont], color=255)
        return mask_cpy


def keep_center_cc(binary_mask):
    from skimage.measure import label

    labels = label(binary_mask)
    label_id = labels[labels.shape[0] // 2, labels.shape[1] // 2, labels.shape[2] // 2]
    mask = (255 * (labels == label_id)).astype(np.uint8)
    return mask
