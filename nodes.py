import json

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from .backend import run_inference
from .model_paths import MODEL_ROOT, discover_models, resolve_model, sam2_models
from .contracts import TASKS, boxes_per_image, parse_points


class WeVisOnePerception(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WeVisOnePerception", display_name="WeVisOne · 统一视觉感知", category="WeVisOne",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("model_name", options=list(discover_models()) or ["请先下载 WeVisOne-VL 模型"]),
                io.Combo.Input("task", options=list(TASKS), default="matting"),
                io.String.Input("query", default="person", multiline=True, tooltip="类别每行一个；指代表达完整输入。freeform 时为原始提示词。"),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True, socketless=False, tooltip="原生像素框。visual_prompt 查找相似物体；box_matting 对给定框抠图。"),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto", advanced=True),
                io.Combo.Input("dtype", options=["bfloat16", "float16", "float32"], default="bfloat16", advanced=True),
                io.Int.Input("max_pixels", default=1024 * 1024, min=16 * 1024, max=16777216, step=1024, advanced=True),
                io.Int.Input("max_new_tokens", default=4096, min=1, max=32768, advanced=True),
                io.Int.Input("decoder_batch_size", default=1, min=1, max=128, advanced=True),
                io.Float.Input("temperature", default=0.0, min=0.0, max=2.0, step=0.05, advanced=True),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=True, advanced=True),
            ],
            outputs=[
                io.Image.Output("preview"),
                io.Mask.Output("mask", tooltip="每张输入一张合并掩码；白色为前景；matting 保留连续 alpha。"),
                io.BoundingBox.Output("bboxes"),
                io.String.Output("response", is_output_list=True),
                io.String.Output("result_json", is_output_list=True),
                io.Mask.Output("instance_masks", is_output_list=True, tooltip="每张输入对应一个实例 MASK batch；无实例时为空 batch，实际数量见 JSON。"),
            ],
        )

    @classmethod
    def execute(cls, images, model_name, task, query, bboxes=None, device="auto", dtype="bfloat16",
                max_pixels=1024 * 1024, max_new_tokens=4096, decoder_batch_size=1, temperature=0.0, seed=0):
        if task not in TASKS:
            raise ValueError(f"未知任务：{task}")
        boxes = boxes_per_image(bboxes, len(images))
        if task in {"visual_prompt", "box_matting"} and any(not item for item in boxes):
            raise ValueError(f"{task} 需要为每张图像连接至少一个 bboxes。")
        path = resolve_model(model_name)
        request = dict(kind="vl", model=str(path), task=task, query=query, boxes=boxes,
                       device=device, dtype=dtype, max_pixels=max_pixels, max_new_tokens=max_new_tokens,
                       decoder_batch_size=decoder_batch_size, temperature=temperature, seed=seed)
        preview, alpha, instances, records = run_inference(request, images)
        return io.NodeOutput(preview, alpha, [r["bboxes"] for r in records],
                             [r["response"] for r in records],
                             [json.dumps(r, ensure_ascii=False, indent=2) for r in records], instances)


class WeVisOneMask2Alpha(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WeVisOneMask2Alpha", display_name="WeVisOne · Mask2Alpha 精细抠图", category="WeVisOne",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("sam2_checkpoint", options=list(sam2_models()) or ["请先下载官方 SAM2.1-B+ checkpoint"]),
                io.Combo.Input("mode", options=["mask", "bbox", "point"], default="mask"),
                io.Mask.Input("mask", optional=True),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True, socketless=False),
                io.String.Input("positive_coords", default="[]", optional=True, tooltip='KJNodes PointsEditor 格式：[{"x": 100, "y": 200}]'),
                io.String.Input("negative_coords", default="[]", optional=True),
                io.Float.Input("mask_threshold", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto", advanced=True),
                io.Combo.Input("precision", options=["bf16", "fp32"], default="bf16", advanced=True),
            ],
            outputs=[io.Mask.Output("alpha"), io.Mask.Output("prompt_mask"), io.String.Output("result_json", is_output_list=True)],
        )

    @classmethod
    def execute(cls, images, sam2_checkpoint, mode, mask=None, bboxes=None, positive_coords="[]",
                negative_coords="[]", mask_threshold=0.0, device="auto", precision="bf16"):
        if mode not in {"mask", "bbox", "point"}:
            raise ValueError(f"未知提示模式：{mode}")
        checkpoint = sam2_models().get(sam2_checkpoint)
        head = MODEL_ROOT / "Mask2Alpha/mask2alpha_sam2_bplus.pt"
        if checkpoint is None or not head.is_file():
            raise FileNotFoundError("缺少官方 SAM2.1-B+ 或 Mask2Alpha head。请按 README 的模型下载说明放置权重。")
        boxes, points, labels = None, None, None
        if mode == "mask":
            if mask is None:
                raise ValueError("mask 模式需要连接 MASK（白色为前景）。")
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if len(mask) == 1:
                mask = mask.expand(len(images), -1, -1)
            if len(mask) != len(images):
                raise ValueError("MASK 批次数必须为 1 或与图像批次相同。")
            mask = F.interpolate(mask[:, None].float(), images.shape[1:3], mode="nearest")[:, 0]
            mask = (mask > mask_threshold).float()
            if not torch.all(mask.flatten(1).any(dim=1)):
                raise ValueError("Mask2Alpha 不接受空前景掩码。")
        elif mode == "bbox":
            boxes = boxes_per_image(bboxes, len(images))
            if any(len(item) != 1 for item in boxes):
                raise ValueError("Mask2Alpha bbox 模式每张图像需要一个框；多实例请用列表映射。")
        else:
            points, labels = parse_points(positive_coords, negative_coords)
        request = dict(kind="mask2alpha", model=str(checkpoint), head=str(head), mode=mode,
                       boxes=boxes, points=points, labels=labels, device=device, precision=precision)
        _, alpha, prompts, records = run_inference(request, images, mask if mode == "mask" else None)
        return io.NodeOutput(alpha, torch.cat(prompts), [json.dumps(r, ensure_ascii=False) for r in records])
