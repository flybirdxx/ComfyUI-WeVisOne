"""Reuse ComfyUI's BOUNDING_BOXES canvas and coordinate conversion helpers."""

import nodes
from comfy_api.latest import io
from comfy_extras.nodes_bounding_boxes import boxes_to_regions, fractions_to_bbox_frame


class WeVisOneBoxEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="WeVisOneBoxEditor",
            display_name="WeVisOne · 图片框选",
            category="WeVisOne",
            description="连接图片后直接拖拽画框；拖动边缘调整大小，Delete 删除选中框。坐标自动映射回原图。",
            inputs=[
                io.Image.Input("background", display_name="图像"),
                io.Int.Input("width", default=1024, min=16, max=16384, step=16),
                io.Int.Input("height", default=1024, min=16, max=16384, step=16),
                io.BoundingBoxes.Input("editor_state", default=[],
                                       tooltip="在图片上拖拽框选；可画多个框。无需填写文字、描述或颜色。"),
            ],
            outputs=[io.BoundingBox.Output("bboxes")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, background, width, height, editor_state):
        if width <= 0 or height <= 0:
            raise ValueError("框选画布尺寸必须大于 0。")
        image_height, image_width = background.shape[1:3]
        regions = boxes_to_regions(editor_state or [], width, height)
        boxes = fractions_to_bbox_frame(regions, image_width, image_height)
        saved = nodes.PreviewImage().save_images(background[:1], "wevisone.boxes.background")
        return io.NodeOutput(boxes, ui={"background_images": saved["ui"]["images"]})
