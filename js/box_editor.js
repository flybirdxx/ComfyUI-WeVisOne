import { app } from "../../scripts/app.js";
import { watchBoxPreviews } from "./box_preview.js";

const EDITOR = "WeVisOneBoxEditor";
const TARGETS = new Set(["WeVisOnePerception", "WeVisOneMask2Alpha"]);

export function connectBoxEditor(target, appInstance, createNode) {
    const graph = target.graph;
    const imageLink = target.getInputLink(target.findInputSlot("images"));
    const source = imageLink && graph?.getNodeById(imageLink.origin_id);
    if (!source) throw new Error("请先连接 WeVisOne 的 images 输入，再添加框选节点。");
    const boxSlot = target.findInputSlot("bboxes");
    let editor = target.getInputNode(boxSlot);
    if (editor?.comfyClass !== EDITOR && editor?.type !== EDITOR) {
        editor = createNode(EDITOR);
        if (!editor) throw new Error("图片框选节点未加载，请重启 ComfyUI 并刷新页面。");
        const nearby = graph.nodes.filter(n => Math.abs(n.pos[0] - source.pos[0]) < 440);
        editor.pos = [source.pos[0], Math.max(...nearby.map(n => n.pos[1] + n.size[1]), source.pos[1]) + 60];
        graph.add(editor);
    }
    source.connect(imageLink.origin_slot, editor, editor.findInputSlot("background"));
    editor.connect(editor.findOutputSlot("bboxes"), target, boxSlot);
    appInstance.canvas.selectNode(editor);
    appInstance.canvas.centerOnNode(editor);
    graph.setDirtyCanvas(true, true);
    return editor;
}

app.registerExtension({
    name: "WeVisOne.BoxEditor",
    setup() {
        watchBoxPreviews();
    },
    nodeCreated(node) {
        const name = node.comfyClass ?? node.constructor.comfyClass;
        if (name === EDITOR) {
            node.properties.bboxGrid ??= false;
            node.setSize([Math.max(node.size[0], 480), Math.max(node.size[1], 640)]);
        }
        if (!TARGETS.has(name)) return;
        node.addWidget("button", "添加／定位图片框选", null, () => {
            try {
                connectBoxEditor(node, app, type => window.LiteGraph.createNode(type));
            } catch (error) {
                app.extensionManager.toast.add({severity: "warn", summary: "WeVisOne 图片框选", detail: error.message, life: 5000});
            }
        }, {serialize: false});
    },
});
