// Ready for init blueoil
let nn
let inputShape

// Init app
const video = document.getElementById('video')
const canvas = document.getElementById('canvas')
const ctx = canvas.getContext('2d')
// const canvasCopy = document.getElementById("canvascopy");
// const copyContext = canvasCopy.getContext('2d');

const videoWidth = video.offsetWidth
const videoHeight = video.offsetHeight

canvas.width  = 160;
canvas.height = 160;

canvas.setAttribute("width", 160);
canvas.setAttribute("height", 160);


const params = {
    format_yolo_v2: {
        image_size: [160, 160],
        num_classes: 1,
        anchors: [[0.25, 0.25], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5], [2.0, 2.0]],
        boxes_per_cell: 5,
        data_format: "NHWC",
        threshold: 0.2,
    },
    classes: ["face"],
    iou_threshold: 0.5,
}


const format_yolo_v2 = new FormatYoloV2(...Object.values(params.format_yolo_v2))
const nms = new NMS(params.classes, params.iou_threshold)
var counter = 0;

const update = () => {
    // Get image data
    const imageData = ctx.getImageData(0, 0, 160, 160)

    let input_size = inputShape.reduce((x, y) => {return x*y})
    let input = new Float32Array(input_size)

    // divid255
    j = 0;
    for (var i = 0; i < imageData.data.length; i++) {
        input[j] = parseFloat(imageData.data[i]) / 255.0;
        if ((i % 4) != 0 ){
            j++;
        }
    }

    var boxes = new Array()

    var result = nn_run(nn, input)
    boxes = format_yolo_v2.run(result);
    boxes = nms.run(boxes)
    ctx.drawImage(video,
                  (video.videoWidth - video.videoHeight)/2, 0, video.videoHeight, video.videoHeight,
                  0, 0, 160, 160)
    for (let box of boxes){
        ctx.strokeStyle = "rgb(200, 0, 0)";
        ctx.strokeRect(box[0], box[1], box[2], box[3])
    }

    // debug
    counter++;
    if( counter > 1000 ) {
        return;
    }

    window.requestAnimationFrame(update)
}

const main = async () => {
    // Init Blueoil
    nn = nn_init()
    inputShape = nn_get_input_shape(nn)

    // Init cra
    navigator.mediaDevices = navigator.mediaDevices || ((navigator.mozGetUserMedia || navigator.webkitGetUserMedia) ? {
        getUserMedia: function(c) {
            return new Promise(function(y, n) {
                (navigator.mozGetUserMedia || navigator.webkitGetUserMedia).call(navigator, c, y, n)
            })
        }
    } : null)

    if (!navigator.mediaDevices) {
        alert('getUserMedia is not available in your browser')
        throw new Error('getUserMedia is not available in your browser')
    }

    const constraints = { audio: false, video: true }
    const stream = await navigator.mediaDevices.getUserMedia(constraints)

    video.srcObject = stream
    update()
}

const createResult = () => {
    const x = Math.random() * videoWidth / 2
    const y = Math.random() * videoHeight / 2
    const w = x + Math.random() * videoWidth
    const h = y + Math.random() * videoHeight
    return {
        coordinates: [x, y, w, h]
    }
}

const predict = () => {

}

document.addEventListener("DOMContentLoaded", function(event) {
    Module['onRuntimeInitialized'] = main
});
