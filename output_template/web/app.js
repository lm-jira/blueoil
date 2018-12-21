// Ready for init blueoil
let nn
let predictor
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

var counter = 0;

const update = () => {
    // Get image data
    const imageData = ctx.getImageData(0, 0, 160, 160)
    const image_size = [160, 160];
    let input_size = inputShape.reduce((x, y) => {return x*y});

    var start = Date.now();

    const rgba_data = imageData.data;
    var rgb_data = new Float32Array(input_size);
    var j = 0;
    for (var i = 0; i < rgba_data.length; i+=4) {
        rgb_data[j] = rgba_data[i];
        rgb_data[j+1] = rgba_data[i+1];
        rgb_data[j+2] = rgba_data[i+2];
        j += 3;
    }
    let input = rgb_data;
    var end = Date.now();
    console.log((end - start), "ms for copy");

    start = Date.now();
    var t = tensor_create(inputShape, input);

    var result = predictor_run(predictor, t);
    tensor_delete(t);
    end = Date.now();
    console.log((end - start), "ms for predictor_run");

    ctx.drawImage(video,
                  (video.videoWidth - video.videoHeight)/2, 0, video.videoHeight, video.videoHeight,
                  0, 0, 160, 160)

    const result_shape = tensor_get_shape(result);
    const result_data = tensor_data(result);

    var num = 0;
    for(var i = 0; i < result_shape[1]; i++) {
        var x = result_data[i*result_shape[2]];
        var y = result_data[i*result_shape[2]+1];
        var w = result_data[i*result_shape[2]+2];
        var h = result_data[i*result_shape[2]+3];
        var class_ = result_data[i*result_shape[2]+4];
        var score = result_data[i*result_shape[2]+5];
        if (score > 0) {
            ctx.strokeStyle = "rgb(200, 0, 0)";
            ctx.strokeRect(x*image_size[0], y*image_size[1], w*image_size[0], h*image_size[1]);
            num++;
        }
    }

    tensor_delete(result);

    // debug
    counter++;
    if( counter > 1000 ) {
        return;
    }

    window.requestAnimationFrame(update)
}

const main = async () => {
    // Init Blueoil
    nn = nn_init();
    inputShape = nn_get_input_shape(nn)
    predictor = predictor_create();

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

    setTimeout(update(), 1000);
}

document.addEventListener("DOMContentLoaded", function(event) {
    Module['onRuntimeInitialized'] = main
});
