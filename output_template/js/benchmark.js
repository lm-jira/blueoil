var em_module = require('./lib_js.js');

const inference_config = `
CLASSES:
- face
DATA_FORMAT: NHWC
IMAGE_SIZE:
- 160
- 160
POST_PROCESSOR:
- FormatYoloV2:
    anchors:
    - - 1.3221
      - 1.73145
    - - 3.19275
      - 4.00944
    - - 5.05587
      - 8.09892
    - - 9.47112
      - 4.84053
    - - 11.2364
      - 10.0071
    boxes_per_cell: 5
    data_format: NHWC
    image_size:
    - 160
    - 160
    num_classes: 1
- ExcludeLowScoreBox:
    threshold: 0.05
- NMS:
    classes:
    - face
    iou_threshold: 0.5
    max_output_size: 100
    per_class: true
PRE_PROCESSOR:
- ResizeWithGtBoxes:
    size:
    - 160
    - 160
- PerImageStandardization: null
TASK: IMAGE.OBJECT_DETECTION
`;


function convertToUint8Array(typedArray) {
  var numBytes = typedArray.length * typedArray.BYTES_PER_ELEMENT;
  var ptr = em_module._malloc(numBytes);
  var heapBytes = new Uint8Array(em_module.HEAPU8.buffer, ptr, numBytes);
  heapBytes.set(new Uint8Array(typedArray.buffer));
  return heapBytes;
}

function uint8ArrayToFloat32Array(a) {
    var nDataBytes = a.length * a.BYTES_PER_ELEMENT;
    var dataHeap = new Float32Array(a.buffer, a.byteOffset, a.length/4);
    dataHeap.set(a.buffer);
    return dataHeap;
}

function uint8ArrayToInt32Array(a) {
    var nDataBytes = a.length * a.BYTES_PER_ELEMENT;
    var dataHeap = new Int32Array(a.buffer, a.byteOffset, a.length/4);
    dataHeap.set(a.buffer);
    return dataHeap;
}

function _freeArray(heapBytes) {
  em_module._free(heapBytes.byteOffset);
}

function init() {
    var network_create = em_module.cwrap("network_create", "number", []);
    var network_init = em_module.cwrap("network_init", "bool", ["number"]);
    var nn = network_create();
    network_init(nn);
    return nn;
}

function  nn_get_input_shape(nn) {
    var network_get_input_rank = em_module.cwrap("network_get_input_rank", "number", ["number"]);
    var network_get_input_shape = em_module.cwrap("network_get_input_shape", "", ["number", "number"]);
    var input_rank = network_get_input_rank(nn);
    var input_shape = new Int32Array(input_rank);
    var input_shape_ = convertToUint8Array(input_shape);
    network_get_input_shape(nn, input_shape_.byteOffset);
    return uint8ArrayToInt32Array(input_shape_);
}

function  nn_get_output_shape(nn) {
    var network_get_output_rank = em_module.cwrap("network_get_output_rank", "number", ["number"]);
    var network_get_output_shape = em_module.cwrap("network_get_output_shape", "", ["number", "number"]);
    var output_rank = network_get_output_rank(nn);
    var output_shape = new Int32Array(output_rank);
    var output_shape_ = convertToUint8Array(output_shape);
    network_get_output_shape(nn, output_shape_.byteOffset);
    return uint8ArrayToInt32Array(output_shape_);
}

function nn_run(nn, input) {
    var output_shape = nn_get_output_shape(nn);
    var output_size = 1;

    for (var i = 0; i < output_shape.length; i++) {
        output_size *= output_shape[i];
    }

    var output = new Float32Array(output_size);
    var input_ = convertToUint8Array(input);
    var output_ = convertToUint8Array(output);

    var network_run = em_module.cwrap("network_run", "", ["number", "number", "number"]);
    var start = Date.now();
    network_run(nn, input_.byteOffset, output_.byteOffset);

    var end = Date.now();
    console.log("elapsed:", (end - start));

    var r = uint8ArrayToFloat32Array(output_);

    _freeArray(input_);
    _freeArray(output_);
    return r;
}

function predictor_create() {
    var predictor_create = em_module.cwrap("predictor_create", "number", []);
    var predictor_configure = em_module.cwrap("predictor_configure", "", ["number", "string"]);
    var predictor = predictor_create();
    predictor_configure(predictor, inference_config);
    return predictor;
}

function predictor_run(predictor, input) {
    var predictor_run = em_module.cwrap("predictor_run", "number", ["number", "number"]);
    var start = Date.now();
    var result = predictor_run(predictor, input);
    var end = Date.now();
    console.log((end - start), "ms");
    tensor_dump(result);

    return result;
}

function tensor_create(shape, data) {
    var tensor_create = em_module.cwrap("tensor_create", "number", ["number", "number", "number"]);
    var shape_i32 = new Int32Array(shape.length);
    for (var i = 0; i < shape_i32.length; i++) {
        shape_i32[i] = shape[i];
    }
    var shape_u8 = convertToUint8Array(shape_i32);
    var data_u8 = convertToUint8Array(data);
    var r = tensor_create(shape_i32.length, shape_u8.byteOffset, data_u8.byteOffset);
    return r;
}

function tensor_get_shape(t) {
    var tensor_get_rank = em_module.cwrap("tensor_get_rank", "number", ["number"]);
    var tensor_get_shape_ = em_module.cwrap("tensor_get_shape", "", ["number", "number"]);
    var rank = tensor_get_rank(t);
    var shape = new Int32Array(rank);
    var shape_ = convertToUint8Array(shape);
    tensor_get_shape_(t, shape_.byteOffset);
    shape = uint8ArrayToInt32Array(shape_);
    return shape;
}

function tensor_data(t) {
    var tensor_data = em_module.cwrap("tensor_data", "number", ["number", "number"]);
    shape = tensor_get_shape(t);
    var size = 1;
    for (var i = 0; i < shape.length; i++) {
        size *= shape[i];
    }

    var buf = new Float32Array(size);
    buf_ = convertToUint8Array(buf);
    tensor_data(t, buf_.byteOffset);
    var r = uint8ArrayToFloat32Array(buf_);
    return r;
}

function tensor_dump(t) {
    shape = tensor_get_shape(t);
    console.log("shape:", shape);
    var data = tensor_data(t);

    for (var i = 0; i < shape[1]; i++) {
        var x = data[i*shape[2]];
        var y = data[i*shape[2]+1];
        var w = data[i*shape[2]+2];
        var h = data[i*shape[2]+3];
        var class_ = data[i*shape[2]+4];
        var score = data[i*shape[2]+5];
//        console.log(x,y,w,h,class_,score);
    }
}

// main function, read from here
em_module['onRuntimeInitialized'] = onRuntimeInitialized;
function onRuntimeInitialized() {
    var nn = init();
    var input_size = 1;
    var input_shape = nn_get_input_shape(nn);
    for (var i = 0; i < input_shape.length; i++) {
        input_size *= input_shape[i];
    }

    var input = new Float32Array(input_size);
    // expects RGBRGBRGB...
    for (var i = 0; i < input_size; i++) {
        var r = Math.random();
        input[i] = r;
    }

    var t = tensor_create(input_shape, input);
    var predictor = predictor_create();

    var start = Date.now();
    var trial = 100;
    for (var i = 0; i < trial; i++) {
        var r = predictor_run(predictor, t);
//        tensor_dump(r);
    }
    var end = Date.now();

    console.log((end - start)/trial, "ms on average");
}
