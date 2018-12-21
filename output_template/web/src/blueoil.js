//var Module = require('../lib_js.js');

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
  var ptr = Module._malloc(numBytes);
  var heapBytes = new Uint8Array(Module.HEAPU8.buffer, ptr, numBytes);
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
  Module._free(heapBytes.byteOffset);
}

function nn_init() {
    var network_create = Module.cwrap("network_create", "number", []);
    var network_init = Module.cwrap("network_init", "bool", ["number"]);
    var nn = network_create();
    network_init(nn);
    return nn;
}

function  nn_get_input_shape(nn) {
    var network_get_input_rank = Module.cwrap("network_get_input_rank", "number", ["number"]);
    var network_get_input_shape = Module.cwrap("network_get_input_shape", "", ["number", "number"]);
    var input_rank = network_get_input_rank(nn);
    var input_shape = new Int32Array(input_rank);
    var input_shape_ = convertToUint8Array(input_shape);
    network_get_input_shape(nn, input_shape_.byteOffset);
    return uint8ArrayToInt32Array(input_shape_);
}

function  nn_get_output_shape(nn) {
    var network_get_output_rank = Module.cwrap("network_get_output_rank", "number", ["number"]);
    var network_get_output_shape = Module.cwrap("network_get_output_shape", "", ["number", "number"]);
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

    var network_run = Module.cwrap("network_run", "", ["number", "number", "number"]);
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
    var predictor_create = Module.cwrap("predictor_create", "number", []);
    var predictor_configure = Module.cwrap("predictor_configure", "", ["number", "string"]);
    var predictor = predictor_create();
    predictor_configure(predictor, inference_config);
    return predictor;
}

function predictor_run(predictor, input) {
    var predictor_run = Module.cwrap("predictor_run", "number", ["number", "number"]);
    var start = Date.now();
    var result = predictor_run(predictor, input);
    var end = Date.now();
    tensor_dump(result);

    return result;
}

function tensor_create(shape, data) {
    var tensor_create = Module.cwrap("tensor_create", "number", ["number", "number", "number"]);
    var shape_i32 = new Int32Array(shape.length);
    for (var i = 0; i < shape_i32.length; i++) {
        shape_i32[i] = shape[i];
    }
    var shape_u8 = convertToUint8Array(shape_i32);
    var data_u8 = convertToUint8Array(data);
    var r = tensor_create(shape_i32.length, shape_u8.byteOffset, data_u8.byteOffset);
    _freeArray(shape_u8);
    _freeArray(data_u8);
    return r;
}

function tensor_delete(t) {
    var func = Module.cwrap("tensor_delete", "", ["number"]);
    func(t);
}

function tensor_get_shape(t) {
    var tensor_get_rank = Module.cwrap("tensor_get_rank", "number", ["number"]);
    var tensor_get_shape_ = Module.cwrap("tensor_get_shape", "", ["number", "number"]);
    var rank = tensor_get_rank(t);
    var shape = new Int32Array(rank);
    var shape_ = convertToUint8Array(shape);
    tensor_get_shape_(t, shape_.byteOffset);
    shape = uint8ArrayToInt32Array(shape_);
    return shape;
}

function tensor_data(t) {
    var tensor_data = Module.cwrap("tensor_data", "number", ["number", "number"]);
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
    var data = tensor_data(t);

    for (var i = 0; i < shape[1]; i++) {
        var x = data[i*shape[2]];
        var y = data[i*shape[2]+1];
        var w = data[i*shape[2]+2];
        var h = data[i*shape[2]+3];
        var class_ = data[i*shape[2]+4];
        var score = data[i*shape[2]+5];
    }
}
