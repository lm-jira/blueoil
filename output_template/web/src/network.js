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
