const softmax = (arr) => {

    if (arr.length == 1) {
        var out = new Float32Array(arr.length);
        out.fill(1)
        return out
    };

    let max = Math.max(...arr)
    var exp = new Float32Array(arr.length);
    var j = 0;
    for (var i of arr){
        exp[j] = Math.exp(i - max);
        j++;
    }

    let sum = exp.reduce((x, y) => {return x + y})
    return exp.map((x) => {return x / sum })
}

const sigmoid = (x) => {
    var out = (1 / (1 + Math.exp(-x)))
    return out
}



class FormatYoloV2 {
    constructor(image_size, num_classes, anchors, boxes_per_cell, data_format, threshold) {
        this.image_size = image_size;
        this.num_classes = num_classes;
        this.anchors = anchors;
        this.boxes_per_cell = boxes_per_cell
        this.data_format = data_format;
        this.threshold = threshold
    }
    get num_cell(){
        return [this.image_size[0] / 32, this.image_size[1] / 32];
    }


    convert_boxes_space_from_yolo_to_real(x, y, w, h, anchor, offset_y, offset_x){
        const image_size_h = this.image_size[0]
        const image_size_w = this.image_size[1]

        const num_cell_y = this.num_cell[0];
        const num_cell_x = this.num_cell[1];

        const anchor_w = anchor[0];
        const anchor_h = anchor[1];

        x = (x + offset_x) / num_cell_x
        y = (y + offset_y) / num_cell_y
        w = Math.exp(w) * anchor_w / num_cell_x
        h = Math.exp(h) * anchor_h / num_cell_y

        x = x * image_size_w
        y = y * image_size_h
        w = w * image_size_w
        h = h * image_size_h

        // center_x to x, center_y to y
        x = x - (w/2)
        y = y - (h/2)
        return [x, y, w, h]
    }

    run(array){
        var num_cell_y = this.num_cell[0];
        var num_cell_x = this.num_cell[1];

        if (array.length != num_cell_y * num_cell_x * this.boxes_per_cell * (this.num_classes + 5)){
            console.log(array.length, num_cell_y * num_cell_x * this.boxes_per_cell *  (this.num_classes + 5))
            console.error("error");
            return
        }
        var batch_size = 1;

        let output = new Array();

        for (var i = 0; i < num_cell_y; i++){
        for (var j = 0; j < num_cell_x; j++){
        for (var k = 0; k < this.boxes_per_cell; k++){
            var anchor = this.anchors[k]
            var index = i * num_cell_x * this.boxes_per_cell * (this.num_classes + 5) +
                j * this.boxes_per_cell * (this.num_classes + 5) + (k * (this.num_classes + 5));
            var next_index = index + this.num_classes + 5;

            var predictions = array.slice(index, next_index);
            var predict_classes = softmax(predictions.slice(0, this.num_classes));
            var predict_confidence = sigmoid(predictions[this.num_classes]);

            var predict_score = predict_classes.map((x) => {return x * predict_confidence})
            var max_score = Math.max(...predict_score)

            if (max_score < this.threshold){
                continue;
            }

            var x = sigmoid(predictions[this.num_classes+1]);
            var y = sigmoid(predictions[this.num_classes+2]);
            var w = predictions[this.num_classes+3];
            var h = predictions[this.num_classes+4];

            var box = this.convert_boxes_space_from_yolo_to_real(x, y, w, h, anchor, i, j)

            for (var class_id = 0; class_id < this.num_classes; class_id++){
                var result = box.concat([
                    class_id,
                    predict_score[class_id]
                ])
                output.push(result)
            }
        }
        }
        }
        console.log(output)
        console.log("format end")
        return output
    }

}



const iou = (box1, box2) => {

    // from x,y,w,h to x,y,right,bottom
    box1 = [
        box1[0],
        box1[1],
        box1[0] + box1[2],
        box1[1] + box1[3],
    ]
    box2 = [
        box2[0],
        box2[1],
        box2[0] + box2[2],
        box2[1] + box2[3],
    ]
    const inter_w = Math.min(box1[2], box2[2]) - Math.max(box1[0], box2[0]);
    const inter_h = Math.min(box1[3], box2[3]) - Math.max(box1[1], box2[1]);
    if (inter_w < 0 || inter_h < 0) {
        return 0;
    }
    const inter_square = inter_w * inter_h

    const square1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    const square2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    const union = square1 + square2 - inter_square
    const iou = inter_square / (union)

    return iou
}


class NMS {
    constructor(classes, iou_threshold, max_output_size, per_class=true){
        this.classes = classes
        this.iou_threshold = iou_threshold
        this.max_output_size = max_output_size
        this.per_class = per_class
    }

    nms(boxes){
        var sortedBoxes = boxes.sort((a, b) => b[5] - a[5])
        const selectedBoxes = [];

        sortedBoxes.forEach(box => {
            let add = true;
            for (let i=0; i < selectedBoxes.length; i++) {
                const curIou = iou(box, selectedBoxes[i]);
                if (curIou > this.iou_threshold) {
                    add = false;
                    break;
                }
            }
            if (add) {
                selectedBoxes.push(box);
            }
        });

        // console.log("selectedBoxes", selectedBoxes)
        return selectedBoxes
    }

    run(boxes){
        let output = []

        if (this.per_class){

            for (var class_id = 0; class_id < this.classes.length; class_id++){
                var class_masked = boxes.filter((box) => { return box[4] == class_id})
                var nms_boxes = this.nms(class_masked)
                for (var box of nms_boxes) {
                    output.push(box)
                }
            }
        }

        // console.log("nms output", output)
        return output
    }
}
