import tensorflow as tf
import functools


class QuantizeParamInit:
    """QuantizeParamInit is a mixin creating a quantized network class more convenient by providing two functions.
    1) __init___ : for initializing variables necessary for quantization, such as quantization function
    2) _quantized_variable_getter : for quantizing weight variables in convolutional layers

    This QuantizeParamInit class can be used for all kinds of task (classification, object detection, and segmentation).

    How to use
    1) Extend this mixin
    2) Set the custom_getter variable to `self.custom_getter` if its base class does not set it yet
       The setting is done by adding the base function defining custom_getter for tensorflow inside the class as follows
             def base(self, images, is_training):
                 with tf.variable_scope("", custom_getter=self.custom_getter):
                     return super().base(images, is_training)

    In the object construction step, users must pass
       * arguments for the quantization (e.g. activation_quantizer) followed by
       * arguments of the subtask class (e.g. weight_decay_rate)
       to the init function, this mixin will call init function of subtask's base class automatically.

    Requirement
    1) The base class must define `self.first_layer_name` and `self.last_layer_name` in the init function, where
       `self.first_layer_name` = prefix of the first convolutional layer and
       `self.last_layer_name` = prefix of the last convolutional layer

    Args:
        activation_quantizer (callable): An activation quantization function
        activation_quantizer_kwargs (dict): A dictionary of arguments for the activation quantization function
        weight_quantizer (callable): A weight quantization function
        weight_quantizer_kwargs (dict): A dictionary of arguments for the weight quantization function
        quantize_first_convolution (boolean): True if using quantization at the first layer, False otherwise
        quantize_last_convolution (boolean): True if using quantization at the last layer, False otherwise
        *args: argument for the initialization of subtask's network class
        **kwargs: keyword arguments for the initialization of subtask's network class
    """

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            quantize_first_convolution=None,
            quantize_last_convolution=None,
            *args,
            **kwargs
    ):
        super(QuantizeParamInit, self).__init__(*args, **kwargs)

        assert callable(weight_quantizer)
        assert callable(activation_quantizer)

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.activation = activation_quantizer(**activation_quantizer_kwargs)

        assert self.first_layer_name is not None
        assert self.last_layer_name is not None

        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization,
                                               quantize_first_convolution=quantize_first_convolution,
                                               quantize_last_convolution=quantize_last_convolution,
                                               first_layer_name=self.first_layer_name,
                                               last_layer_name=self.last_layer_name,)
        if quantize_last_convolution:
            self.before_last_activation = self.activation
        else:
            self.before_last_activation = lambda x: tf.nn.leaky_relu(x, alpha=0.1, name="leaky_relu")

    @staticmethod
    def _quantized_variable_getter(getter,
                                   name,
                                   weight_quantization=None,
                                   quantize_first_convolution=None,
                                   quantize_last_convolution=None,
                                   use_histogram=True,
                                   first_layer_name=None,
                                   last_layer_name=None,
                                   *args,
                                   **kwargs):
        """Get the quantized weight variables of convolutional layers (variables end with "kernel").
        The quantization of the first and last convolutional layers can be skipped by two variables,
        quantize_first_convolution and quantize_last_convolution, respectively.
        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            quantize_first_convolution (boolean): True if using quantization at the first layer, False otherwise
            quantize_last_convolution (boolean): True if using quantization at the last layer, False otherwise
            use_histogram (boolean): True to summarize tf.summary.histogram of quantized var before return
            first_layer_name (string): name of the first layer's variable_scope
            last_layer_name (string): name of the last layer's variable_scope
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)

        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                if quantize_first_convolution is not None and not quantize_first_convolution:
                    if var.op.name.startswith(first_layer_name):
                        return var

                if quantize_last_convolution is not None and not quantize_last_convolution:
                    if var.op.name.startswith(last_layer_name):
                        return var

                quantized_kernel = weight_quantization(var)
                if use_histogram:
                    tf.summary.histogram("quantized_kernel", quantized_kernel)

                return quantized_kernel

        return var
