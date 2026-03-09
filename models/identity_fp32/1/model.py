import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            output0 = pb_utils.Tensor("OUTPUT0", input0.as_numpy())
            responses.append(pb_utils.InferenceResponse([output0]))
        return responses
