#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

class ONNXModel {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session* session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<std::string> input_node_names;
    std::vector<std::string> output_node_names;
    std::vector<const char*> input_node_names_cstr;
    std::vector<const char*> output_node_names_cstr;
    std::vector<int64_t> input_node_dims;

public:
    ONNXModel(const std::string& model_path) : env(ORT_LOGGING_LEVEL_WARNING, "NexusML") {
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Convert string to wstring for Windows
        std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
        session = new Ort::Session(env, widestr.c_str(), session_options);

        // Extract input info
        size_t num_input_nodes = session->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            Ort::AllocatedStringPtr input_name_ptr = session->GetInputNameAllocated(i, allocator);
            input_node_names.push_back(input_name_ptr.get());
            Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_node_dims = tensor_info.GetShape();
        }

        // Extract output info
        size_t num_output_nodes = session->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            Ort::AllocatedStringPtr output_name_ptr = session->GetOutputNameAllocated(i, allocator);
            output_node_names.push_back(output_name_ptr.get());
        }

        for (const auto& str : input_node_names) input_node_names_cstr.push_back(str.c_str());
        for (const auto& str : output_node_names) output_node_names_cstr.push_back(str.c_str());
    }

    ~ONNXModel() {
        delete session;
    }

    std::vector<float> predict(const std::vector<std::vector<float>>& input_data) {
        if (input_data.empty()) return {};

        size_t batch_size = input_data.size();
        size_t feature_size = input_data[0].size();

        // Flatten input
        std::vector<float> input_tensor_values;
        input_tensor_values.reserve(batch_size * feature_size);
        for (const auto& row : input_data) {
            if (row.size() != feature_size) {
                throw std::runtime_error("Inconsistent feature size in batch.");
            }
            input_tensor_values.insert(input_tensor_values.end(), row.begin(), row.end());
        }

        std::vector<int64_t> input_shape = { (int64_t)batch_size, (int64_t)feature_size };

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size()
        );

        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_node_names_cstr.data(),
            &input_tensor,
            1,
            output_node_names_cstr.data(),
            1
        );

        float* floatarr = output_tensors.front().GetTensorMutableData<float>();
        size_t output_count = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
        
        std::vector<float> result(floatarr, floatarr + output_count);
        return result;
    }
};

void init_ml_inference(py::module_& m) {
    py::module_ sm = m.def_submodule("ml_inference", "ONNX Runtime ML Inference bindings");
    
    py::class_<ONNXModel>(sm, "ONNXModel")
        .def(py::init<const std::string&>())
        .def("predict", &ONNXModel::predict, "Run inference on a batch of inputs");
}
