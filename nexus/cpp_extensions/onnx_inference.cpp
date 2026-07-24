#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

struct CppONNXBrain {
    py::object session;
    py::object ort_module;
    std::string model_path;
    bool is_loaded;

    CppONNXBrain(const std::string& path, bool use_gpu = true) : model_path(path), is_loaded(false) {
        try {
            ort_module = py::module_::import("onnxruntime");
            py::list providers;
            if (use_gpu) {
                providers.append("CUDAExecutionProvider");
            }
            providers.append("CPUExecutionProvider");
            
            session = ort_module.attr("InferenceSession")(path, py::arg("providers") = providers);
            is_loaded = true;
        } catch (py::error_already_set& e) {
            // Silently fail to allow graceful degradation in python
            is_loaded = false;
        }
    }

    std::vector<float> predict(const std::vector<float>& features) {
        if (!is_loaded || features.empty()) return {0.0f};

        py::list inputs_meta = session.attr("get_inputs")();
        auto first_input = inputs_meta[0];
        auto input_name = first_input.attr("name");
        py::list shape = first_input.attr("shape").cast<py::list>();
        
        py::array_t<float> input_array;
        if (py::len(shape) == 3) {
            input_array = py::array_t<float>({1, 1, (int)features.size()}, features.data());
        } else {
            input_array = py::array_t<float>({1, (int)features.size()}, features.data());
        }
        
        py::dict inputs;
        inputs[input_name] = input_array;

        py::list outputs = session.attr("run")(py::none(), inputs);
        py::array_t<float> output_array = outputs[0].cast<py::array_t<float>>();
        
        auto buf = output_array.request();
        float* ptr = static_cast<float*>(buf.ptr);
        return std::vector<float>(ptr, ptr + buf.size);
    }
};

struct EnsembleFuser {
    std::vector<double> model_weights;
    std::vector<double> model_bias;

    EnsembleFuser() : model_weights{0.40, 0.35, 0.25}, model_bias{0.0, 0.0, 0.0} {}

    void set_weights(const std::vector<double>& w) {
        double total = std::accumulate(w.begin(), w.end(), 0.0);
        total = std::max(total, 1e-10);
        model_weights.clear();
        for (auto& v : w) model_weights.push_back(v / total);
    }

    double fuse(double xgb_score, double lstm_score, double transformer_score) {
        std::vector<double> scores = {xgb_score, lstm_score, transformer_score};
        double fused = 0.0;
        for (size_t i = 0; i < scores.size() && i < model_weights.size(); ++i) {
            fused += model_weights[i] * (scores[i] + model_bias[i]);
        }
        return std::tanh(fused * 1.2); // slight amplification
    }
    
    void recalibrate(const std::vector<double>& trailing_ics) {
        double total = 0.0;
        for (size_t i = 0; i < model_weights.size() && i < trailing_ics.size(); ++i) {
            double ic = std::max(0.0, trailing_ics[i]);
            model_weights[i] = 0.8 * model_weights[i] + 0.2 * (ic + 0.5);
            total += model_weights[i];
        }
        total = std::max(total, 1e-10);
        for (auto& w : model_weights) w /= total;
    }
};

void bind_onnx_inference(py::module_& m) {
    py::class_<CppONNXBrain>(m, "CppONNXBrain")
        .def(py::init<const std::string&, bool>(), 
             py::arg("model_path"), py::arg("use_gpu") = true)
        .def("predict", &CppONNXBrain::predict, py::arg("features"))
        .def_readonly("is_loaded", &CppONNXBrain::is_loaded);

    py::class_<EnsembleFuser>(m, "EnsembleFuser")
        .def(py::init<>())
        .def("set_weights", &EnsembleFuser::set_weights)
        .def("fuse", &EnsembleFuser::fuse, py::arg("xgb"), py::arg("lstm"), py::arg("transformer"))
        .def("recalibrate", &EnsembleFuser::recalibrate, py::arg("trailing_ics"))
        .def_readwrite("model_bias", &EnsembleFuser::model_bias);
}

PYBIND11_MODULE(nexus_onnx, m) {
    m.doc() = "Nexus ONNX Inference Module";
    bind_onnx_inference(m);
}