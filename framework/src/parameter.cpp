#include "instrument/parameter.hpp"

#include <algorithm>
#include <stdexcept>

namespace instrument {

ParameterSpace::ParameterSpace(std::vector<ParameterSpec> specs, std::string name)
    : specs_(std::move(specs)), name_(std::move(name)) {
    if (name_.empty()) name_ = "default";
    for (const auto& s : specs_) {
        if (!s.valid()) {
            throw std::invalid_argument("ParameterSpace: invalid spec '" + s.name + "'");
        }
    }
}

int ParameterSpace::index_of(const std::string& name) const {
    for (std::size_t i = 0; i < specs_.size(); ++i) {
        if (specs_[i].name == name) return static_cast<int>(i);
    }
    return -1;
}

std::vector<std::string> ParameterSpace::names() const {
    std::vector<std::string> out;
    out.reserve(specs_.size());
    for (const auto& s : specs_) out.push_back(s.name);
    return out;
}

std::vector<std::string> ParameterSpace::units() const {
    std::vector<std::string> out;
    out.reserve(specs_.size());
    for (const auto& s : specs_) out.push_back(s.unit);
    return out;
}

ParameterBlock ParameterSpace::encode(const std::vector<double>& values) const {
    if (values.size() != specs_.size()) {
        throw std::invalid_argument("ParameterSpace::encode: size mismatch");
    }
    return {name_, values};
}

std::vector<double> ParameterSpace::decode(const ParameterBlock& block) const {
    if (block.values.size() != specs_.size()) {
        throw std::invalid_argument("ParameterSpace::decode: size mismatch");
    }
    return block.values;
}

int ParameterSpace::clamp(ParameterBlock& block) const {
    if (block.values.size() != specs_.size()) {
        throw std::invalid_argument("ParameterSpace::clamp: size mismatch");
    }
    int changed = 0;
    for (std::size_t i = 0; i < specs_.size(); ++i) {
        const auto& s = specs_[i];
        const double v = block.values[i];
        const double c = std::max(s.lower, std::min(s.upper, v));
        if (c != v) {
            block.values[i] = c;
            ++changed;
        }
    }
    return changed;
}

Status ParameterSpace::validate(const ParameterBlock& block) const {
    if (block.values.size() != specs_.size()) {
        return Status::failure("parameter size mismatch");
    }
    for (std::size_t i = 0; i < specs_.size(); ++i) {
        const auto& s = specs_[i];
        if (!(s.lower <= block.values[i] && block.values[i] <= s.upper)) {
            return Status::failure("parameter out of bounds: " + s.name);
        }
    }
    return Status::success();
}

}  // namespace instrument
