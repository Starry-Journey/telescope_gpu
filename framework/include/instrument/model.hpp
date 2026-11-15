#pragma once

/// @file model.hpp
/// @brief Forward model, noise model and target extraction contracts.
///
/// Observation model: y = F(p) + noise.  A model may return images (phase
/// diversity), a vector (sensitivity/control), or both.  TargetExtractor turns
/// an Observable into the vector that sensitivity/control actually uses.

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "instrument/data.hpp"
#include "instrument/parameter.hpp"

namespace instrument {

struct Context {
    std::map<std::string, double> scalars;  // theta, temperature, ...
    std::string tag;
};

struct Observable {
    std::string kind;                 // "frames", "vector", "mixed"
    std::vector<Frame> frames;        // optional
    std::vector<double> values;       // optional target vector
    std::vector<std::string> names;   // names for values
};

class ForwardModel {
public:
    virtual ~ForwardModel() = default;
    virtual std::string name() const = 0;

    /// Evaluate the model for one parameter block.
    /// Implementations must be deterministic for a given context.
    virtual Observable evaluate(const ParameterBlock& params,
                                const Context& context) const = 0;

    /// Optional analytic Jacobian (rows = target names, cols = parameter
    /// values).  Returning false means "not available"; sensitivity then uses
    /// finite differences.
    virtual bool jacobian(const ParameterBlock&, const Context&,
                          std::vector<std::vector<double>>&) const {
        return false;
    }
};

struct NoiseModel {
    double read_noise_e = 5.0;
    double background_e = 200.0;
    double gain_e_per_adu = 1.0;
    double exposure_s = 1.0;

    /// Per-pixel variance for a signal in electrons (Poisson + read + dark).
    double variance(double signal_e, double dark_e_per_s = 0.0) const {
        const double s = signal_e < 0.0 ? 0.0 : signal_e;
        const double dark = dark_e_per_s * exposure_s;
        return s + dark + read_noise_e * read_noise_e;
    }
};

class TargetExtractor {
public:
    virtual ~TargetExtractor() = default;
    virtual std::string name() const = 0;
    /// Extract the target vector from an Observable.
    virtual Observable extract(const Observable& observable) const = 0;
};

/// Identity extractor: models that already return the target vector.
class IdentityExtractor : public TargetExtractor {
public:
    std::string name() const override { return "identity"; }
    Observable extract(const Observable& observable) const override { return observable; }
};

}  // namespace instrument
