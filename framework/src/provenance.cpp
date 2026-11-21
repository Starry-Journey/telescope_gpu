#include "instrument/provenance.hpp"

#include <iomanip>
#include <sstream>

namespace instrument {
namespace {
std::string q(const std::string& s) {
    std::ostringstream os;
    os << '"';
    for (char c : s) {
        if (c == '"' || c == '\\') os << '\\';
        os << c;
    }
    os << '"';
    return os.str();
}
}  // namespace

std::string RunManifest::to_json() const {
    std::ostringstream os;
    os << "{\n";
    os << "  \"run_id\": " << q(run_id) << ",\n";
    os << "  \"git_commit\": " << q(git_commit) << ",\n";
    os << "  \"git_dirty\": " << (git_dirty ? "true" : "false") << ",\n";
    os << "  \"config_hash\": " << q(config_hash) << ",\n";
    os << "  \"dataset_hash\": " << q(dataset_hash) << ",\n";
    os << "  \"backend\": " << q(backend) << ",\n";
    os << "  \"seed\": " << seed << ",\n";
    os << "  \"versions\": {";
    bool first = true;
    for (const auto& kv : versions) {
        if (!first) os << ", ";
        first = false;
        os << q(kv.first) << ": " << q(kv.second);
    }
    os << "},\n";
    os << "  \"metrics\": {";
    first = true;
    for (const auto& kv : metrics) {
        if (!first) os << ", ";
        first = false;
        os << q(kv.first) << ": " << std::setprecision(12) << kv.second;
    }
    os << "},\n";
    os << "  \"wall_time_s\": " << std::setprecision(12) << wall_time_s << "\n";
    os << "}\n";
    return os.str();
}

}  // namespace instrument
